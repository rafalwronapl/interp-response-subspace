"""Reviewed stages only. TEST targets are structurally downstream of forecast freeze."""
import sys
import time
import traceback
import numpy as np
from research_checks.gate import read,digest,environment_identity
from research_checks.persistence import immutable
from .common import ROOT,PREP,OUT,OFFLINE,Budget,Measurements,guard,status,payload,array_id,save_arrays,load_arrays,append
from .data import conditions

def natural(sub,atlas,measure,docs):
    references={};features={}
    for i,doc in enumerate(docs):
        ref=measure.run(doc['id']+'_natural','network',doc,dict(mode='full',natural=True),lambda doc=doc:sub.forward(doc))
        enc=measure.run(doc['id']+'_natural_clt','clt',doc,dict(inputs=array_id(ref)),lambda ref=ref:atlas.evaluate(ref))
        references[doc['id']]=ref;features[doc['id']]=enc
        status(measure.stage+'_references',completed=i+1,total=len(docs))
    return references,features

def target(sub,measure,doc,c,prefix):
    return measure.run(c['id']+'_target','target',doc,dict(input=array_id({'x21':prefix['x21']}),tokens=len(doc['ids'])),lambda:dict(y=sub.local(prefix['x21'],tokens=len(doc['ids']))))['y']

def collect(sub,atlas,measure,docs,refs,features,evaluate_targets):
    rows=[];vectors={k:[] for k in ('dx20','dx21','P','B6','block','branch','dose','cell')}
    if evaluate_targets:vectors['D']=[]
    for i,doc in enumerate(docs):
        ref=refs[doc['id']];donor=refs[doc['donor']]
        for c in conditions(doc):
            x=float(ref['X'])+c['dose']*(float(donor['X'])-float(ref['X']))
            m=float(ref['M']) if c['branch']=='C' else None
            settings=dict(x=x,m=m,prefix=True,reference=array_id(ref),donor=array_id(donor),operator=c)
            prefix=measure.run(c['id']+'_prefix','network',doc,settings,lambda:sub.forward(doc,prefix=True,x=x,m=m))
            np.testing.assert_array_equal(prefix['x19'],ref['x19'])
            for l in (19,20,21):np.testing.assert_array_equal(prefix['earlier'+str(l)],ref['earlier'+str(l)])
            clt=measure.run(c['id']+'_clt','clt',doc,dict(inputs=array_id(prefix),reference=array_id(features[doc['id']])),lambda:atlas.evaluate(prefix,features[doc['id']]))
            dx=prefix['x21']-ref['x21']
            jvp=measure.run(c['id']+'_jvp','jvp',doc,dict(x=array_id({'x':ref['x21'],'dx':dx}),dtype='float32'),lambda:dict(pred=sub.jvp(ref['x21'],dx)))['pred']
            for k,v in [('dx20',prefix['x20']-ref['x20']),('dx21',dx),('P',clt['p']),('B6',jvp),('block',doc['block']),('branch',int(c['branch']=='C')),('dose',c['dose']),('cell',int(c['branch']=='C')*4+int(c['dose']==1.)*2+doc['S'])]:vectors[k].append(v)
            if evaluate_targets:vectors['D'].append(target(sub,measure,doc,c,prefix)-ref['y'])
            rows.append(dict(document=doc['id'],**c))
        status(measure.stage+'_prefixes',completed=i+1,total=len(docs))
    return {k:np.asarray(v) for k,v in vectors.items()},rows

def pilot_controls(sub,measure,docs,refs,data):
    import torch
    epsilons=[];positive=[];fd=[];arithmetic_drift=[]
    for doc in docs:
        ref=refs[doc['id']];donor=refs[doc['donor']];x=float(donor['X'])
        def run(suffix,**kw):
            return measure.run(doc['id']+'_'+suffix,'network',doc,dict(settings=kw,reference=array_id(ref)),lambda:sub.forward(doc,**kw))
        sham=run('sham',x=float(ref['X']),m=float(ref['M']))
        for k in ('x19','x20','x21','y','logits'):
            np.testing.assert_array_equal(sham[k],ref[k]);epsilons.append(float(np.linalg.norm(sham[k]-ref[k])))
        post=run('post',x=x,style='post')
        pre=measure.get(doc['id']+'_B_one_prefix')
        native_full=run('native_full',x=x)
        for k in ('x20','x21'):np.testing.assert_array_equal(native_full[k],pre[k])
        corrected=measure.get(doc['id']+'_B_one_target')['y']
        np.testing.assert_allclose(native_full['y'],corrected,rtol=1e-5,atol=1e-6)
        epsilons.append(float(np.linalg.norm(native_full['y']-corrected)))
        for k in ('x20','x21'):
            arithmetic_drift.append(dict(document=doc['id'],state=k,L2=float(np.linalg.norm(post[k]-pre[k])),max_abs=float(np.max(np.abs(post[k]-pre[k])))))
        neg=run('negative',negative=True)
        for k in ('x19','x20','x21','y'):np.testing.assert_array_equal(neg[k],ref[k])
        forced=run('force',x=float(ref['X'])+.1)
        forceprefix=run('forceprefix',x=float(ref['X'])+.1,prefix=True)
        np.testing.assert_array_equal(forced['x21'],forceprefix['x21'])
        positive.append(dict(dx=float(np.linalg.norm(forced['x21']-ref['x21'])),dy=float(np.linalg.norm(forced['y']-ref['y']))))
        mlp=sub.model.model.layers[21].mlp;original=mlp.forward
        def forbidden(*args,**kw):raise AssertionError('TEST prefix evaluated target')
        try:
            mlp.forward=forbidden
            poison=run('poison',prefix=True)
            if 'y' in poison:raise AssertionError('Leak')
        finally:mlp.forward=original
        local=measure.run(doc['id']+'_local','target',doc,dict(input=array_id({'x':post['x21']}),tokens=len(doc['ids'])),lambda:dict(y=sub.local(post['x21'],tokens=len(doc['ids']))))['y']
        np.testing.assert_allclose(local,post['y'],rtol=1e-5,atol=1e-6)
        epsilons.append(float(np.linalg.norm(local-post['y'])))
        delta=pre['x21'].astype(float)-ref['x21'].astype(float)
        j=measure.run(doc['id']+'_jvp64','jvp',doc,dict(input=array_id({'x':ref['x21'],'dx':delta}),dtype='float64'),lambda:dict(pred=sub.jvp(ref['x21'],delta,torch.float64)))['pred']
        errors=[]
        for i,h in enumerate((1e-2,1e-3,1e-4)):
            vals=[]
            for sign,suffix in ((1,'p'),(-1,'m')):
                xx=ref['x21'].astype(float)+sign*h*delta
                vals.append(measure.run(doc['id']+f'_fd{i}_{suffix}','target',doc,dict(input=array_id({'x':xx}),dtype='float64',tokens=len(doc['ids'])),lambda xx=xx:dict(y=sub.local(xx,torch.float64,tokens=len(doc['ids']))))['y'])
            errors.append(float(np.linalg.norm((vals[0]-vals[1])/(2*h)-j)))
        tolerance=max(1e-6,1e-3*float(np.linalg.norm(j)))
        if not any(errors[i]<=tolerance and errors[i+1]<=tolerance for i in (0,1)):raise ValueError('FP64 derivative gate')
        fd.append(dict(document=doc['id'],errors=errors,tolerance=tolerance))
    epsilon=max([1e-7]+epsilons)
    adequate=[p['dx']>100*epsilon and p['dy']>100*epsilon for p in positive[:4]]
    if sum(adequate)<3:raise ValueError('Positive sensitivity gate')
    return dict(ready=True,epsilon_num=epsilon,positive_controls=positive,first_four_adequate=adequate,FD=fd,
                local_full_rtol=1e-5,local_full_atol=1e-6,sham_exact=True,prefix_poison_passed=True,
                source_le19_encoder_inputs_invariant=True,decoder_bias_cancels_algebraically=True,alternative_arithmetic_drift=arithmetic_drift,
                native_PRE_full_prefix_inputs_exact=True)

def freeze_dataset(stage,data,rows):
    receipt=OUT/stage/'data.json'
    if receipt.exists():
        old=payload(receipt);loaded=load_arrays(old['arrays'])
        if old['rows']!=rows or array_id(loaded)!=array_id(data):raise ValueError('Stage data changed')
        return old
    arrays=save_arrays(OUT/stage,'data',data)
    return immutable(receipt,dict(arrays=arrays,rows=rows))['payload']

def offline(stage,compute,label):
    from threadpoolctl import threadpool_limits
    from .common import events
    log=OUT/stage/'offline.jsonl'
    spent=sum(e['seconds'] for e in events(log))
    if spent>=OFFLINE[stage]:raise RuntimeError('Offline cap before operation')
    start=time.monotonic()
    ok=False
    try:
        with threadpool_limits(limits=4):result=compute()
        ok=True;return result
    finally:
        seconds=time.monotonic()-start
        append(log,dict(seconds=seconds,label=label,ok=ok,cap=OFFLINE[stage]))
        if spent+seconds>OFFLINE[stage]:raise RuntimeError('Offline fit/prediction/analysis cap')

def execute(stage,sub,atlas,measure,docs):
    from .learning import fit,predict,signal,assess
    refs,features=natural(sub,atlas,measure,docs)
    data,rows=collect(sub,atlas,measure,docs,refs,features,stage!='test')
    if stage=='pilot':
        checks=pilot_controls(sub,measure,docs,refs,data)
        checks['ledger']=measure.budget.summary()
        receipt=OUT/stage/'checks_receipt.json'
        if receipt.exists():
            original=payload(receipt)
            if {k:v for k,v in original.items() if k!='resources'}!=checks:raise ValueError('Pilot technical evidence changed')
            checks=original
        else:
            checks['resources']=guard();immutable(receipt,checks)
        from .prepare import write
        if not (OUT/stage/'checks.json').exists():write(OUT/stage/'checks.json',checks)
        elif read(OUT/stage/'checks.json')!=checks:raise ValueError('Pilot checks changed')
        freeze_dataset(stage,data,rows)
    elif stage=='development':
        freeze_dataset(stage,data,rows)
        masks={s:np.asarray([r['document'] in {d['id'] for d in docs if d['split']==s} for r in rows]) for s in ('fit','select')}
        parts={s:{k:v[mask] for k,v in data.items()} for s,mask in masks.items()}
        if not (OUT/stage/'predictors.json').exists():
            parameters,meta=offline(stage,lambda:fit(parts['fit'],parts['select']),'fit')
            record=save_arrays(OUT/stage,'predictors',parameters)
            immutable(OUT/stage/'predictors.json',dict(arrays=record,selection=meta,data_sha256=payload(OUT/stage/'data.json')['arrays']['sha256']))
        fitted=payload(OUT/stage/'predictors.json');load_arrays(fitted['arrays'])
        if fitted['data_sha256']!=payload(OUT/stage/'data.json')['arrays']['sha256']:raise ValueError('Predictor data provenance changed')
        eps=read(OUT/'pilot/checks.json')['epsilon_num'];sig=signal(parts['select'],eps,3)
        from .prepare import write
        checks=dict(select_signal=sig['ready'],details=sig,predictors_frozen=True)
        if not (OUT/stage/'checks.json').exists():write(OUT/stage/'checks.json',checks)
        elif read(OUT/stage/'checks.json')!=checks:raise ValueError('Development checks changed')
    else:
        # All natural references, prefixes, CLT and JVP are obtained without any TEST local target.
        freeze_dataset(stage,data,rows)
        fitted=payload(OUT/'development/predictors.json');parameters=load_arrays(fitted['arrays'])
        if not (OUT/stage/'forecasts.json').exists():
            allowed={k:data[k] for k in ('dx20','dx21','P','cell','B6')}
            forecasts=offline(stage,lambda:predict(parameters,allowed),'forecast')
            arrays=save_arrays(OUT/stage,'forecasts',forecasts)
            immutable(OUT/stage/'forecasts.json',dict(arrays=arrays,rows=len(rows),parameters_sha256=fitted['arrays']['sha256'],
                input_sha256=payload(OUT/stage/'data.json')['arrays']['sha256'],ledger_before_targets=measure.budget.summary()))
        forecasts=load_arrays(payload(OUT/stage/'forecasts.json')['arrays'])
        targets=[];lookup={d['id']:d for d in docs}
        for i,row in enumerate(rows):
            doc=lookup[row['document']];prefix=measure.get(row['id']+'_prefix')
            targets.append(target(sub,measure,doc,row,prefix)-refs[doc['id']]['y'])
            status('test_targets',completed=i+1,total=len(rows))
        data['D']=np.asarray(targets)
        target_receipt=OUT/stage/'targets.json'
        if not target_receipt.exists():immutable(target_receipt,dict(arrays=save_arrays(OUT/stage,'targets',dict(D=data['D']))))
        if not (OUT/stage/'results.json').exists():
            results=offline(stage,lambda:assess(data,forecasts,read(OUT/'pilot/checks.json')['epsilon_num'],fitted['selection']['candidate']['rank'],parameters['fit_U']),'analysis')
            immutable(OUT/stage/'results.json',results)
    immutable(OUT/stage/'complete.json',dict(stage=stage,documents=len(docs),ledger=measure.budget.summary()))
    status(stage+'_complete',ledger=measure.budget.summary())

def run():
    import torch
    from e031.loading import load
    from .subject import Subject,Atlas
    stage=sys.argv[1]
    if stage not in ('pilot','development','test'):raise ValueError('Stage')
    if digest(environment_identity())!=read(PREP/'identity.json')['environment_sha256']:raise ValueError('Environment changed')
    guard(start=True);torch.set_num_threads(4)
    splits={'pilot':['pilot'],'development':['fit','select'],'test':['test']}[stage]
    docs=[d for d in read(PREP/'dataset.json')['documents'] if d['split'] in splits]
    status('loading_'+stage)
    try:
        start=time.monotonic()
        model=load(ROOT/'model');sub=Subject(model);atlas=Atlas()
        repair=read(PREP/'identity.json').get('pilot_repair') if stage=='pilot' else None
        folder=ROOT/repair['new_cache'] if repair else OUT/stage
        carry=[ROOT/p for p in repair['carry_ledgers']] if repair and repair.get('carry_ledgers') else ROOT/repair['carry_ledger'] if repair else None
        budget=Budget(stage,folder=folder,carry_ledger=carry)
        measure=Measurements(stage,budget,folder=folder)
        append(folder/'loading.jsonl',dict(seconds=time.monotonic()-start,resources=guard(),excluded_from_model_operation_seconds=True))
        try:execute(stage,sub,atlas,measure,docs)
        finally:sub.close()
    except Exception as exc:
        append(OUT/'failures.jsonl',dict(stage=stage,error=repr(exc),traceback=traceback.format_exc()))
        status('checkpointed',failed_stage=stage,error=repr(exc));raise

if __name__=='__main__':run()
