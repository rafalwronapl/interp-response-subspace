"""All vector/readout forecasts precede all intervened native MLP21 targets."""
import sys
import time
import traceback
import numpy as np
from .common import (ROOT,PREP,DATA,OUT,METHODS,Budget,Measurements,conditions,guard,status,
    read,sha,digest,payload,events,append,array_id,load_arrays,freeze,immutable,environment_identity)


class Operations:
    def __init__(self,stage,m):
        self.m=m;self.ops=read(PREP/(stage+'_operations.json'))['operations']
        self.lookup={self.key(o):o for o in self.ops}
        if len(self.lookup)!=len(self.ops):raise ValueError('Duplicate operation semantics')
    @staticmethod
    def key(o):return tuple(o[k] for k in ('role','document','branch','dose','method'))
    def run(self,role,doc,compute,branch=None,dose=None,method=None,**metadata):
        op=self.lookup[(role,doc['id'],branch,dose,method)]
        return self.m.run(op['id'],op['category'],doc,dict(operation=op,**metadata),compute)
    def get(self,role,doc,branch=None,dose=None,method=None):
        return self.m.get(self.lookup[(role,doc['id'],branch,dose,method)]['id'])


def contrast(state):return float(np.asarray(state['logits'],dtype=float)[1]-np.asarray(state['logits'],dtype=float)[0])

def invariant(a,b):
    for k in ('x19','x20','x21','earlier19','earlier20','earlier21'):
        np.testing.assert_array_equal(a[k],b[k])

def replacement(ref,delta):
    value=ref['Y'].copy()
    value[-1]=ref['y']+np.asarray(delta,dtype=np.float32)
    return value

def settings(doc,c,refs):
    ref=refs[doc['id']];donor=refs[doc['donor']]
    x=float(ref['X'])+c['dose']*(float(donor['X'])-float(ref['X']))
    return dict(x=x,m=float(ref['M']) if c['branch']=='C' else None)


def collect(sub,atlas,op,docs,parameters):
    from .prediction import predict
    refs={};features={}
    for i,doc in enumerate(docs):
        ref=op.run('natural_full',doc,lambda:sub.forward(doc),settings={})
        np.testing.assert_array_equal(ref['Y'][-1],ref['y'])
        refs[doc['id']]=ref
        features[doc['id']]=op.run('natural_features',doc,lambda:atlas.evaluate(ref),input=array_id(ref))
        status(op.m.stage+'_naturals',completed=i+1,total=len(docs))
    data={k:[] for k in ('dx20','dx21','P','B6','cell','block','branch','dose','frame','S','A','Fnat')};rows=[]
    for i,doc in enumerate(docs):
        ref=refs[doc['id']]
        for c in conditions(doc):
            kw=settings(doc,c,refs)
            prefix=op.run('prefix_before_MLP21',doc,lambda:sub.forward(doc,prefix=True,**kw),c['branch'],c['dose'],
                settings=kw,reference=array_id(ref),donor=array_id(refs[doc['donor']]))
            np.testing.assert_array_equal(prefix['x19'],ref['x19'])
            for layer in (19,20,21):np.testing.assert_array_equal(prefix['earlier'+str(layer)],ref['earlier'+str(layer)])
            ct=op.run('delta_features_decode',doc,lambda:atlas.evaluate(prefix,features[doc['id']]),c['branch'],c['dose'],
                input=array_id(prefix),reference=array_id(features[doc['id']]))
            dx=prefix['x21']-ref['x21']
            jvp=op.run('native_local',doc,lambda:dict(pred=sub.jvp(ref['x21'],dx)),c['branch'],c['dose'],
                input=array_id({'x':ref['x21'],'dx':dx}),dtype='float32')['pred']
            values=dict(dx20=prefix['x20']-ref['x20'],dx21=dx,P=ct['p'],B6=jvp,
                cell=4*int(c['branch']=='C')+2*int(c['dose']==1)+doc['S'],block=doc['block'],
                branch=int(c['branch']=='C'),dose=c['dose'],frame=doc['frame'],S=doc['S'],A=doc['A'],Fnat=contrast(ref))
            for k,v in values.items():data[k].append(v)
            rows.append(dict(document=doc['id'],**c))
        status(op.m.stage+'_prefixes',completed=i+1,total=len(docs))
    data={k:np.asarray(v) for k,v in data.items()}
    spent=sum(e['seconds'] for e in events(op.m.folder/'offline.jsonl'))
    if spent>=1200:raise RuntimeError('Offline prediction time cap')
    start=time.monotonic()
    vectors=predict(parameters,{k:data[k] for k in ('dx20','dx21','P','cell','B6')})
    elapsed=time.monotonic()-start
    append(op.m.folder/'offline.jsonl',dict(role='frozen_parameter_prediction',seconds=elapsed,ok=True))
    if spent+elapsed>1200:raise RuntimeError('Offline prediction time cap after computation')
    return refs,data,vectors,rows


def readouts(sub,op,docs,refs,vectors,rows):
    result={k:[] for k in METHODS};lookup={d['id']:d for d in docs}
    for i,row in enumerate(rows):
        doc=lookup[row['document']];ref=refs[doc['id']];kw=settings(doc,row,refs)
        prefix=op.get('prefix_before_MLP21',doc,row['branch'],row['dose'])
        for method in METHODS:
            y=replacement(ref,vectors[method][i])
            state=op.run('forecast_full_bypass_MLP21',doc,lambda:sub.forward(doc,replacement=y,**kw),row['branch'],row['dose'],method,
                forecast_readout=True,settings=kw,replacement=array_id({'Y':y}),reference=array_id(ref),prefix=array_id(prefix))
            invariant(prefix,state)
            np.testing.assert_array_equal(state['Y'],y)
            np.testing.assert_array_equal(state['Y'][:-1],ref['Y'][:-1])
            result[method].append(contrast(state))
        if i%8==0:status(op.m.stage+'_forecast_readouts',completed=i+1,total=len(rows))
    return {k:np.asarray(v) for k,v in result.items()}


def targets(sub,op,docs,refs,rows):
    lookup={d['id']:d for d in docs};result=dict(Ftrue=[],D=[])
    for i,row in enumerate(rows):
        doc=lookup[row['document']];ref=refs[doc['id']];kw=settings(doc,row,refs)
        state=op.run('true_full',doc,lambda:sub.forward(doc,**kw),row['branch'],row['dose'],
            intervention_target=True,settings=kw,reference=array_id(ref))
        invariant(op.get('prefix_before_MLP21',doc,row['branch'],row['dose']),state)
        np.testing.assert_array_equal(state['Y'][:-1],ref['Y'][:-1])
        result['Ftrue'].append(contrast(state));result['D'].append(state['y']-ref['y'])
        if i%8==0:status(op.m.stage+'_targets',completed=i+1,total=len(rows))
    return {k:np.asarray(v) for k,v in result.items()}


def controls(sub,op,docs,refs,data):
    import torch
    pilot=op.m.stage=='pilot';errors=[1e-6];positive=[];fds=[];records=[]
    selected=docs if pilot else [next(d for d in docs if d['block']==b and d['frame']==f)
        for b in sorted({d['block'] for d in docs}) for f in (0,1,2)]
    def identical(name,a,b):
        error=abs(contrast(a)-contrast(b));errors.append(error)
        if error>1e-4:raise AssertionError('Contrast control tolerance '+name)
        for k in ('x19','x20','x21','Y','y','logits'):
            np.testing.assert_allclose(a[k],b[k],rtol=1e-5,atol=1e-6)
        records.append(dict(control=name,contrast_error=error))
    for doc in selected:
        ref=refs[doc['id']]
        common=dict(intervention_target=True,reference=array_id(ref))
        repeated=op.run('repeat_natural',doc,lambda:sub.forward(doc),**common)
        identical('repeat_'+doc['id'],repeated,ref)
        for b in ('B','C') if pilot else (('B' if (doc['block']+doc['frame'])%2==0 else 'C'),):
            kw=dict(x=float(ref['X']),m=float(ref['M']) if b=='C' else None)
            sham=op.run('zero_sham',doc,lambda:sub.forward(doc,replacement=ref['Y'],**kw),b,0.,settings=kw,replacement=array_id({'Y':ref['Y']}),**common)
            identical('sham_'+b+'_'+doc['id'],sham,ref)
        if pilot:
            neg=op.run('negative_after_MLP21',doc,lambda:sub.forward(doc,negative=True),**common)
            for k in ('x19','x20','x21','earlier19','earlier20','earlier21'):
                np.testing.assert_array_equal(neg[k],ref[k])
            np.testing.assert_array_equal(neg['Y'][:-1],ref['Y'][:-1])
            values=[];direction=sub.readout_direction()
            for sign,role in ((1,'positive_plus'),(-1,'positive_minus')):
                y=replacement(ref,np.float32(sign*.1)*direction)
                pos=op.run(role,doc,lambda:sub.forward(doc,replacement=y),replacement=array_id({'Y':y}),**common)
                values.append(abs(contrast(pos)-contrast(ref)))
            positive.append(dict(document=doc['id'],frame=doc['frame'],changes=values))
            poisonprefix=op.run('poison_prefix',doc,lambda:sub.forward(doc,prefix=True,poison=True,x=float(ref['X'])+.1),**common)
            if 'y' in poisonprefix or 'logits' in poisonprefix:raise AssertionError('Poison prefix target leak')
            poison=op.run('poison_replacement',doc,lambda:sub.forward(doc,replacement=ref['Y'],poison=True),replacement=array_id({'Y':ref['Y']}),**common)
            identical('poison_replacement_'+doc['id'],poison,ref)
            prefix=op.get('prefix_before_MLP21',doc,'B',1.)
            dx=prefix['x21'].astype(float)-ref['x21'].astype(float)
            j=op.run('validate_B_d1',doc,lambda:dict(pred=sub.jvp(ref['x21'],dx,torch.float64)),'B',1.,
                input=array_id({'x':ref['x21'],'dx':dx}),dtype='float64',**common)['pred']
            fd_errors=[]
            for h in (.01,.001,.0001):
                ys=[]
                for sign in (-1,1):
                    x=ref['x21'].astype(float)+sign*h*dx
                    ys.append(op.run(f'FD_h{h}_sign{sign}',doc,lambda:dict(y=sub.local(x,torch.float64,tokens=len(doc['ids']))),'B',1.,
                        input=array_id({'x':x}),dtype='float64',**common)['y'])
                fd_errors.append(float(np.linalg.norm((ys[1]-ys[0])/(2*h)-j)))
            tolerance=max(1e-6,1e-3*float(np.linalg.norm(j)))
            j32=op.get('native_local',doc,'B',1.)['pred']
            j32_error=float(np.linalg.norm(j32.astype(float)-j))
            fds.append(dict(document=doc['id'],errors=fd_errors,tolerance=tolerance,
                jvp32_vs_64_error=j32_error,
                ok=j32_error<=tolerance and any(fd_errors[i]<=tolerance and fd_errors[i+1]<=tolerance for i in (0,1))))
    oracle_docs=[d for f in (0,1,2) for d in [x for x in docs if x['frame']==f][:2]] if pilot else selected
    for doc in oracle_docs:
        ref=refs[doc['id']]
        for b in ('B','C'):
            for dose in (.5,1.) if pilot else (1.,):
                c=dict(branch=b,dose=dose);kw=settings(doc,c,refs)
                truth=op.get('true_full',doc,b,dose)
                # Mirror the actual numerical y0 + (ytrue-y0) pathway.
                y=replacement(ref,truth['y']-ref['y'])
                oracle=op.run('oracle',doc,lambda:sub.forward(doc,replacement=y,**kw),b,dose,
                    intervention_target=True,settings=kw,replacement=array_id({'Y':y}),truth=array_id(truth))
                identical('oracle_'+b+'_'+str(dose)+'_'+doc['id'],oracle,truth)
    epsilon=max(errors)
    if not pilot:epsilon=max(epsilon,payload(OUT/'pilot/checks.json')['epsilon_pilot'])
    from .analysis import signal
    sig=signal(data,epsilon,1 if pilot else 9)
    pcounts={str(frame):sum(max(p['changes'])>100*epsilon for p in positive if p['frame']==frame) for frame in (0,1,2)}
    technical=(not pilot or (all(n>=3 for n in pcounts.values()) and all(f['ok'] for f in fds)))
    return dict(ready=bool(technical and sig['ready']),technical_controls_ok=bool(technical),
        **({'epsilon_pilot':epsilon} if pilot else {'epsilon_main':epsilon}),signal=sig,
        identical_controls=records,positive=positive,positive_passed=pcounts,finite_difference=fds,
        stage_scope='Pilot bound/decision remain immutable; MAIN uses max of pilot bound and its own control errors.')


def execute(stage,sub,atlas,m,docs,parameters,interrupt=None):
    from .analysis import assess
    folder=m.folder;op=Operations(stage,m)
    refs,data,vectors,rows=collect(sub,atlas,op,docs,parameters)
    freeze(folder,'inputs',data,dict(rows=rows,identity_sha256=digest(m.identity)))
    freeze(folder,'vectors',vectors,dict(rows=rows,input_sha256=sha(folder/'inputs.json'),parameters_sha256=m.identity['parameter_sha256']))
    if interrupt:interrupt('vectors')
    forecasts=readouts(sub,op,docs,refs,vectors,rows)
    freeze(folder,'forecasts',forecasts,dict(rows=rows,vectors_sha256=sha(folder/'vectors.json')))
    if interrupt:interrupt('readouts')
    truth=targets(sub,op,docs,refs,rows)
    freeze(folder,'targets',truth,dict(rows=rows,forecasts_sha256=sha(folder/'forecasts.json')))
    merged=dict(data,**forecasts,**truth)
    checks=controls(sub,op,docs,refs,merged)
    immutable(folder/'checks.json',checks)
    if stage=='main' and not (folder/'results.json').exists():
        start=time.monotonic();spent=sum(e['seconds'] for e in events(folder/'offline.jsonl'))
        if spent>=1200:raise RuntimeError('Offline cap')
        result=assess(merged,checks['epsilon_main'],checks['technical_controls_ok'])
        from .analysis import vector_diagnostics
        diagnostics,vector_groups=vector_diagnostics(merged,vectors)
        result['vector_diagnostics']=diagnostics
        result['vector_breakdown']=vector_groups
        seconds=time.monotonic()-start;append(folder/'offline.jsonl',dict(seconds=seconds,ok=True))
        if seconds+spent>1200:raise RuntimeError('Offline cap after analysis')
        immutable(folder/'results.json',result)
    if m.budget.summary()['counts']!=read(PREP/(stage+'_operations.json'))['counts']:
        raise ValueError('Not all scheduled operations completed or unexpected retries; audit required')
    immutable(folder/'complete.json',dict(stage=stage,documents=len(docs),conditions=len(rows),ledger=m.budget.summary(),ready=checks['ready']))
    status(stage+'_complete',ready=checks['ready'],ledger=m.budget.summary())


def run():
    import torch
    from e031.loading import load
    from .subject import Subject,Atlas
    from threadpoolctl import threadpool_limits
    stage=sys.argv[1]
    if stage not in ('pilot','main'):raise ValueError('Stage')
    identity=read(PREP/'identity.json')
    if digest(environment_identity())!=identity['environment_sha256']:raise ValueError('Environment changed')
    if stage=='main' and not payload(OUT/'pilot/checks.json')['ready']:raise ValueError('Pilot gate closed')
    guard(True);torch.set_num_threads(4);docs=[d for d in read(DATA)['documents'] if d['split']==stage]
    status('loading_'+stage)
    try:
        start=time.monotonic();model=load(ROOT/'model');sub=Subject(model);atlas=Atlas()
        parameters=load_arrays(payload(ROOT/'runs/E036/development/predictors.json')['arrays'])
        budget=Budget(stage);m=Measurements(stage,budget)
        append(m.folder/'loading.jsonl',dict(seconds=time.monotonic()-start,resources=guard(),excluded_from_model_seconds=True))
        try:
            with threadpool_limits(limits=4):execute(stage,sub,atlas,m,docs,parameters)
        finally:sub.close()
    except Exception as exc:
        append(OUT/'failures.jsonl',dict(stage=stage,error=repr(exc),traceback=traceback.format_exc()))
        status('checkpointed',failed_stage=stage,error=repr(exc));raise

if __name__=='__main__':run()
