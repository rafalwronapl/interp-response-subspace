"""FIT-only nested feasibility and model freeze; zero native evaluations."""
import argparse,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from research_checks.gate import digest
from research_checks.persistence import immutable
from .common import ROOT,BASE,read,sha,unwrap,array,packed
from .models import select,fit,predict,loss

def main():
    cfg=read(BASE/'config.json');manifest=read(BASE/'input_manifest.json')['files'];out=BASE/'stage1_run'
    for name,h in manifest.items():
        if sha(ROOT/name)!=h:raise ValueError('Changed stage input '+name)
    if (out/'completed.json').exists():
        for name,h in unwrap(out/'completed.json')['files'].items():
            if sha(out/name)!=h:raise ValueError('Changed completed result')
        return
    index=read(BASE/'source_index.json');prior=dict(np.load(ROOT/cfg['prior']));U=prior['B5_U'].astype(float)
    if U.shape!=(1024,8) or not np.allclose(U.T@U,np.eye(8),atol=1e-5):raise ValueError('Historical U identity')
    def source(entry):
        receipt=unwrap(ROOT/entry['receipt']);cache=unwrap(ROOT/entry['cache'])
        if digest(cache['request'])!=receipt['request']:raise ValueError('Source request mismatch')
        if cache['request']['document']!=entry['document']:raise ValueError('Document mismatch')
        return cache['result']
    natural={i:source(e) for i,e in index['natural'].items()}
    xs=[];ys=[];blocks=[];groups=[];meta=[]
    for e in index['cells']:
        d=e['document'];n=natural[d['id']];s=n['state'];p=source(e['prefix']);t=source(e['target'])['state']
        if any(k in p for k in ['y','Y','logits']):raise ValueError('Target leakage in prefix')
        if e['prefix']['document']['split']=='TEST':raise ValueError('Forbidden TEST source')
        for k in ['x20','x21']:np.testing.assert_array_equal(array(p[k]),array(t[k]))
        dx20=array(p['x20'])-array(s['x20']);dx21=array(p['x21'])-array(s['x21'])
        xs.append(np.concatenate([dx20/prior['scale'][0],dx21/prior['scale'][1]])/np.sqrt(2.))
        ys.append(array(t['y'])-array(s['y']));blocks.append(d['block']);groups.append(d['split']);meta.append(dict(id=e['id'],block=d['block'],group=d['split'],branch=e['branch'],dose=e['dose']))
    x=np.asarray(xs);D=np.asarray(ys,float);block=np.asarray(blocks);group=np.asarray(groups)
    assert len(x)==384 and len(set(block))==24 and set(group)=={'FIT0','FIT1','FIT2'}
    P=np.asarray(x@prior['B5_W']@prior['B5_U'].T,np.float32).astype(float)
    A=(D-P)@U;targetU=D@U;outer=[];preds={k:np.zeros_like(D) for k in ['P','C','same_U','direct','mean']};preds['P']=P
    start=time.monotonic()
    with threadpool_limits(limits=2):
        for g in ['FIT0','FIT1','FIT2']:
            valid=group==g;train=~valid;record=dict(group=g,train_labels=int(train.sum()),validation_labels=int(valid.sum()))
            for name,target,linear in [('C',A,False),('same_U',targetU,True),('direct',D,False)]:
                param,cv=select(x[train],target[train],block[train],linear_only=linear)
                m=fit(x[train],target[train],param);raw=predict(m,x[valid])
                preds[name][valid]=P[valid]+raw@U.T if name=='C' else raw@U.T if name=='same_U' else raw
                record[name]=dict(param=param,inner_scores=cv)
            preds['mean'][valid]=P[valid]+A[train].mean(0)@U.T
            record['vector_L2']={k:loss(D[valid],v[valid],block[valid]) for k,v in preds.items()}
            record['coefficient_L2']={k:loss(A[valid],(v[valid]-P[valid])@U,block[valid]) for k,v in preds.items()}
            outer.append(record);immutable(out/f'outer_{g}.json',record)
        frozen={};final_selection={}
        for name,target,linear in [('C',A,False),('same_U',targetU,True),('direct',D,False)]:
            param,cv=select(x,target,block,linear_only=linear);m=fit(x,target,param)
            frozen[name]={k:packed(v) if isinstance(v,np.ndarray) else v for k,v in m.items()};final_selection[name]=dict(param=param,cv=cv)
    immutable(out/'MODELS.json',dict(models=frozen,mean_coefficients=packed(A.mean(0)),U=packed(U),historical_prior_sha256=sha(ROOT/cfg['prior']),training_new_labels=384,selection=final_selection))
    perblock={k:[loss(D[block==b],v[block==b],block[block==b]) for b in sorted(set(block))] for k,v in preds.items()}
    strata={}
    for branch in ['B','C']:
        for dose in [.5,1.]:
            mask=np.array([m['branch']==branch and m['dose']==dose for m in meta])
            strata[f'{branch}/{dose}']=dict(vector_L2={k:loss(D[mask],v[mask],block[mask]) for k,v in preds.items()},
                coefficient_L2={k:loss(A[mask],(v[mask]-P[mask])@U,block[mask]) for k,v in preds.items()})
    summary=dict(scope='Nested FIT-only coefficient/vector feasibility, no downstream readout measurements or fresh TEST claim',
        groups=outer,outer_vector_L2={k:loss(D,v,block) for k,v in preds.items()},outer_coefficient_L2={k:loss(A,(v-P)@U,block) for k,v in preds.items()},
        per_block_vector_L2=perblock,strata=strata,metadata=meta,selection=final_selection,new_training_labels=384,old_prior_extra_training=True,native_forward=0,native_backward=0)
    immutable(out/'SUMMARY.json',summary)
    immutable(out/'OUTER_PREDICTIONS.json',dict(D=packed(D),P=packed(P),U=packed(U),predictions={k:packed(v) for k,v in preds.items()},block=block.tolist(),group=group.tolist()))
    immutable(out/'completed.json',dict(files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='completed.json'},native_forward=0,native_backward=0,scope='stage1 only; stage2 not launched'))
    print({k:summary[k] for k in ['outer_vector_L2','outer_coefficient_L2']})
if __name__=='__main__':main()
