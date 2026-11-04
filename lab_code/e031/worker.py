"""Natural-only derivative forecasts, whole-stage freeze, then fixed finite differences/outcomes."""
import sys
import numpy as np
from research_checks.gate import read,digest,environment_identity
from research_checks.persistence import immutable
from e012.common import append,now,rows
from e012.common import sha
from e024.common import payload
from .common import ROOT,OUT,PREP,CONFIG,Measurements,Budget,guard,status,donor
from .protocol import predictions,summarize

H=(.05,.025,.0125)
HLABELS=('050','025','0125')


def historical(stage,d,kind,rec,arrays=None):
    if stage!='pilot':return
    label=read(PREP/'historical_map.json')[d['id']][kind]
    original=read(ROOT/label)
    if max(abs(np.asarray(rec['scores'])-original['scores']))>1e-5:raise RuntimeError('E014 reproduction outside frozen tolerance')
    if kind=='natural':
        if arrays is None:raise ValueError('Missing natural-state comparison')
        path=ROOT/'runs/E014'/original['cache']
        if sha(path)!=original['cache_sha256']:raise ValueError('Historical natural cache changed')
        with np.load(path,allow_pickle=False) as z:
            for u,index in [('U',0),('W',2)]:
                for suffix,key in [('', 'units_z'),('_gate','units_gate'),('_up','units_up')]:
                    np.testing.assert_allclose(arrays[u+suffix],z[key][index],rtol=0,atol=1e-5)
            np.testing.assert_allclose(arrays['MLP20_after'],z['M20'],rtol=0,atol=1e-5)


def forecast(m,stage,docs):
    coefs=read(PREP/'frozen_baselines.json')['coefficients'];packet={}
    for d in docs:
        prefix=f'{stage}_{d["id"]}';natural,ref=m.indexed(prefix+'_natural');_,other=m.indexed(f'{stage}_{donor(d,docs)["id"]}_natural')
        delta=float(other['U'])-float(ref['U']);branches={}
        for branch in ('b','c'):
            rec,arr=m.run(d,prefix+'_'+branch+'_zero',reference=ref,t=0.,delta=delta,branch=branch,derivatives=True)
            np.testing.assert_array_equal(rec['scores'],natural['scores'])
            np.testing.assert_array_equal(arr['final_hidden'],ref['final_hidden']);branches[branch]=rec
        q=2*d['S']-1;cf=coefs[d['frame']+'_S'+str(d['S'])]
        pp={str(dose):predictions(q,delta,dose,branches['b']['prime'],branches['c']['prime'],branches['b']['second'],branches['c']['second'],cf['mu'],cf['beta']) for dose in (.5,1.)}
        packet[d['id']]=dict(delta=delta,derivatives={b:{k:r[k] for k in ('prime','second','margin')} for b,r in branches.items()},predictions=pp)
    return packet


def controls(m,stage,docs,ids,packet):
    checks=[];precision_pairs=[True,True]
    for d in docs:
        if d['id'] not in ids:continue
        prefix=f'{stage}_{d["id"]}';natural,ref=m.indexed(prefix+'_natural');delta=packet[d['id']]['delta']
        probes={}
        for label,settings in [('sham_U',dict(U_override=float(ref['U']),clamp_mlp20=False)),('sham_M20',{}),
                               ('positive_U',dict(U_override=float(ref['U'])+.1)),('positive_W',dict(W_override=float(ref['W'])+.1))]:
            rec,arr=m.run(d,prefix+'_'+label,reference=ref,**settings);probes[label]=(rec,arr)
            if label.startswith('sham'):np.testing.assert_array_equal(rec['scores'],natural['scores'])
            else:
                n=float(np.linalg.norm(np.asarray(rec['scores'])-natural['scores']))
                if n<=1e-6:raise RuntimeError('Positive readout control absent '+label)
                if label=='positive_U' and abs(float(arr['W'])-float(ref['W']))<=1e-6:raise RuntimeError('U cannot reach W sensitivity')
        values={}
        for branch in ('b','c'):
            for h,label in zip(H,HLABELS):
                for sign,t in [('plus',h),('minus',-h)]:
                    rec,_=m.run(d,f'{prefix}_fd_{branch}_{label}_{sign}',reference=ref,t=t,delta=delta,branch=branch)
                    values[branch,label,sign]=rec['margin']
        deriv=packet[d['id']]['derivatives'];ad=np.array([deriv['b']['prime']-deriv['c']['prime'],.5*(deriv['b']['second']-deriv['c']['second'])])
        d0=deriv['b']['margin']-deriv['c']['margin'];estimates=[]
        for h,label in zip(H,HLABELS):
            plus=values['b',label,'plus']-values['c',label,'plus'];minus=values['b',label,'minus']-values['c',label,'minus']
            estimates.append(np.array([(plus-minus)/(2*h),(plus-2*d0+minus)/(2*h*h)]))
        tolerance=np.maximum(.001,.05*abs(ad))
        for i in (0,1):precision_pairs[i] &= bool(np.all(abs(estimates[i]-ad)<=tolerance) and np.all(abs(estimates[i+1]-ad)<=tolerance) and np.all(abs(estimates[i+1]-estimates[i])<=tolerance))
        checks.append(dict(doc_id=d['id'],AD=ad.tolist(),FD=[e.tolist() for e in estimates],tolerance=tolerance.tolist(),
                           positive_norms={k:float(np.linalg.norm(np.asarray(v[0]['scores'])-natural['scores'])) for k,v in probes.items() if k.startswith('positive')},shams_exact=True))
    return dict(ok=True,checks=checks,adjacent_precision_pairs=precision_pairs,precision_ready=any(precision_pairs))


def signal(cases,pilot=False):
    cc=[r for r in cases if r['dose']==1.];G=float(np.mean([abs(r['C']) for r in cc]))
    by={f'{f}_S{s}':float(np.mean([abs(r['C']) for r in cc if r['frame']==f and r['S']==s])) for f in ('F0','F1') for s in (0,1)}
    blocks={str(g):float(np.mean([abs(r['C']) for r in cc if r['group']==g])) for g in sorted({r['group'] for r in cc})}
    return dict(G=G,by_group=by,by_block=blocks,ready=G>=.02 and min(by.values())>=.01 and sum(x>=.02 for x in blocks.values())>=(3 if pilot else 8))


def perform(m,stage,data,persist=True):
    docs=data['pilot_documents' if stage=='pilot' else 'documents']
    for i,d in enumerate(docs):
        rec,arr=m.run(d,f'{stage}_{d["id"]}_natural');historical(stage,d,'natural',rec,arr)
        if persist:status(stage+'_natural',documents=i+1,total=len(docs))
    packet=forecast(m,stage,docs);frozen=dict(documents=packet,whole_stage_natural_only=True)
    if persist:immutable(OUT/f'{stage}_forecasts.json',frozen)
    elif payload(OUT/f'{stage}_forecasts.json')!=frozen:raise ValueError('Frozen forecasts differ')
    cc=controls(m,stage,docs,data[stage+'_control_ids'],packet)
    if persist:immutable(OUT/f'{stage}_controls.json',cc)
    elif payload(OUT/f'{stage}_controls.json')!=cc:raise ValueError('Controls differ')
    if not cc['precision_ready']:
        result=dict(ok=True,ready=False,status='numerically_unresolved',rows=[],controls=cc)
    else:
        cases=[]
        for i,d in enumerate(docs):
            prefix=f'{stage}_{d["id"]}';natural,ref=m.indexed(prefix+'_natural');delta=packet[d['id']]['delta']
            for dose,label in [(.5,'half'),(1.,'one')]:
                observations={}
                for branch in ('b','c'):
                    rec,_=m.run(d,prefix+'_'+branch+'_'+label,reference=ref,t=dose,delta=delta,branch=branch)
                    if dose==1.:historical(stage,d,branch,rec)
                    observations[branch]=rec['margin']
                cases.append(dict(doc_id=d['id'],group=d['group'],frame=d['frame'],S=d['S'],A=d['A'],dose=dose,delta=delta,
                    C=(2*d['S']-1)*(observations['b']-observations['c']),predictions=packet[d['id']]['predictions'][str(dose)],observations=observations,
                    natural_margin=natural['margin'],natural_correct=(1-2*d['S'])*natural['margin']>0,
                    gate={u:float(ref[u+'_gate']) for u in ('U','W')},up={u:float(ref[u+'_up']) for u in ('U','W')}))
            if persist:status(stage+'_observations',documents=i+1,total=len(docs))
        sig=signal(cases,pilot=stage=='pilot');result=dict(ok=True,ready=bool(sig['ready']),status='pilot_complete' if stage=='pilot' else 'main_complete',rows=cases,signal=sig,controls=cc)
        if stage=='main':result['metrics']=summarize(cases)
    if persist:
        immutable(OUT/('pilot_results.json' if stage=='pilot' else 'results.json'),result)
        status(result['status'],ready=result['ready'])
    return result


def run(stage):
    if sys.flags.optimize:raise RuntimeError('Assertions required')
    identity=read(PREP/'identity.json')
    if digest(environment_identity())!=identity['model_identity']['environment_sha256']:raise RuntimeError('Environment changed')
    try:
        # Imports required for differentiation precede the 10GiB load guard.
        import torch
        from .subject import Subject
        from .loading import load
        guard(start=True)
        if stage=='main':
            if not payload(OUT/'pilot_results.json')['ready']:raise RuntimeError('Pilot gate closed')
            prior=payload(PREP/'main_prior_ledger.json')
            if list(rows(OUT/'ledger.jsonl'))[:len(prior)]!=prior:raise RuntimeError('Prior ledger changed')
        torch.set_num_threads(4);status('loading_'+stage)
        model=load(ROOT/read(CONFIG)['model']['local_path'],lambda done,total,key:status('loading_'+stage,done=done,total=total,key=key))
        guard();subject=Subject(Budget(stage,read(PREP/f'schedule_{stage}.json')),model)
        try:perform(Measurements(subject,stage),stage,read(PREP/'dataset.json'))
        finally:subject.close()
    except Exception as exc:
        import traceback
        append(OUT/'failures.jsonl',dict(utc=now(),stage=stage,error=repr(exc),traceback=traceback.format_exc()));status('checkpointed',error=repr(exc));raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','main'],required=True);run(p.parse_args().stage)
