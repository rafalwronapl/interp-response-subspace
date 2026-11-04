"""Synthetic full worker with injected FIT sources and cheap deterministic regressors."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research_checks.gate import digest
from research_checks.persistence import immutable
from . import stage1 as worker
from .common import BASE,read,sha,packed,unwrap

def run():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);base=root/'stage';base.mkdir();prior=root/'prior.npz'
        U=np.eye(1024,8);np.savez(prior,B5_U=U,B5_W=np.zeros((2048,8)),scale=np.ones(2))
        index=dict(natural={},cells=[]);sources={}
        def entry(key,document,value):
            q=dict(document=document);r=dict(request=digest(q));p=dict(request=q,result=value)
            sources[root/(key+'_receipt')]=r;sources[root/(key+'_cache')]=p
            return dict(document=document,receipt=key+'_receipt',cache=key+'_cache')
        z=np.zeros(1024,dtype=np.float32)
        for block in range(24):
            for form in range(4):
                d=dict(id=f'd{block}_{form}',block=block,split='FIT'+str(block//8))
                index['natural'][d['id']]=entry(d['id'],d,dict(state={k:packed(z) for k in ['x20','x21','y']}))
                for branch in ['B','C']:
                    for dose in [.5,1.]:
                        tag=d['id']+'_'+branch+'_'+str(dose);x=z.copy();x[:8]=dose*(block+1)/24
                        p={k:packed(x) for k in ['x20','x21']};t=dict(state={**p,'y':packed(x*.25)})
                        index['cells'].append(dict(id=tag,document=d,branch=branch,dose=dose,prefix=entry(tag+'p',d,p),target=entry(tag+'t',d,t)))
        cfg=dict(prior='prior.npz');manifest=dict(files={'prior.npz':sha(prior)})
        def fake_read(p):
            if Path(p)==base/'config.json':return cfg
            if Path(p)==base/'input_manifest.json':return manifest
            if Path(p)==base/'source_index.json':return index
            return read(p)
        def fake_unwrap(p):return sources[Path(p)] if Path(p) in sources else unwrap(p)
        def fake_select(x,y,b,linear_only=False):return dict(kernel='linear',lam=.01),[dict(kernel='linear',lam=.01,block_L2=0.)]
        def fake_fit(x,y,param):return dict(ymean=y.mean(0),param=param)
        def fake_predict(m,x):return np.tile(m['ymean'],(len(x),1))
        actual_immutable=worker.immutable;failed=[False];calls=[0]
        def crash_after_summary(path,payload):
            out=actual_immutable(path,payload)
            if Path(path).name=='SUMMARY.json' and not failed[0]:failed[0]=True;raise RuntimeError('Simulated crash after durable summary')
            return out
        def counted_fit(*args):calls[0]+=1;return fake_fit(*args)
        with patch.multiple(worker,ROOT=root,BASE=base,read=fake_read,unwrap=fake_unwrap,select=fake_select,fit=counted_fit,predict=fake_predict):
            try:
                with patch.object(worker,'immutable',crash_after_summary):worker.main()
            except RuntimeError as e:assert 'Simulated crash' in str(e)
            else:raise AssertionError('Failure window not exercised')
            summary_before=(base/'stage1_run/SUMMARY.json').read_bytes();worker.main()
            assert (base/'stage1_run/SUMMARY.json').read_bytes()==summary_before
            before=calls[0];worker.main();assert calls[0]==before
            p=base/'stage1_run/MODELS.json';p.write_text(p.read_text()+' ')
            try:worker.main()
            except ValueError as e:assert 'Changed completed' in str(e)
            else:raise AssertionError('Corruption accepted')
    immutable(BASE/'e2e.json',dict(passed=True,synthetic_cells=384,outer_groups=3,partial_summary_recovery=True,completed_replay_fits=0,
        corrupt_seal_rejected=True,native_evaluations=0,worker_sha256=sha(Path(worker.__file__))))
    print('synthetic full worker/restart PASS')
if __name__=='__main__':run()
