import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .common import (ROOT,PREP,Budget,Measurements,CAPS,TIMES,METHODS,read,create,
    immutable,payload,append,load_arrays,array_id)


class Persistence(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name)
        self.b=Budget('pilot',self.folder,{'a':'network','b':'network'},lambda:None)
        self.identity=dict(model_identity={'random':True},code_identity='test',protocol_identity={'test':True})
        self.m=Measurements('pilot',self.b,self.identity,self.folder);self.doc=dict(id='unit',ids=[1,2]);self.calls=0
    def tearDown(self):self.tmp.cleanup()
    def compute(self):self.calls+=1;return dict(x=np.array([1.,2.],dtype=np.float32))
    def once(self,meta=None):return self.m.run('a','network',self.doc,meta or {},self.compute)
    def test_resume_and_changed_request(self):
        self.once();before={p:p.read_bytes() for p in self.folder.rglob('*') if p.is_file()}
        self.once();self.assertEqual(self.calls,1)
        self.assertTrue(all(p.read_bytes()==v for p,v in before.items()))
        with self.assertRaises(ValueError):self.once({'dose':1})
    def test_corruption(self):
        self.once();p=next((self.folder/'arrays').glob('*.npz'))
        with p.open('ab') as f:f.write(b'bad')
        with self.assertRaises(ValueError):self.once()
    def test_raw_corruption(self):
        self.once();next((self.folder/'raw').glob('*.json')).write_text('{}',encoding='utf-8')
        with self.assertRaises((ValueError,KeyError)):self.once()
    def test_interruption_after_raw(self):
        def fail(path,value):
            if Path(path).parent.name=='index':raise OSError('injected')
            return immutable(path,value)
        with patch('e039.common.immutable',fail):
            with self.assertRaises(OSError):self.once()
        self.once();self.assertEqual(self.calls,1)
    def test_caps_unscheduled_and_failed(self):
        with self.assertRaises(ValueError):self.b.call('unknown','network',self.compute)
        with patch.dict(CAPS['pilot'],network=1):
            self.once()
            with self.assertRaises(RuntimeError):self.b.call('b','network',self.compute)
        with patch.dict(TIMES,pilot=0):
            with self.assertRaises(RuntimeError):self.b.call('b','network',self.compute)
        with self.assertRaises(ZeroDivisionError):self.b.call('b','network',lambda:1/0)
        self.assertEqual(self.b.summary()['failed'],1)
    def test_unfinished_and_prospective_gate(self):
        append(self.b.log,dict(event='reserve',id='x',label='b',category='network'))
        other=Budget('pilot',self.folder,self.b.allowed,lambda:None)
        with self.assertRaises(RuntimeError):other.call('a','network',self.compute)
        for metadata in ({'forecast_readout':True},{'intervention_target':True}):
            with self.assertRaises((ValueError,FileNotFoundError)):self.once(metadata)
        self.assertEqual(self.calls,0)


def decisions():
    from .analysis import assess,comparison,decide,joint
    rows=[(b,f,s,a,br,d) for b in range(12) for f in (0,1,2) for s in (0,1) for a in (0,1) for br in (0,1) for d in (.5,1.)]
    data={k:np.array([r[i] for r in rows]) for i,k in enumerate(('block','frame','S','A','branch','dose'))}
    data.update(Ftrue=np.ones(576),Fnat=np.zeros(576))
    for k in METHODS:data[k]=np.zeros(576)
    data['B5'][:]=.9;data['C'][:]=.5
    r=assess(data,1e-6)
    assert r['H1_joint']=='supported'
    assert all(r['frames'][str(f)]['B5_leads']['decision']=='supported' for f in range(3))
    assert r['paired_degradation']['1']['B5']['point']==0
    data['B5'][data['frame']==1]=0
    r=assess(data,1e-6)
    assert r['H1_joint']=='not_supported'
    assert r['frames']['2']['transfers']['B5']['decision']=='supported'
    assert r['frames']['1']['C_leads']['decision']=='supported'
    assert r['paired_degradation']['1']['B5']['degradation_supported']
    data['B5'][:]=.95;data['C'][:]=.5;data['B2'][:]=1.
    r=assess(data,1e-6)
    assert r['H1_joint']=='uninformative'
    assert r['frames']['1']['B5_leads']['decision']=='supported'
    assert assess(data,1e-6,technical=False)['H1_joint']=='technically_unresolved'
    data['B0'][data['frame']==1]=1.
    r=assess(data,1e-6)
    assert not r['signal']['frames']['1']['ready'] and r['signal']['frames']['2']['ready']
    assert r['frames']['1']['B5_leads']['decision']=='uninformative'
    comp=comparison(1.,1.,np.ones(4),np.array([1.,1.,0.,1.]))
    assert not comp['defined'] and comp['gain'] is None
    valid=dict(gain=.4,gain_upper=.6,raw_difference_lower=.01,defined=True)
    assert decide({'B1':valid},{'B1':.2},[1,1,1,-1],True,True)=='undetermined'
    assert decide({'B1':valid},{'B1':.2},[1]*4,True,True,dict(point=.1,lower=.01,upper=.2))=='not_supported'
    assert decide({'B1':valid},{'B1':.2},[1]*4,True,True,dict(point=.01,lower=-.1,upper=-.001))=='undetermined'
    assert joint(['supported','uninformative'])=='uninformative'
    assert joint(['not_supported','uninformative'])=='not_supported'
    assert joint(['not_supported','technically_unresolved'])=='technically_unresolved'
    # Independently reconstruct bootstrap first frame from per-block losses.
    data['B0'][:]=0.;data['B2'][:]=0.;data['B5']=.85+data['block']*.005
    r=assess(data,1e-6)
    block=np.array([(1-(.85+b*.005))**2 for b in range(12)])
    ix=np.random.Generator(np.random.PCG64(39039)).integers(0,12,size=(20000,12))
    expected=np.quantile(block[ix].mean(axis=1),[.05,.95],method='linear')
    np.testing.assert_allclose([r['frames']['0']['metrics']['B5'][k] for k in ('MSE_lower','MSE_upper')],expected,rtol=1e-12)
    return dict(frame_decisions=True,joint_not_pooled=True,independent_H2=True,undefined_bootstrap=True,
        technical_precedence=True,frame_signal=True,harmed_cell=True,absolute_accuracy=True,point_accuracy=True,paired_bootstrap=True)


def formula_check():
    from .prediction import predict
    from e036.learning import predict as old_predict
    pars=load_arrays(payload(ROOT/'runs/E036/development/predictors.json')['arrays'])
    rng=np.random.default_rng(38038)
    x={k:rng.normal(size=(8,1024)).astype(np.float32) for k in ('dx20','dx21','P','B6')}
    x['cell']=np.arange(8)
    got=predict(pars,x);old=old_predict(pars,x)
    for method in METHODS:np.testing.assert_array_equal(got[method],old[method].astype(np.float32))
    with unittest.TestCase().assertRaises(ValueError):predict(pars,dict(x,D=x['P']))
    return dict(all_eight_frozen_formulas_exact_FP32=True,all_B3_cells=True,allowlist=True)


def vector_checks():
    from .analysis import vector_diagnostics
    x=np.arange(1.,17.).reshape(4,4)
    data=dict(D=x,block=np.array([0,0,1,1]),frame=np.array([0,1,0,1]),S=np.array([0,1,0,1]),
        A=np.array([0,1,1,0]),branch=np.array([0,0,1,1]),dose=np.array([.5,1.,.5,1.]))
    vectors={k:x.copy() for k in METHODS};vectors['B0']=np.zeros_like(x)
    pooled,groups=vector_diagnostics(data,vectors)
    assert pooled['C']['MSE']==0 and pooled['B0']['NRMSE']==1
    assert pooled['B0']['cosine'] is None
    for key,values in groups.items():
        assert sum(g['rows'] for g in values.values())==4
        for value,g in values.items():
            mask=data[key]==float(value)
            assert g['metrics']['C']['MSE']==0
            assert g['metrics']['B0']['MSE']==float(np.mean(np.sum(x[mask]**2,axis=1)))
    data['D']*=0
    assert vector_diagnostics(data,vectors)[0]['C']['NRMSE'] is None
    return dict(pooled=True,all_six_subgroup_axes=True,zero_norms_explicit=True)


def random_pilot():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from .subject import Subject,Atlas
    from . import worker
    from .prepare import operations
    torch.manual_seed(38038);torch.set_num_threads(4)
    cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=28,num_attention_heads=2,num_key_value_heads=1,head_dim=8)
    cfg._attn_implementation='eager';model=Qwen3ForCausalLM(cfg).float().eval();sub=Subject(model,2,3)
    atlas=Atlas.__new__(Atlas);atlas.layers={}
    for layer in (20,21):atlas.layers[layer]=dict(W_enc=torch.randn(7,16)*.1,b_enc=torch.ones(7)*.2,
        threshold=torch.ones(7)*.05,decoder=torch.randn(7,16)*.1)
    docs=[]
    for frame in (0,1,2):
        for s in (0,1):
            for a in (0,1):
                docs.append(dict(id=f'test_{frame}_{s}_{a}',ids=[1+frame,3+s,5+a]+([7] if s else []),block=12,frame=frame,S=s,A=a))
    for d in docs:d['donor']=next(x['id'] for x in docs if (x['frame'],x['S'],x['A'])==(d['frame'],1-d['S'],d['A']))
    rng=np.random.default_rng(38039)
    pars=dict(scale=np.ones(2),C_W=rng.normal(size=(32,2))*.1,C_U=rng.normal(size=(16,2))*.1,
        B5_W=rng.normal(size=(32,2))*.1,B5_U=rng.normal(size=(16,2))*.1,
        B4_W=rng.normal(size=(32,16))*.1,B3_mean=np.zeros((8,16)),alpha=np.array(1.))
    identity=dict(model_identity={'untrained':True},code_identity='test',protocol_identity={'test':True},
        parameter_sha256='synthetic',hidden_size=16,row_counts=dict(pilot=48))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);folder=out/'pilot';op=operations(docs,'pilot')
            create(out/'pilot_operations.json',op)
            budget=Budget('pilot',folder,op['categories'],lambda:None);m=Measurements('pilot',budget,identity,folder)
            with patch.object(worker,'PREP',out),patch.object(worker,'OUT',out),patch.object(worker,'status'):
                original_run=worker.Operations.run
                prefix_calls=0
                def early_stop(self,role,*args,**kwargs):
                    nonlocal prefix_calls
                    if role=='prefix_before_MLP21':
                        prefix_calls+=1
                        if prefix_calls==5:raise OSError('partial prefix phase')
                    return original_run(self,role,*args,**kwargs)
                with patch.object(worker.Operations,'run',early_stop):
                    with unittest.TestCase().assertRaises(OSError):worker.execute('pilot',sub,atlas,m,docs,pars)
                partial={p:p.read_bytes() for p in (folder/'index').glob('*.json')}
                def vector_stop(which):
                    if which=='vectors':raise OSError('after vector freeze')
                with unittest.TestCase().assertRaises(OSError):worker.execute('pilot',sub,atlas,m,docs,pars,vector_stop)
                assert all(p.read_bytes()==v for p,v in partial.items())
                vector_receipt=(folder/'vectors.json').read_bytes()
                def stop(which):
                    if which=='readouts':raise OSError('after all forecasts')
                with unittest.TestCase().assertRaises(OSError):worker.execute('pilot',sub,atlas,m,docs,pars,stop)
                forecast=(folder/'forecasts.json').read_bytes()
                assert not any(payload(p)['request']['intervention']['metadata']['operation']['role']=='true_full' for p in (folder/'index').glob('*.json'))
                original=worker.Operations.run;calls=0
                def middle(self,role,*args,**kwargs):
                    nonlocal calls
                    if role=='true_full':
                        calls+=1
                        if calls==4:raise OSError('between targets')
                    return original(self,role,*args,**kwargs)
                with patch.object(worker.Operations,'run',middle):
                    with unittest.TestCase().assertRaises(OSError):worker.execute('pilot',sub,atlas,m,docs,pars)
                before={p:p.read_bytes() for p in folder.rglob('*') if p.is_file() and p.suffix!='.jsonl'}
                worker.execute('pilot',sub,atlas,m,docs,pars)
                assert payload(folder/'checks.json')['technical_controls_ok']
                assert (folder/'vectors.json').read_bytes()==vector_receipt
                assert (folder/'forecasts.json').read_bytes()==forecast
                assert all(p.read_bytes()==v for p,v in before.items())
                assert budget.summary()['counts']==op['counts']
                count=budget.summary().copy();complete=(folder/'complete.json').read_bytes()
                worker.execute('pilot',sub,atlas,m,docs,pars)
                assert budget.summary()==count and (folder/'complete.json').read_bytes()==complete
                assert len(list((folder/'index').glob('*.json')))==len(op['categories'])
                for p in (folder/'index').glob('*.json'):
                    ix=payload(p);role=ix['request']['intervention']['metadata']['operation']['role']
                    if role in ('forecast_full_bypass_MLP21','prefix_before_MLP21','poison_prefix','poison_replacement'):
                        np.testing.assert_array_equal(load_arrays(ix['record'])['projection_calls'],np.zeros(3,dtype=int))
                return dict(actual_random_Qwen=True,trained_calls=0,pilot_operation_counts=op['counts'],
                    complete_worker=True,unequal_donor_lengths=True,all_forecasts_before_targets=True,
                    forbidden_target_counts_zero=True,sham=True,poison=True,oracle=True,negative=True,positive=True,
                    JVP_FD=True,partial_prefix_resume=True,vector_freeze_resume=True,readout_freeze_resume=True,mid_target_resume=True,complete_resume_no_calls=True)
    finally:sub.close()


def main():
    from threadpoolctl import threadpool_limits
    import sys
    result=unittest.TextTestRunner(verbosity=2,stream=sys.stdout).run(unittest.defaultTestLoader.loadTestsFromTestCase(Persistence))
    if not result.wasSuccessful():raise RuntimeError('Persistence failure')
    with threadpool_limits(limits=4):
        d=decisions();print('Synthetic decisions passed',flush=True)
        vg=vector_checks();print('Subgroup vector diagnostics passed',flush=True)
        f=formula_check();print('All eight frozen formulas passed',flush=True)
        h=random_pilot();print('Random Qwen worker/restart/control tests passed',flush=True)
    prepared=read(ROOT/'analysis/E039_preparation/checks.json')
    create(PREP/'tests.json',dict(ok=True,hooks_ok=True,data_ok=prepared['ok'],pilot_ready=True,accounting_ok=True,restart_ok=True,
        unit_tests=result.testsRun,decisions=d,vector_diagnostics=vg,formula_equivalence=f,hooks=h,new_pretrained_calls=0,
        scope='Random-model technical flow and synthetic decision checks; trained pilot has not run.'))

if __name__=='__main__':main()
