"""Meaningful synthetic regression, actual random-Qwen hook and hostile cache checks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research_checks.gate import read,create
from research_checks.persistence import immutable
from .common import PREP,CAPS,TIMES,Budget,Measurements,events

class Persistence(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name)
        self.b=Budget('pilot',self.folder,{'a':'network','b':'network'},lambda:None)
        self.m=Measurements('pilot',self.b,dict(model_identity={'model':'synthetic'},code_identity='test'),self.folder)
        self.doc=dict(id='unit',ids=[1,2]);self.calls=0
    def tearDown(self):self.tmp.cleanup()
    def compute(self):self.calls+=1;return dict(x=np.asarray([1.,2.],dtype=np.float32))
    def run_once(self,metadata=None):return self.m.run('a','network',self.doc,{} if metadata is None else metadata,self.compute)
    def test_identical_resume_preserves_receipts(self):
        self.run_once();before={str(p):p.read_bytes() for p in self.folder.rglob('*') if p.is_file()}
        self.run_once();self.assertEqual(self.calls,1)
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.folder.rglob('*') if p.is_file()})
    def test_changed_request_rejected(self):
        self.run_once()
        with self.assertRaises(ValueError):self.run_once({'dose':1.})
        self.assertEqual(self.calls,1)
    def test_corrupt_array_rejected(self):
        self.run_once();p=next((self.folder/'arrays').glob('*.npz'))
        with p.open('ab') as f:f.write(b'corruption')
        with self.assertRaises(ValueError):self.run_once()
        self.assertEqual(self.calls,1)
    def test_corrupt_raw_rejected(self):
        self.run_once();p=next((self.folder/'raw').glob('*.json'));p.write_text('{}',encoding='utf-8')
        with self.assertRaises((KeyError,ValueError)):self.run_once()
    def test_partial_index_failure_resumes_from_raw(self):
        original=immutable
        def fail_index(path,value):
            if Path(path).parent.name=='index':raise OSError('simulated interruption after raw publication')
            return original(path,value)
        with patch('e036.common.immutable',fail_index):
            with self.assertRaises(OSError):self.run_once()
        self.run_once();self.assertEqual(self.calls,1)
    def test_count_time_unscheduled_and_failure(self):
        with self.assertRaises(ValueError):self.b.call('unknown','network',self.compute)
        with patch.dict(CAPS['pilot'],network=1):
            self.run_once()
            with self.assertRaises(RuntimeError):self.b.call('b','network',self.compute)
        with patch.dict(TIMES,pilot=0.):
            with self.assertRaises(RuntimeError):self.b.call('b','network',self.compute)
        with self.assertRaises(ZeroDivisionError):self.b.call('b','network',lambda:1/0)
        self.assertEqual(self.b.summary()['failed'],1)
    def test_unfinished_reservation_fails_closed(self):
        from .common import append
        append(self.b.log,dict(event='reserve',id='interrupted',label='a',category='network'))
        with self.assertRaises(RuntimeError):self.run_once()
        self.assertEqual(self.calls,0)
    def test_target_before_forecast_freeze_blocked(self):
        m=Measurements('test',self.b,self.m.identity,self.folder)
        with self.assertRaises((FileNotFoundError,ValueError)):
            m.run('a','target',self.doc,{},self.compute)
        self.assertEqual(self.calls,0)
    def adopted(self,category='network'):
        identity=dict(self.m.identity,code_identity='new-code',pilot_repair=dict(old_cache=str(self.folder),old_code_identity='test',
            adoptable_labels=['a'],equivalence_sha256='synthetic-equivalence'))
        folder=self.folder/'revision'
        budget=Budget('pilot',folder,{'a':category,'b':'network'},lambda:None,carry_ledger=self.b.log)
        return Measurements('pilot',budget,identity,folder),budget
    def test_reviewed_adoption_preserves_old_records_and_carries_cost(self):
        self.run_once();old=self.b.log.read_bytes()
        m,b=self.adopted();m.run('a','network',self.doc,{},self.compute)
        self.assertEqual(self.calls,1);self.assertEqual(old,self.b.log.read_bytes())
        self.assertEqual(b.summary()['counts']['network'],1)
        self.assertTrue((m.folder/'adoptions/a.json').exists())
        with patch.dict(CAPS['pilot'],network=1):
            with self.assertRaises(RuntimeError):b.call('b','network',self.compute)
    def test_adoption_never_reuses_target_and_checks_semantics(self):
        self.run_once();m,b=self.adopted('target')
        m.run('a','target',self.doc,{},self.compute)
        self.assertEqual(self.calls,2);self.assertEqual(b.summary()['counts']['target'],1)
        m,b=self.adopted()
        with self.assertRaises(ValueError):m.run('a','network',self.doc,{'changed':True},self.compute)
    def test_adoption_rejects_corrupt_historical_array(self):
        self.run_once();p=next((self.folder/'arrays').glob('*.npz'))
        with p.open('ab') as f:f.write(b'bad')
        m,b=self.adopted()
        with self.assertRaises(ValueError):m.run('a','network',self.doc,{},self.compute)
        self.assertEqual(self.calls,1)
    def test_explicit_equivalent_kernel_target_adoption(self):
        self.b.allowed={'a':'target'}
        self.m.run('a','target',self.doc,{'tokens':3},self.compute)
        m,b=self.adopted('target');m.adoption['adopt_categories']=['target']
        m.run('a','target',self.doc,{'tokens':3},self.compute)
        self.assertEqual(self.calls,1);self.assertEqual(b.summary()['counts']['target'],1)

class StageResume(unittest.TestCase):
    """Interrupt actual stage finalization, outside per-measurement caching."""
    def test_pilot_resume_preserves_original_resource_receipt(self):
        from types import SimpleNamespace
        from . import worker
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            measure=SimpleNamespace(stage='pilot',budget=SimpleNamespace(summary=lambda:dict(counts={},seconds=0.,failed=0,unfinished=0)))
            data={'D':np.ones((2,4))};rows=[{'document':'pilot'}]*2
            with patch.object(worker,'OUT',folder),patch.object(worker,'natural',return_value=({},{})),patch.object(worker,'collect',return_value=(data,rows)),patch.object(worker,'pilot_controls',side_effect=lambda *args:dict(ready=True,epsilon_num=1e-7)),patch.object(worker,'guard',side_effect=[{'free':10.},{'free':9.}]),patch.object(worker,'status'):
                with patch.object(worker,'freeze_dataset',side_effect=OSError('interrupt after checks publication')):
                    with self.assertRaises(OSError):worker.execute('pilot',None,None,measure,[{'id':'pilot'}])
                receipt=folder/'pilot/checks_receipt.json';before=receipt.read_bytes()
                worker.execute('pilot',None,None,measure,[{'id':'pilot'}])
                self.assertEqual(before,receipt.read_bytes())
                self.assertEqual(read(folder/'pilot/checks.json')['resources'],{'free':10.})
                self.assertTrue((folder/'pilot/complete.json').exists())
    def test_development_resume_after_predictor_before_checks(self):
        from types import SimpleNamespace
        from . import worker,prepare,learning
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);(folder/'pilot').mkdir()
            (folder/'pilot/checks.json').write_text('{"epsilon_num":1e-7}',encoding='utf-8')
            measure=SimpleNamespace(stage='development',budget=SimpleNamespace(summary=lambda:dict(counts={},seconds=0.,failed=0,unfinished=0)))
            docs=[dict(id='fit',split='fit'),dict(id='select',split='select')]
            rows=[dict(document='fit')]*4+[dict(document='select')]*4
            data=dict(D=np.ones((8,2)),P=np.zeros((8,2)))
            with patch.object(worker,'OUT',folder),patch.object(worker,'natural',return_value=({},{})),patch.object(worker,'collect',return_value=(data,rows)),patch.object(worker,'status'),patch.object(learning,'fit',return_value=({'matrix':np.ones((2,1))},{'fixture':True})) as fitting,patch.object(learning,'signal',return_value=dict(ready=True)):
                with patch.object(prepare,'write',side_effect=OSError('interrupt before SELECT check publication')):
                    with self.assertRaises(OSError):worker.execute('development',None,None,measure,docs)
                receipt=folder/'development/predictors.json';before=receipt.read_bytes()
                worker.execute('development',None,None,measure,docs)
                self.assertEqual(fitting.call_count,1)
                self.assertEqual(before,receipt.read_bytes())
                self.assertTrue(read(folder/'development/checks.json')['select_signal'])
                self.assertTrue((folder/'development/complete.json').exists())

def hooks():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from .subject import Subject,Atlas
    torch.manual_seed(36036);torch.set_num_threads(4)
    cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=28,num_attention_heads=2,num_key_value_heads=1,head_dim=8)
    cfg._attn_implementation='eager';model=Qwen3ForCausalLM(cfg).float().eval()
    sub=Subject(model,2,3);doc=dict(ids=[1,2,3])
    try:
        ref=sub.forward(doc);sham=sub.forward(doc,x=float(ref['X']),m=float(ref['M']))
        for k in ref:np.testing.assert_array_equal(ref[k],sham[k])
        value=float(ref['X'])+.2
        pre=sub.forward(doc,x=value);post=sub.forward(doc,x=value,style='post')
        for k in ('x20','x21','y','logits'):np.testing.assert_allclose(pre[k],post[k],rtol=1e-5,atol=1e-6)
        if np.linalg.norm(pre['x21']-ref['x21'])<=1e-7:raise AssertionError('No synthetic sensitivity')
        np.testing.assert_array_equal(pre['x19'],ref['x19'])
        neg=sub.forward(doc,negative=True)
        for k in ('x19','x20','x21','y'):np.testing.assert_array_equal(neg[k],ref[k])
        target=model.model.layers[21].mlp;old=target.forward
        def forbidden(*a,**kw):raise AssertionError('Prefix leak')
        try:
            target.forward=forbidden;stopped=sub.forward(doc,prefix=True,x=value)
            if 'y' in stopped:raise AssertionError('Target exposed')
        finally:target.forward=old
        for l in (19,20,21):np.testing.assert_array_equal(pre['earlier'+str(l)],ref['earlier'+str(l)])
        np.testing.assert_allclose(sub.local(ref['x21'],tokens=len(doc['ids'])),ref['y'],rtol=1e-5,atol=1e-6)
        # Same-input full/local parity, plus row independence under different padding lengths.
        np.testing.assert_allclose(sub.local(post['x21'],tokens=len(doc['ids'])),post['y'],rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(sub.local(ref['x21'],tokens=5),sub.local(ref['x21'],tokens=1),rtol=1e-5,atol=1e-6)
        dx=pre['x21'].astype(float)-ref['x21'].astype(float);j=sub.jvp(ref['x21'],dx,torch.float64)
        fd=(sub.local(ref['x21'].astype(float)+1e-4*dx,torch.float64)-sub.local(ref['x21'].astype(float)-1e-4*dx,torch.float64))/(2e-4)
        np.testing.assert_allclose(j,fd,rtol=1e-5,atol=1e-8)
        # Actual CLT evaluation code: contribution differences, all features, threshold and decoder axes.
        a=Atlas.__new__(Atlas);a.layers={}
        for l in (20,21):a.layers[l]=dict(W_enc=torch.randn(7,16)*.1,b_enc=torch.ones(7)*.2,threshold=torch.ones(7)*.05,decoder=torch.randn(7,16)*.1)
        f0=a.evaluate(ref);delta=a.evaluate(pre,f0);direct=a.evaluate(pre)
        np.testing.assert_allclose(delta['p'],direct['p']-f0['p'],rtol=1e-5,atol=1e-7)
        np.testing.assert_array_equal(a.evaluate(ref,f0)['p'],np.zeros(16,dtype=np.float32))
    finally:sub.close()
    return dict(actual_random_Qwen=True,untrained=True,sham_exact=True,positive=True,negative=True,
                poison_prefix=True,pre_post_patch_equivalence=True,analytic_JVP=True,CLT_difference=True,pretrained_model_calls=0)

def learning_checks():
    from .learning import predict,assess,fit
    rng=np.random.default_rng(36)
    def sample(n):
        x=rng.normal(size=(n,8));z=rng.normal(size=(8,16))
        d=x@z;p=.15*d
        return dict(dx20=x[:,:4],dx21=x[:,4:],D=d,P=p,B6=.9*d,cell=np.tile(np.arange(8),n//8))
    f=sample(384);s=sample(128)
    # A fixed target map makes the held-out check genuinely predictive.
    w=rng.normal(size=(8,16))
    for part in (f,s):part['D']=np.concatenate([part['dx20'],part['dx21']],axis=1)@w;part['P']=.15*part['D']
    params,meta=fit(f,s)
    allowed={k:s[k] for k in ('dx20','dx21','P','cell','B6')};pred=predict(params,allowed)
    if np.linalg.norm(pred['C']-s['D'])>=.1*np.linalg.norm(s['D']):raise AssertionError('Known predictive map not learned')
    with unittest.TestCase().assertRaises(ValueError):predict(params,dict(allowed,D=s['D']))
    d=rng.normal(size=(384,16));data=dict(D=d,P=.1*d,block=np.repeat(np.arange(12),32),branch=np.tile(np.repeat([0,1],16),12),dose=np.tile(np.repeat([.5,1.],8),24))
    p=dict(C=.99*d,B0=np.zeros_like(d),B1=.1*d,B2=.2*d,B3=.3*d,B4=.97*d,B5=.95*d,B6=.9*d)
    result=assess(data,p,1e-7,2,np.eye(16)[:,:8])
    if result['decision']!='supported':raise AssertionError('Known supported case')
    p['C']=np.zeros_like(d)
    if assess(data,p,1e-7,2,np.eye(16)[:,:8])['decision']!='not_supported':raise AssertionError('Known negative case')
    p['B1']=d.copy();data['P']=d.copy()
    if assess(data,p,1e-7,2,np.eye(16)[:,:8])['decision']!='uninformative':raise AssertionError('Zero denominator')
    return dict(known_map_generalizes=True,target_allowlist=True,block_decisions=True,zero_denominator=True)

def main():
    from threadpoolctl import threadpool_limits
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (Persistence,StageResume)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise RuntimeError('Persistence/accounting tests failed')
    with threadpool_limits(limits=4):
        h=hooks();l=learning_checks()
    evidence=dict(ok=True,hooks_ok=True,data_ok=True,pilot_ready=True,accounting_ok=True,restart_ok=True,
                  unit_tests=result.testsRun,hooks=h,learning=l,pretrained_calls=0)
    create(PREP/'tests.json',evidence);print(evidence)

if __name__=='__main__':main()
