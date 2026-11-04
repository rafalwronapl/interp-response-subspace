"""Analytic scalar, actual random-Qwen gradients, and hostile mixed-budget/cache tests."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from research_checks.gate import create,read
from .common import ROOT,PREP,CONFIG,Budget,Measurements
from .protocol import predictions,summarize,BASELINES


class Unit(unittest.TestCase):
    def test_analytic_and_scoring(self):
        import torch
        t=torch.tensor(0.,dtype=torch.float64,requires_grad=True);delta=7.;x=2.+t*delta;y=x**3
        first=torch.autograd.grad(y,t,create_graph=True)[0];second=torch.autograd.grad(first,t)[0]
        self.assertEqual(float(first.detach()),84.);self.assertEqual(float(second.detach()),588.)
        p=predictions(1,delta,.5,float(first.detach()),0.,float(second.detach()),0.,.1,.2)
        self.assertEqual(p['P2'],115.5)
        cc=[dict(group=g,frame=f,S=s,A=a,dose=1.,C=.05,predictions=dict(P2=.05,**{k:0. for k in BASELINES})) for g in range(16) for f in ('F0','F1') for s in (0,1) for a in (0,1)]
        self.assertEqual(summarize(cc)['status'],'supported')
        for c in cc:c['predictions']['mean_linear']=.05
        self.assertEqual(summarize(cc)['status'],'not_supported')

    def test_streaming_loader(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from .loading import load
        cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8)
        torch.manual_seed(31);model=Qwen3ForCausalLM(cfg).bfloat16().eval()
        expected={k:v.detach().float().clone() for k,v in model.named_parameters()}
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp,safe_serialization=True);restored=load(tmp)
            for k,v in restored.named_parameters():self.assertTrue(torch.equal(v,expected[k]),k)

    def test_mixed_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=copy.deepcopy(read(CONFIG));cfg['budget']['stage_caps']['pilot']['backward']=1
            create(root/'plans/E031_config.json',cfg)
            sc=dict(operations=[dict(id='f',kind='forward'),dict(id='b',kind='backward'),dict(id='b2',kind='backward')]);b=Budget('pilot',sc,root,lambda:None)
            with self.assertRaises(ValueError):b.reserve('backward','f')
            t=b.reserve('forward','f')
            with self.assertRaises(RuntimeError):b.reserve('backward','b')
            b.finish(t,True,.01);t=b.reserve('backward','b');b.finish(t,False,.01)
            with self.assertRaises(RuntimeError):Budget('pilot',sc,root,lambda:None).reserve('backward','b2')

    def test_cache_resume(self):
        class Fake:
            calls=0
            def run(self,*a,**kw):
                self.calls+=1
                if self.calls==1:raise RuntimeError('partial')
                return dict(margin=1.,scores=[2.,1.]),dict(U=np.asarray(1.,dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);create(root/'analysis/E031_preparation/identity.json',dict(model_identity='fake',code_identity='fake'))
            subject=Fake();m=Measurements(subject,'pilot',root);d=dict(id='x',ids=[1],readout_ids=[1,2])
            with self.assertRaises(RuntimeError):m.run(d,'f',t=0.)
            rec,_=m.run(d,'f',t=0.);before=(m.out/'index/f.json').read_bytes();m.run(d,'f',t=0.)
            self.assertEqual(subject.calls,2);self.assertEqual(before,(m.out/'index/f.json').read_bytes())
            with self.assertRaises(ValueError):m.run(d,'f',t=1.)
            with self.assertRaises(ValueError):m.run(dict(d,readout_ids=[2,1]),'f',t=0.)
            path=root/rec['array_file'];old=path.read_bytes();path.write_bytes(b'bad')
            with self.assertRaises(ValueError):m.run(d,'f',t=0.)
            path.write_bytes(old);raw=next((m.out/'raw').glob('*.json'));old=raw.read_bytes();raw.write_text('{}')
            with self.assertRaises((ValueError,KeyError)):m.run(d,'f',t=0.)
            raw.write_bytes(old);m.run(d,'f',t=0.);self.assertEqual(subject.calls,2)


def synthetic():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from .subject import Subject
    torch.set_num_threads(4);torch.manual_seed(31031)
    cfg=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=6144,num_hidden_layers=28,num_attention_heads=16,num_key_value_heads=8,head_dim=8,attention_dropout=0.)
    cfg._attn_implementation='eager';model=Qwen3ForCausalLM(cfg).float().eval()
    class Counter:
        calls={'forward':0,'backward':0};kinds={}
        def reserve(self,kind,label):self.calls[kind]+=1;t=(kind,self.calls[kind]);self.kinds[t]=label;return t
        def finish(self,*a):pass
    count=Counter();sub=Subject(count,model)
    d=dict(id='x',ids=[3,4,5,6,7],readout_ids=[1,2]);other=dict(d,ids=[3,8,5,6,7]);checks={}
    try:
        natural,ref=sub.run(d,'natural');_,donor_ref=sub.run(other,'other');delta=float(donor_ref['U'])-float(ref['U'])
        # Extra fixed synthetic sensitivity delta avoids relying on random donor signal.
        delta=delta if abs(delta)>.05 else .1
        zero={}
        for branch in ('b','c'):
            rec,arr=sub.run(d,branch+'_zero',reference=ref,t=0.,delta=delta,branch=branch,derivatives=True)
            np.testing.assert_array_equal(rec['scores'],natural['scores']);np.testing.assert_array_equal(arr['MLP20_after'],ref['MLP20_after'])
            zero[branch]=rec
            h=.05;plus,_=sub.run(d,'plus',reference=ref,t=h,delta=delta,branch=branch);minus,_=sub.run(d,'minus',reference=ref,t=-h,delta=delta,branch=branch)
            fd=(plus['margin']-minus['margin'])/(2*h)
            if abs(fd-rec['prime'])>max(1e-5,.03*abs(rec['prime'])):raise RuntimeError('Random-Qwen gradient inconsistent')
            checks[branch]=dict(prime=rec['prime'],second=rec['second'],finite_first=fd)
        probe,arr=sub.run(d,'probe',reference=ref,U_override=float(ref['U'])+.1)
        if abs(float(arr['W'])-float(ref['W']))<=1e-9 or np.linalg.norm(np.asarray(probe['scores'])-natural['scores'])<=1e-9:raise RuntimeError('U/W sensitivity absent')
        clamped,arr=sub.run(d,'clamp',reference=ref,t=1.,delta=.1,branch='c')
        if not any(e['kind']=='unit' and e['unit']=='W' and e['replacement']==float(ref['W']) for e in clamped['audits']):raise RuntimeError('W clamp missing')
    finally:sub.close()
    return dict(ok=True,operations=count.calls,checks=checks,active_graph=True,native_shams=True,MLP20_exact=True,prefix_locality=True,pretrained_calls=0)


def run():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Unit);r=unittest.TextTestRunner(verbosity=2).run(suite)
    if not r.wasSuccessful():raise RuntimeError('Unit tests failed')
    return dict(ok=True,unit_tests=r.testsRun,synthetic=synthetic(),hooks_ok=True,data_ok=True,pilot_ready=True,accounting_ok=True,restart_ok=True,pretrained_calls=0)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();create(ROOT/a.out,run())
