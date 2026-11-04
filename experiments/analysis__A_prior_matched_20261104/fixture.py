"""Actual worker synthetic crash/restart, global target freeze and call accounting."""
from pathlib import Path
import json,tempfile,numpy as np
from research_checks.persistence import immutable
from research_checks.gate import sha
from analysis.AB_five_experiments_20261103.A_L21.adapter_v4 import enc
import torch,types
def adapter_fixture():
 from .adapter import Adapter
 from ab_confirmatory_20261101.native_tail_b import encode
 class Layer(torch.nn.Module):
  def forward(self,h,**kwargs):return torch.tanh(h+.03)
 model=types.SimpleNamespace(model=types.SimpleNamespace(layers=[Layer() for _ in range(28)],norm=torch.nn.Identity()),lm_head=torch.nn.Linear(4,20,bias=False))
 with torch.no_grad():model.lm_head.weight.copy_(torch.arange(80).reshape(20,4)/80)
 a=Adapter.__new__(Adapter);a.model=model;a.heads={' is':2,' are':7,' has':11,' have':17};a.four_weight=model.lm_head.weight[[2,7,11,17]].detach().clone()
 h=torch.arange(12,dtype=torch.float32).reshape(1,3,4)/12;r=h*.3;record=dict(H=encode(h),R=encode(r),kwargs={str(k):encode({}) for k in range(22,28)})
 ys=np.arange(108,dtype=np.float32).reshape(27,4)/108;actual=[];expected=[]
 for start in [0,16]:actual+=a.both(record,ys[start:start+16])
 for y in ys:
  hh=h.clone();hh[0,-1]=r[0,-1]+torch.as_tensor(y)
  for layer in model.model.layers[22:]:hh=layer(hh)
  z=model.lm_head(hh)[0,-1].detach().numpy();expected.append([float(z[7]-z[2]),float(z[17]-z[11])])
 np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
 def full(d,x,m,Y):model.lm_head(torch.as_tensor(Y)[None,:])
 a.full=full;Y=np.arange(12,dtype=np.float32).reshape(3,4)/12;z=model.lm_head(torch.as_tensor(Y)[None,:])[0,-1].detach().numpy();np.testing.assert_allclose(a.full_both({},None,None,Y),[z[7]-z[2],z[17]-z[11]],atol=1e-6)
 return dict(actual_new_adapter_batches=[16,11],four_head_orientation=True,full_hook_four_head_reference=True)
from . import worker as w
BASE=Path(__file__).parent
def main():
 adapter_checks=adapter_fixture();original={k:getattr(w,k) for k in ['BASE','ROOT','Adapter','load','immutable']};counter=dict(load=0,physical=0,max_batch=0,test_targets=0)
 with tempfile.TemporaryDirectory(prefix='fixture_',dir=BASE) as tmp:
  root=Path(tmp);stage=root/'stage';stage.mkdir();w.BASE=stage;w.ROOT=root;put=lambda p,v:p.write_text(json.dumps(v))
  dim=16;rng=np.random.default_rng(17);U=np.linalg.qr(rng.normal(size=(dim,8)))[0];np.savez(root/'prior.npz',B5_U=U,B5_W=rng.normal(size=(2*dim,8))*.001,scale=np.ones(2))
  fitdocs=[dict(id=f'F{b}_{s}_{a}',block=b,cohort=b//2,split='FIT_NEW',S=s,A=a,donor=f'F{b}_{1-s}_{a}') for b in range(12) for s in [0,1] for a in [0,1]]
  testdocs=[dict(id=f'T{b}_{s}_{a}',block=b,split='TEST_NEW',S=s,A=a,donor=f'T{b}_{1-s}_{a}') for b in range(2) for s in [0,1] for a in [0,1]]
  class Fake:
   def __init__(self,*args):pass
   def make(self,d,x=None,m=None):
    v=np.arange(dim,dtype=np.float32)/dim+float(d['block'])*.02;X=float(d['S']+.1*d['A']);M=float(d['A']-.2*d['S']);delta=0 if x is None else x-X;v=v+delta*.1
    y=np.asarray(np.tanh(v)+delta*.03,np.float32);return dict(X=np.asarray(X),M=np.asarray(M),x19=v,x20=v*.7,x21=v*.9,earlier19=np.asarray('old'),earlier20=np.asarray('old'),earlier21=np.asarray('old'),y=y,Y=np.stack([v*0,y]),logits=np.array([0.,float(y.sum())]))
   def full(self,d,x=None,m=None):
    if d['split']=='TEST_NEW' and x is not None:
     assert (stage/'run/TEST_forecasts.json').exists();assert len(w.unwrap(stage/'run/TEST_forecasts.json')['arrays'])==25;counter['test_targets']+=1
    counter['physical']+=1;return dict(state={k:enc(v) for k,v in self.make(d,x,m).items()})
   def prefix(self,d,x,m):counter['physical']+=1;return {k:enc(v) for k,v in self.make(d,x,m).items() if k not in ['y','Y','logits']}
   def value(self,y):return [float(np.sum(y)),float(np.sum(np.asarray(y)**2))]
   def both(self,record,ys):counter['physical']+=1;counter['max_batch']=max(counter['max_batch'],len(ys));assert len(ys) in [16,11];return [self.value(y) for y in ys]
   def full_both(self,d,x,m,Y):counter['physical']+=1;return self.value(Y[-1])
   def close(self):pass
  ops=w.schedule(fitdocs,testdocs);physical=len(fitdocs)*9+len(testdocs)*17+10
  put(stage/'corpus.json',dict(documents=fitdocs,test_documents=testdocs,heads={}));put(stage/'input_manifest.json',dict(files={}));put(stage/'operations.json',ops);put(stage/'config.json',dict(model='synthetic',prior='prior.npz',checkpoint='model',maximum_forward=len(ops['operations']),physical_maximum=physical))
  def loader(*args):counter['load']+=1;return None
  w.Adapter=Fake;w.load=loader;preserved={}
  try:
   for crashpoint in ['TEST_forecasts.json','rows']:
    triggered=[False]
    def crashing(p,v):
     result=immutable(p,v)
     if not triggered[0] and (Path(p).name==crashpoint or Path(p).parent.name==crashpoint):triggered[0]=True;raise RuntimeError('fixture crash')
     return result
    w.immutable=crashing
    try:w.main()
    except RuntimeError as e:assert str(e)=='fixture crash'
    else:raise AssertionError('No crash')
    assert triggered[0]
    for p,h in preserved.items():assert sha(stage/'run'/p)==h
    preserved={p.relative_to(stage/'run').as_posix():sha(p) for p in (stage/'run').rglob('*') if p.is_file() and p.name not in ['STATUS.json','ledger.jsonl'] and 'loads' not in p.parts}
   w.immutable=immutable;w.main();done=w.unwrap(stage/'run/completed.json');assert done['physical_forward_calls']==counter['physical']==physical;assert counter['max_batch']==16 and counter['test_targets']==4*len(testdocs)
   for p,h in preserved.items():assert sha(stage/'run'/p)==h
   before=counter.copy();w.main();assert counter==before
   row=next((stage/'run/rows').glob('*.json'));row.write_text(row.read_text()+' ')
   try:w.main()
   except AssertionError:pass
   else:raise AssertionError('Corrupt seal accepted')
   assert counter==before
   (stage/'run/completed.json').unlink();next((stage/'run/cache').glob('*.json')).unlink()
   try:w.main()
   except ValueError as e:assert 'Uncertain prior attempt' in str(e)
   else:raise AssertionError('Uncertain reservation accepted')
   assert counter==before
  finally:
   for k,v in original.items():setattr(w,k,v)
 result=dict(hooks=True,data=True,pilot=True,accounting=True,restart=True,worker_sha256=sha(BASE/'worker.py'),models_sha256=sha(BASE/'models.py'),adapter_checks=adapter_checks,pretrained_calls=0,synthetic_counter=before,global_freeze_before_TEST_targets=True,crash_restart_preserves_receipts=True,completed_replay_zero_calls=True,corrupt_seal_rejected=True,uncertain_reservation_rejected_before_load=True,logical_schedule_enforced=True,physical_forward_exact=True,batch_shapes=[16,11])
 (BASE/'fixture.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
