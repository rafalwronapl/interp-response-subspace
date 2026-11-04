"""Exercise actual MAIN worker end-to-end with synthetic native/suffix only."""
from pathlib import Path
import json,tempfile,numpy as np,torch
from research_checks.gate import sha
from . import worker as w
from .adapter_v3 import enc
from .math import fit as realfit
BASE=Path(__file__).parent

def main():
 worker_hash=sha(Path(w.__file__))
 counter=dict(loads=0,forward=0,fits=0,bootstrap=0);original={k:getattr(w,k) for k in ['ROOT','BASE','load','Adapter','fit','score']};originalscore=w.score
 with tempfile.TemporaryDirectory(prefix='pipeline_fixture_',dir=BASE) as tmp:
  root=Path(tmp).resolve();assert root.parent==BASE.resolve()
  base=root/'stage';base.mkdir();w.ROOT=root;w.BASE=base
  put=lambda p,v:p.write_text(json.dumps(v))
  docs=[]
  for b in range(28):
   for s in [0,1]:
    for a in [0,1]:docs.append(dict(id=f'd{b}_{s}_{a}',block=b,split='FIT'+str(b//8) if b<24 else 'TEST',S=s,A=a,ids=[1,b+2,s+32,a+35],donor=f'd{b}_{1-s}_{a}'))
  put(base/'corpus.json',dict(documents=docs));put(base/'main_config.json',dict(model='synthetic'))
  put(base/'main_inputs.json',dict(files={}));put(base/'library_identity.json',dict(versions={},sources={}))
  put(base/'main_schedule.json',w.schedule(docs))
  rng=np.random.default_rng(54);U=np.linalg.qr(rng.normal(size=(40,8)))[0]
  pp=root/'runs/E036/development';pp.mkdir(parents=True);np.savez(pp/'predictors.npz',scale=np.ones(2),B5_W=rng.normal(size=(80,8))*.01,B5_U=U)
  class Fake:
   def __init__(self,model):self.W=np.random.default_rng(1).normal(size=(40,40))*.1
   def vectors(self,d,x=None,m=None):
    rng=np.random.default_rng(d['block']*4+d['S']*2+d['A']+101)
    xx=float(d['S']+d['A']*.1);mm=float(d['A']-.2*d['S']);a=rng.normal(size=40);b=rng.normal(size=40);c=rng.normal(size=40)
    delta=0 if x is None else x-xx;dm=.1*delta if m is None else m-mm
    if d['block']==24:delta=dm=0. # Explicit alias/undefined-direction test stratum.
    x20=a+delta*b;x21=b+delta*c+dm*a+delta*delta*b*.1
    y=np.asarray(np.tanh(x21@self.W)+.01*x21*x21,np.float32);Y=np.stack([np.zeros(40,np.float32),y]);R=c
    s=dict(X=np.asarray(xx,np.float32),M=np.asarray(mm,np.float32),x19=a.astype(np.float32),x20=x20.astype(np.float32),x21=x21.astype(np.float32),earlier19=np.asarray('e19'),earlier20=np.asarray('e20'),earlier21=np.asarray('e21'),y=y,Y=Y)
    return s,R
   def full(self,d,x=None,m=None):
    counter['forward']+=1
    if d['split']=='TEST' and x is not None:assert (base/'main_run/TEST_forecasts.json').exists()
    s,R=self.vectors(d,x,m);value=float(np.sin((R+s['y']).sum()*.03));s['logits']=np.asarray([0.,value],np.float32)
    return dict(state={k:enc(v) for k,v in s.items()},R=R.tolist())
   def prefix(self,d,x,m):
    counter['forward']+=1;s,R=self.vectors(d,x,m)
    return {k:enc(v) for k,v in s.items() if k not in ['y','Y']}
   def margin(self,record,y):
    counter['forward']+=1;return float(np.sin((np.asarray(record['R'])+y).sum()*.03))
   def close(self):pass
  def load(path):counter['loads']+=1;return object()
  def fit(*args):counter['fits']+=1;return realfit(*args)
  def score(rows):counter['bootstrap']+=1;return originalscore(rows)
  w.load=load;w.Adapter=Fake;w.fit=fit;w.score=score
  try:
   w.main();done=w.unwrap(base/'main_run/completed.json');assert counter['forward']==done['attempts'];assert done['aliases']>0 and done['skips']>0;before=counter.copy()
   w.main();assert counter==before
   row=next((base/'main_run/rows').glob('*.json'));row.write_text(row.read_text()+' ')
   try:w.main()
   except ValueError as e:assert 'Completed stage changed' in str(e)
   else:raise AssertionError('Corrupt completion accepted')
  finally:
   for key,value in original.items():setattr(w,key,value)
 assert worker_hash==sha(Path(w.__file__))
 result=dict(ok=True,worker_sha256=worker_hash,trained_calls=0,synthetic_counter=before,conditions=done['conditions'],attempts=done['attempts'],aliases=done['aliases'],skips=done['skips'],global_forecast_before_TEST_targets=True,completed_replay_zero_calls=True,corrupt_completion_rejected=True)
 (BASE/'pipeline_fixture.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
