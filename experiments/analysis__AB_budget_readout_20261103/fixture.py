"""Actual stage2 worker with synthetic states: no checkpoint loads or native calls."""
from pathlib import Path
import json,tempfile,numpy as np
from research_checks.gate import sha,digest
from research_checks.persistence import immutable
from analysis.AB_five_experiments_20261103.A_L21.adapter_v4 import enc
from analysis.A_L21_learned_20261103.common import packed
from . import worker as w
BASE=Path(__file__).parent
def main():
 original={k:getattr(w,k) for k in ['BASE','ROOT','load','Adapter','forecasts','summarize','immutable']};oldforecast=w.forecasts;oldsummary=w.summarize;counter=dict(load=0,logical=0,physical=0,predictions=0,bootstrap=0,max_batch=0)
 with tempfile.TemporaryDirectory(prefix='fixture_',dir=BASE) as tmp:
  root=Path(tmp).resolve();assert root.parent==BASE.resolve();stage=root/'stage';stage.mkdir();w.BASE=stage;w.ROOT=root
  put=lambda p,v:p.write_text(json.dumps(v))
  dim=32;rng=np.random.default_rng(36);U=np.linalg.qr(rng.normal(size=(dim,8)))[0];W=rng.normal(size=(2*dim,8))*.001;np.savez(root/'prior.npz',B5_U=U,B5_W=W,scale=np.ones(2))
  def model(size):return dict(xmean=packed(np.zeros(2*dim)),ymean=packed(np.zeros(size)),train=packed(np.zeros((1,2*dim))),alpha=packed(np.zeros((1,size))),scale=None,degenerate=False,param=dict(kernel='linear',lam=.01))
  immutable(root/'MODELS.json',dict(models={arm:model(dim if arm.startswith('Direct') else 8) for arm in w.LEARNED},U=packed(U),historical_prior_sha256=sha(root/'prior.npz')))
  docs=[dict(id=f'D{b}_{s}_{a}',block=b,split='TEST',S=s,A=a,ids=[1,b+2,s+10,a+12],donor=f'D{b}_{1-s}_{a}') for b in range(4) for s in [0,1] for a in [0,1]]
  class Fake:
   def __init__(self,model):self.direction=np.arange(dim)/dim
   def make(self,d,x=None,m=None):
    rng=np.random.default_rng(d['block']*4+d['S']*2+d['A']+50);a=rng.normal(size=dim);b=rng.normal(size=dim);c=rng.normal(size=dim);X=float(d['S']+.1*d['A']);M=float(d['A']-.2*d['S']);delta=0 if x is None else x-X;dm=.1*delta if m is None else m-M
    if d['block']==0:delta=dm=0.
    x20=a+delta*b;x21=b+delta*c+dm*a;y=np.asarray(.1*np.tanh(x21)+.05*x20,np.float32)
    return dict(X=np.asarray(X,np.float32),M=np.asarray(M,np.float32),x19=a.astype(np.float32),x20=x20.astype(np.float32),x21=x21.astype(np.float32),earlier19=np.asarray('old19'),earlier20=np.asarray('old20'),earlier21=np.asarray('old21'),y=y,Y=np.stack([np.zeros(dim,np.float32),y])),float(delta*.1)
   def f(self,y,R):return float(y@self.direction+.05*(y*y).sum()+R)
   def full(self,d,x=None,m=None,replacement=None):
    if d['split']=='TEST' and x is not None:assert (stage/'run/TEST_forecasts.json').exists(),'Target before all forecasts frozen'
    counter['logical']+=1;counter['physical']+=1;s,R=self.make(d,x,m)
    if replacement is not None:s['y']=np.asarray(replacement[-1],np.float32);s['Y']=replacement
    s['logits']=np.array([0.,self.f(s['y'],R)]);return dict(state={k:enc(v) for k,v in s.items()},R=R)
   def prefix(self,d,x,m):
    counter['logical']+=1;counter['physical']+=1;s,_=self.make(d,x,m);return {k:enc(v) for k,v in s.items() if k not in ['y','Y']}
   def margin_many(self,record,ys):
    assert 1<=len(ys)<=16
    counter['max_batch']=max(counter['max_batch'],len(ys))
    counter['logical']+=len(ys);counter['physical']+=1;return [self.f(y,record['R']) for y in ys]
   def close(self):pass
  fake=Fake(None);quals=[]
  for i in range(2):
   d=dict(id=f'FIT{i}',block=90+i,split='FIT0',S=0,A=i);setting=dict(x=.8,m=None,branch='B',dose=.5);q=dict(id=f'FIT{i}_B_0.5',document=d)
   for role,result in [('natural',fake.full(d)),('prefix',fake.prefix(d,setting['x'],setting['m'])),('target',fake.full(d,setting['x'],setting['m']))]:
    request=dict(document=d,intervention=setting);key=digest(request);cache=root/f'{i}_{role}_cache.json';receipt=root/f'{i}_{role}_receipt.json';immutable(cache,dict(request=request,result=result));immutable(receipt,dict(request=key));q[role]=dict(document=d,cache=cache.name,receipt=receipt.name)
   quals.append(q)
  counter['logical']=counter['physical']=0
  ops=w.schedule(docs,quals);put(stage/'config.json',dict(model='synthetic',checkpoint='model',prior='prior.npz',models='MODELS.json',dimension=dim,test_blocks=4,maximum_forward=len(ops['operations'])))
  put(stage/'input_manifest.json',dict(files={}));put(stage/'library_identity.json',dict(versions={},sources={}));put(stage/'corpus.json',dict(documents=docs));put(stage/'qualification.json',dict(contexts=quals));put(stage/'operations.json',ops)
  def load(*args):counter['load']+=1;return None
  def forecast(*args):counter['predictions']+=1;return oldforecast(*args)
  def summary(*args):counter['bootstrap']+=1;return oldsummary(*args)
  w.load=load;w.Adapter=Fake;w.forecasts=forecast;w.summarize=summary
  try:
   crashes=['TEST_forecasts.json','rows'];preserved={};forecast_counts=[]
   for checkpoint in crashes:
    triggered=[False]
    def crash_after_publish(path,value):
     result=immutable(path,value);p=Path(path)
     if not triggered[0] and (p.name==checkpoint or p.parent.name==checkpoint):triggered[0]=True;raise RuntimeError('synthetic durable checkpoint crash')
     return result
    w.immutable=crash_after_publish
    try:w.main()
    except RuntimeError as e:assert str(e)=='synthetic durable checkpoint crash'
    else:raise AssertionError('Synthetic checkpoint did not trigger')
    assert triggered[0]
    for name,h in preserved.items():assert sha(stage/'run'/name)==h
    preserved={p.relative_to(stage/'run').as_posix():sha(p) for p in (stage/'run').rglob('*') if p.is_file() and ('cache' in p.parts or 'bundles' in p.parts or 'qualification_forecasts' in p.parts or 'qualification' in p.parts or p.name=='TEST_forecasts.json' or p.parent.name=='rows')}
    forecast_counts.append(counter['predictions'])
   w.immutable=immutable
   w.main();done=w.unwrap(stage/'run/completed.json');assert counter['logical']==done['attempts'] and counter['physical']==done['physical_forward_calls'];assert done['aliases']>0;assert counter['predictions']==3 and counter['bootstrap']==1
   for name,h in preserved.items():assert sha(stage/'run'/name)==h
   assert forecast_counts==[3,3] and counter['load']==3
   saved_replace=Path.replace;saved_sleep=w.time.sleep;locked=[0]
   def fail_status(a,b):locked[0]+=1;raise PermissionError('synthetic advisory lock')
   Path.replace=fail_status;w.time.sleep=lambda _:None
   try:w.status(stage/'run','test_locked');assert locked[0]==5
   finally:Path.replace=saved_replace;w.time.sleep=saved_sleep
   before=counter.copy();w.main();assert counter==before
   oldcfg=json.loads((stage/'config.json').read_text());put(stage/'config.json',dict(oldcfg,model='incompatible'))
   try:w.main()
   except ValueError as e:assert 'Completed identity changed' in str(e)
   else:raise AssertionError('Changed completed identity accepted')
   assert counter==before;put(stage/'config.json',oldcfg)
   row=next((stage/'run/rows').glob('*.json'));row.write_text(row.read_text()+' ')
   try:w.main()
   except ValueError as e:assert 'Completed output changed' in str(e)
   else:raise AssertionError('Corrupt completed seal accepted')
   (stage/'run/completed.json').unlink();next((stage/'run/cache').glob('*.json')).unlink();before_uncertain=counter.copy()
   try:w.main()
   except ValueError as e:assert 'Uncertain previous operation' in str(e)
   else:raise AssertionError('Uncertain attempt accepted')
   assert counter==before_uncertain
  finally:
   for k,v in original.items():setattr(w,k,v)
 result=dict(hooks=True,data=True,pilot=True,accounting=True,restart=True,worker_sha256=sha(BASE/'worker.py'),trained_calls=0,synthetic_counter=before,conditions=done['conditions'],attempts=done['attempts'],aliases=done['aliases'],global_forecast_before_targets=True,completed_replay_zero_calls=True,corruption_rejected=True,partial_crash_after_global_freeze_and_row=True,preserved_cache_receipts_unchanged=True,frozen_predictions_not_regenerated=True,uncertain_attempt_rejected_before_load=True)
 (BASE/'fixture.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
