"""Charged pretrained adapter qualification; no new scientific outcome selection."""
from pathlib import Path
import json,uuid,numpy as np,torch
from research_checks.gate import sha,digest,read
from research_checks.persistence import immutable
from ab_confirmatory_20261101.runtime import Runtime
from e031.loading import load
from .adapter_v3 import Adapter,state,scalar,unpack
BASE=Path(__file__).parent;ROOT=BASE.parents[2]

def main():
 cfg=read(BASE/'pilot_config.json');manifest=read(BASE/'pilot_inputs_v3.json')['files']
 for path,h in manifest.items():
  if sha(ROOT/path)!=h:raise ValueError('Changed '+path)
 out=BASE/'pilot_run';schedule=read(BASE/'pilot_schedule.json');identity=dict(model=cfg['model'],code=digest(manifest))
 rt=Runtime(out,identity,schedule)
 if (out/'completed.json').exists():return
 immutable(out/'loads'/(uuid.uuid4().hex+'.json'),dict(model=cfg['model']))
 torch.set_num_threads(4);model=load(ROOT/'model');a=Adapter(model);d=cfg['document']
 def run(op,fn,extra=None):return rt.compute(op,d,dict(role=op,extra=extra),dict(ids=[525,374]),fn)
 try:
  natural=run('natural',lambda:a.full(d));s=state(natural)
  # Fixed technical perturbation, not data-dependent scientific selection.
  x=float(s['X'])+.25;m=float(s['M'])
  prefix=run('prefix',lambda:a.prefix(d,x,m));p=unpack(prefix)
  native=run('native',lambda:a.full(d,x,m));t=state(native)
  assert all(np.array_equal(p[k],t[k]) for k in ['x19','x20','x21','earlier19','earlier20','earlier21'])
  y=t['y'];direction=(model.lm_head.weight[525]-model.lm_head.weight[374]).detach().numpy();direction/=np.linalg.norm(direction)
  changed=(y+.1*max(1.,np.linalg.norm(y))*direction).astype(np.float32)
  values={};errors=[]
  for label,target in [('native',y),('baseline',s['y']),('changed',changed)]:
   Y=s['Y'].copy();Y[-1]=target
   full=run('full_'+label,lambda Y=Y:a.full(d,x,m,Y),target.tolist())
   tail=run('tail_'+label,lambda target=target:a.margin(native,target),target.tolist())
   errors.append(abs(tail-scalar(state(full))));values[label]=tail
  assert max(errors)<=1e-4
  assert abs(values['changed']-values['native'])>1e-4
  assert np.array_equal(s['Y'][:-1],t['Y'][:-1])
  immutable(out/'completed.json',dict(ready=True,tail_parity_max=max(errors),sensitivity=values['changed']-values['native'],forwards=9,backwards=0,scope='Technical qualification only, historical document'))
 finally:a.close()
if __name__=='__main__':main()
