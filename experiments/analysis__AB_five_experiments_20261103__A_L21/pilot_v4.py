"""35 logical evaluations qualify exact suffix batches of16 against full native."""
from pathlib import Path
import uuid,numpy as np,torch
from research_checks.gate import read,sha,digest
from research_checks.persistence import immutable
from analysis.Qwen06_ABC_extension_20261102.bundle_runtime import BundleRuntime
from e031.loading import load
from .adapter_v4 import Adapter,state,scalar,unpack
BASE=Path(__file__).parent;ROOT=BASE.parents[2]
def main():
 cfg=read(BASE/'pilot_config.json');inputs=read(BASE/'pilot_inputs_v4.json')['files']
 for p,h in inputs.items():
  if sha(ROOT/p)!=h:raise ValueError('Changed input '+p)
 out=BASE/'pilot_run_v4';rt=BundleRuntime(out,dict(model=cfg['model'],code=digest(inputs)),read(BASE/'pilot_schedule_v4.json'))
 if (out/'completed.json').exists():return
 immutable(out/'loads'/(uuid.uuid4().hex+'.json'),dict(model=cfg['model']))
 torch.set_num_threads(4);model=load(ROOT/'model');a=Adapter(model);d=cfg['document']
 def run(op,fn,extra=None):return rt.compute(op,d,dict(role=op,extra=extra),dict(ids=[525,374]),fn)
 try:
  natural=run('natural',lambda:a.full(d));s=state(natural);x=float(s['X'])+.25;m=float(s['M'])
  prefix=unpack(run('prefix',lambda:a.prefix(d,x,m)));native=run('native',lambda:a.full(d,x,m));t=state(native)
  assert all(np.array_equal(prefix[k],t[k]) for k in ['x19','x20','x21','earlier19','earlier20','earlier21'])
  np.testing.assert_array_equal(s['Y'][:-1],t['Y'][:-1])
  direction=(model.lm_head.weight[525]-model.lm_head.weight[374]).detach().numpy();direction/=np.linalg.norm(direction)
  scales=[-2,-1,-.5,-.25,-.1,-.05,-.01,.01,.05,.1,.25,.5,1,2]
  targets=[t['y'],s['y']]+[np.asarray(t['y']+k*.1*max(1.,np.linalg.norm(t['y']))*direction,np.float32) for k in scales]
  full=[]
  for i,y in enumerate(targets):
   Y=s['Y'].copy();Y[-1]=y;f=run(f'full_{i}',lambda Y=Y:a.full(d,x,m,Y),y.tolist());full.append(scalar(state(f)))
  result=rt.compute('tail_0',d,dict(role='independent_batch16',targets=[y.tolist() for y in targets]),dict(ids=[525,374]),lambda:a.margin_many(native,targets),extra_operations=[f'tail_{i}' for i in range(1,16)])
  errors=np.abs(np.asarray(full)-result);assert max(errors)<=1e-4
  assert max(result)-min(result)>1e-4
  immutable(out/'completed.json',dict(ready=True,tail_parity_max=float(max(errors)),sensitivity=float(max(result)-min(result)),logical_forward=35,physical_suffix_batches=1,full_or_prefix_calls=19,backwards=0,scope='Historical document, independent batch rows, no scientific cohort selection'))
 finally:a.close()
if __name__=='__main__':main()
