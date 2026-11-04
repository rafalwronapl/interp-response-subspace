"""One capture per native state, target-free forecast freeze, shared exact suffix."""
from pathlib import Path
import json,uuid,time,importlib.metadata
import numpy as np
import torch
from research_checks.gate import read,sha,digest
from research_checks.persistence import immutable
from analysis.Qwen06_ABC_extension_20261102.bundle_runtime import BundleRuntime as Runtime
from ab_confirmatory_20261101.native_tail_b import encode,decode
from e031.loading import load
from .adapter_v4 import Adapter,state,scalar,unpack
from .math import predict,fit,corrections,RANKS
BASE=Path(__file__).parent;ROOT=BASE.parents[2]
def unwrap(path):
 value=read(path)
 if value['payload_sha256']!=digest(value['payload']):raise ValueError('Corrupt immutable output '+str(path))
 return value['payload']
def output_files(out):return {p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['completed.json','STATUS.json','STATUS.tmp']}
def pack(p):return {k:encode(torch.as_tensor(v)) for k,v in p.items()}
def unpk(p):return {k:decode(v).numpy() for k,v in p.items()}
def cells(doc):
 for branch in ['B','C']:
  for dose in [.5,1.]:yield dict(id=doc['id']+'_'+branch+'_'+str(dose),branch=branch,dose=dose)
def arms():
 return ['old_P','old_inside','old_outside','old_oracle','zero']+[f'f{j}_P' for j in range(3)]+[f'f{j}_r{r}_{a}' for j in range(3) for r in RANKS for a in ['learned','random','random_norm']]
def schedule(docs):
 ops=[]
 for d in docs:
  ops.append(dict(id=d['id']+'_natural',kind='forward'))
  for c in cells(d):
   ops.extend(dict(id=c['id']+'_'+role,kind='forward') for role in ['prefix','target'])
   if d['split']=='TEST':ops.extend(dict(id=c['id']+'_'+arm,kind='forward') for arm in arms())
 return dict(operations=ops)
def score(rows):
 blocks=sorted({r['block'] for r in rows});results={};rng=np.random.default_rng(3110301);ix=rng.integers(0,len(blocks),size=(1999,len(blocks)))
 for arm in arms():
  per=[];counts=[]
  for b in blocks:
   rr=[r for r in rows if r['block']==b and r['margins'].get(arm) is not None]
   counts.append(len(rr));per.append(np.mean([(r['margins'][arm]-r['truth'])**2 for r in rr]) if rr else np.nan)
  a=np.asarray(per);valid=np.isfinite(a).all()
  results[arm]=dict(MSE=float(a.mean()) if valid else None,block_MSE=[float(x) if np.isfinite(x) else None for x in a],block_counts=counts,nominal95=list(np.quantile(a[ix].mean(1),[.025,.975])) if valid else None)
 old=np.array(results['old_P']['block_MSE']);contrasts={}
 for arm in ['old_inside','old_outside']:
  a=np.array(results[arm]['block_MSE']);diff=a-old
  contrasts[arm]=dict(delta_MSE=float(diff.mean()),nominal95=list(np.quantile(diff[ix].mean(1),[.025,.975])),reduction=1-float(a.mean()/old.mean()) if old.mean()>0 else None)
 rank_summary={}
 for rank in RANKS:
  for kind in ['learned','random','random_norm']:
   vals=np.asarray([results[f'f{j}_r{rank}_{kind}']['block_MSE'] for j in range(3)],dtype=float)
   bases=np.asarray([results[f'f{j}_P']['block_MSE'] for j in range(3)],dtype=float)
   paired=(vals-bases).mean(0);valid=np.isfinite(paired).all()
   rank_summary[f'r{rank}_{kind}']=dict(delta_MSE_equal_fit=float(paired.mean()) if valid else None,nominal95=list(np.quantile(paired[ix].mean(1),[.025,.975])) if valid else None)
 groups={}
 for branch in ['B','C']:
  for dose in [.5,1.]:
   rr=[r for r in rows if r['branch']==branch and r['dose']==dose]
   groups[f'{branch}_{dose}']={a:(float(np.mean([(r['margins'][a]-r['truth'])**2 for r in rr if r['margins'][a] is not None])) if any(r['margins'][a] is not None for r in rr) else None) for a in arms()}
 return dict(blocks=len(blocks),conditions=len(rows),metrics=results,primary=contrasts,rank_summary=rank_summary,subgroups=groups,bootstrap=dict(draws=1999,seed=3110301,unit='lexical_pair_block',scope='nominal conditional fixed fits; no family qualification'),scope='Strict frozen B5/F1 replication and separate fixed-new-fit rank controls; fresh pairs, no training-population claim.')
def main():
 cfg=read(BASE/'main_config.json');manifest=read(BASE/'main_inputs.json')['files']
 for p,h in manifest.items():
  if sha(ROOT/p)!=h:raise ValueError('Changed input '+p)
 libraries=read(BASE/'library_identity.json')
 for package,version in libraries['versions'].items():
  if importlib.metadata.version(package)!=version:raise ValueError('Library version changed')
 for p,h in libraries['sources'].items():
  if sha(Path(p))!=h:raise ValueError('Native library source changed')
 out=BASE/'main_run';docs=read(BASE/'corpus.json')['documents'];ops=schedule(docs)
 assert ops==read(BASE/'main_schedule.json')
 rt=Runtime(out,dict(model=cfg['model'],code=digest(manifest)),ops)
 if (out/'completed.json').exists():
  if unwrap(out/'completed.json')['files']!=output_files(out):raise ValueError('Completed stage changed')
  return
 def status(stage,**kw):
  tmp=out/'STATUS.tmp';tmp.write_text(json.dumps(dict(stage=stage,**kw)));tmp.replace(out/'STATUS.json')
 immutable(out/'loads'/(uuid.uuid4().hex+'.json'),dict(model=cfg['model']))
 torch.set_num_threads(4);model=load(ROOT/'model');adapter=Adapter(model)
 old=dict(np.load(ROOT/'runs/E036/development/predictors.npz'))
 natural={};prefixes={};newfits=[];preds={};rows=[]
 def measure(op,d,settings,fn):return rt.compute(op,d,settings,dict(ids=[525,374],orientation='are-minus-is'),fn)
 def capture_group(group,isfit):
  targets=[];dx20=[];dx21=[]
  for d in group:
   natural[d['id']]=measure(d['id']+'_natural',d,dict(role='natural'),lambda d=d:adapter.full(d))
  for n,d in enumerate(group):
   s=state(natural[d['id']]);donor=state(natural[d['donor']])
   for c in cells(d):
    x=float(s['X'])+c['dose']*(float(donor['X'])-float(s['X']));m=float(s['M']) if c['branch']=='C' else None
    settings=dict(x=x,m=m,branch=c['branch'],dose=c['dose'])
    pref=measure(c['id']+'_prefix',d,dict(settings,role='target_free_prefix'),lambda d=d,x=x,m=m:adapter.prefix(d,x,m));p=unpack(pref)
    x20=p['x20']-s['x20'];x21=p['x21']-s['x21'];prefixes[c['id']]=(settings,p,x20,x21)
    if isfit:
     target=measure(c['id']+'_target',d,dict(settings,role='FIT_native_target'),lambda d=d,x=x,m=m:adapter.full(d,x,m));t=state(target)
     for k in ['x19','x20','x21','earlier19','earlier20','earlier21']:np.testing.assert_array_equal(p[k],t[k])
     targets.append(t['y']-s['y']);dx20.append(x20);dx21.append(x21)
   status('FIT_capture' if isfit else 'TEST_forecast',document=n+1,total=len(group))
  return np.asarray(dx20),np.asarray(dx21),np.asarray(targets)
 try:
  for j in range(3):
   group=[d for d in docs if d['split']=='FIT'+str(j)]
   x20,x21,D=capture_group(group,True)
   input_hash=digest(pack(dict(dx20=x20,dx21=x21,D=D)))
   fit_path=out/f'fit_{j}.json'
   if fit_path.exists():
    saved=unwrap(fit_path);assert saved['inputs_sha256']==input_hash;p=unpk(saved['arrays'])
   else:
    p=fit(x20,x21,D,3110340+j);immutable(fit_path,dict(inputs_sha256=input_hash,arrays=pack(p),seed=3110340+j))
   newfits.append(p)
  tests=[d for d in docs if d['split']=='TEST'];capture_group(tests,False)
  keys=[c['id'] for d in tests for c in cells(d)]
  x20=np.stack([prefixes[k][2] for k in keys]);x21=np.stack([prefixes[k][3] for k in keys])
  matrices=dict(old=predict(old,x20,x21),**{f'f{j}':predict(p,x20,x21) for j,p in enumerate(newfits)})
  preds={key:{name:matrix[i] for name,matrix in matrices.items()} for i,key in enumerate(keys)}
  frozen=immutable(out/'TEST_forecasts.json',{k:pack(p) for k,p in preds.items()})
  for nd,d in enumerate(tests):
   s=state(natural[d['id']])
   for c in cells(d):
    settings,p,x20,x21=prefixes[c['id']];forecast=preds[c['id']]
    native=measure(c['id']+'_target',d,dict(settings,role='TEST_native_target',forecast_freeze=frozen['payload_sha256']),lambda d=d,settings=settings:adapter.full(d,settings['x'],settings['m']))
    t=state(native)
    for k in ['x19','x20','x21','earlier19','earlier20','earlier21']:np.testing.assert_array_equal(p[k],t[k])
    np.testing.assert_array_equal(s['Y'][:-1],t['Y'][:-1]);D=t['y']-s['y'];truth=scalar(t)
    error=D.astype(float)-forecast['old'].astype(float);inside=(error@old['B5_U'])@old['B5_U'].T;base=s['y'].astype(float)+forecast['old'].astype(float)
    values=dict(old_P=base,old_inside=base+inside,old_outside=base+error-inside,old_oracle=base+error,zero=s['y'])
    geometry={}
    for j,f in enumerate(newfits):
     P=forecast[f'f{j}'].astype(float);e=D.astype(float)-P;b=s['y'].astype(float)+P;values[f'f{j}_P']=b
     for rank in RANKS:
      ca,cb,cc=corrections(e,f['U'][:,:rank],f['random_U'][:,:rank]);tag=f'f{j}_r{rank}'
      values[tag+'_learned']=b+ca;values[tag+'_random']=b+cb;values[tag+'_random_norm']=None if cc is None else b+cc
      actual=lambda corr:float(np.linalg.norm(np.asarray(b+corr,np.float32).astype(float)-np.asarray(b,np.float32).astype(float)))
      geometry[tag]=dict(learned_norm=float(np.linalg.norm(ca)),random_norm=float(np.linalg.norm(cb)),normmatched_norm=None if cc is None else float(np.linalg.norm(cc)),rounded_learned_norm=actual(ca),rounded_random_norm=actual(cb),rounded_normmatched_norm=None if cc is None else actual(cc),undefined=cc is None)
    margins={};known={t['y'].tobytes():('native',truth)};pending={};alias={}
    for arm,y in values.items():
     op=c['id']+'_'+arm
     if y is None:
      immutable(out/'skips'/(op+'.json'),dict(reason='Undefined near-zero random direction',charged=0));margins[arm]=None;continue
     rounded=np.asarray(y,np.float32);key=rounded.tobytes()
     if key in known:
      alias[arm]=known[key];continue
     pending[arm]=rounded;known[key]=(arm,None)
    if pending:
     names=list(pending);operations=[c['id']+'_'+arm for arm in names]
     def evaluate_suffix():
      result={}
      for start in range(0,len(names),16):
       chunk=names[start:start+16];values=adapter.margin_many(native,[pending[a] for a in chunk]);result.update(zip(chunk,values))
      return result
     obtained=rt.compute(operations[0],d,dict(settings,role='coupled_suffix_batches16',targets={a:y.tolist() for a,y in pending.items()},forecast_freeze=frozen['payload_sha256']),dict(ids=[525,374],orientation='are-minus-is'),evaluate_suffix,extra_operations=operations[1:])
     margins.update(obtained)
    for arm,(source,value) in alias.items():
     margins[arm]=value if source=='native' else margins[source]
     immutable(out/'aliases'/(c['id']+'_'+arm+'.json'),dict(source=source,value=margins[arm],charged=0))
    if abs(margins['old_oracle']-truth)>1e-4:raise ValueError('Full oracle repair/native readout mismatch')
    row=dict(id=c['id'],block=d['block'],S=d['S'],A=d['A'],branch=c['branch'],dose=c['dose'],truth=truth,margins=margins,geometry=geometry,physical_suffix_batches=(len(pending)+15)//16)
    immutable(out/'rows'/(c['id']+'.json'),row);rows.append(row)
   status('TEST_suffix',documents=nd+1,total=len(tests),conditions=len(rows),operations=len(rt.attempts),max_operations=len(ops['operations']))
  if not (out/'summary.json').exists():immutable(out/'summary.json',score(rows))
  else:unwrap(out/'summary.json')
  aliases={p.stem for p in (out/'aliases').glob('*.json')};skips={p.stem for p in (out/'skips').glob('*.json')}
  assert not aliases&skips and not aliases&rt.attempts and not skips&rt.attempts
  assert aliases|skips|rt.attempts=={o['id'] for o in ops['operations']}
  immutable(out/'completed.json',dict(complete=True,conditions=len(rows),attempts=len(rt.attempts),aliases=len(aliases),skips=len(skips),scheduled_max=len(ops['operations']),model_loads=len(list((out/'loads').glob('*.json'))),result_sha256=sha(out/'summary.json'),files=output_files(out)))
  status('completed',conditions=len(rows),operations=len(rt.attempts))
 finally:adapter.close()
if __name__=='__main__':main()
