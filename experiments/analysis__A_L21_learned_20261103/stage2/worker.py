"""Frozen prospective non-oracle readout test; canonical reviewed launch only."""
from pathlib import Path
import json,time,uuid,importlib.metadata,numpy as np,torch
from research_checks.gate import digest
from research_checks.persistence import immutable
from analysis.Qwen06_ABC_extension_20261102.bundle_runtime import BundleRuntime
from analysis.AB_five_experiments_20261103.A_L21.adapter_v4 import Adapter,state,scalar,unpack
from e031.loading import load
from ..common import read,sha,unwrap,array,packed
from ..models import predict
BASE=Path(__file__).parent;ROOT=BASE.parents[2]
LEARNED=['P','C','same_U','Direct384','mean8'];ARMS=LEARNED+['oracle_inside'];ALL=ARMS+['zero_change']
def cells(d):
 for branch in ['B','C']:
  for dose in [.5,1.]:yield dict(id=d['id']+'_'+branch+'_'+str(dose),branch=branch,dose=dose)
def schedule(docs,quals):
 ops=[]
 for q in quals:
  for arm in ['P','C','Direct384']:
   for mode in ['full','tail']:ops.append(dict(id='qual_'+q['id']+'_'+arm+'_'+mode,kind='forward'))
 for d in docs:
  ops.append(dict(id=d['id']+'_natural',kind='forward'))
  for c in cells(d):
   for name in ['prefix','target']+ARMS:ops.append(dict(id=c['id']+'_'+name,kind='forward'))
 return dict(operations=ops)
def outputs(out):return {p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['completed.json','STATUS.json','STATUS.tmp']}
def status(out,stage,**kw):
 for delay in [0,.01,.03,.1,.2]:
  if delay:time.sleep(delay)
  try:
   tmp=out/'STATUS.tmp';tmp.write_text(json.dumps(dict(stage=stage,**kw)));tmp.replace(out/'STATUS.json');return
  except PermissionError:pass
def adopt(entry):
 receipt=unwrap(ROOT/entry['receipt']);cache=unwrap(ROOT/entry['cache'])
 if digest(cache['request'])!=receipt['request'] or cache['request']['document']!=entry['document']:raise ValueError('Adopted FIT provenance differs')
 if entry['document']['split']!='FIT0':raise ValueError('Qualification must use FIT0')
 return cache
def phi(p,s,prior):return np.concatenate([(p['x20']-s['x20'])/prior['scale'][0],(p['x21']-s['x21'])/prior['scale'][1]])/np.sqrt(2.)
def forecasts(x,prior,bundle):
 U=array(bundle['U']);models={k:{a:array(v) if isinstance(v,dict) and set(v)=={'dtype','shape','data'} else v for a,v in m.items()} for k,m in bundle['models'].items()}
 P=np.asarray(x@prior['B5_W']@prior['B5_U'].T,np.float32).astype(float)
 result=dict(P=P,C=P+predict(models['C'],x)@U.T,same_U=predict(models['same_U'],x)@U.T,Direct384=predict(models['direct'],x),mean8=P+array(bundle['mean_coefficients'])@U.T)
 if any(v.shape!=P.shape or not np.isfinite(v).all() for v in result.values()):raise ValueError('Invalid learned forecast')
 return result
def summarize(rows):
 blocks=sorted({r['block'] for r in rows});idx=np.random.default_rng(3110362).integers(0,len(blocks),(1999,len(blocks)));metrics={};contrasts={}
 for arm in ALL:
  mse=np.array([np.mean([(r['margins'][arm]-r['truth'])**2 for r in rows if r['block']==b]) for b in blocks])
  metrics[arm]=dict(MSE=float(mse.mean()),block_MSE=mse.tolist(),nominal95=np.quantile(mse[idx].mean(1),[.025,.975]).tolist(),vector_L2=float(np.mean([r['vector_L2'][arm] for r in rows])),coefficient_L2=float(np.mean([r['coefficient_L2'][arm] for r in rows])))
 for arm,reference in [('C','P'),('C','Direct384')]+[(a,'P') for a in ALL if a not in ['P','C']]:
  diff=np.array(metrics[arm]['block_MSE'])-metrics[reference]['block_MSE'];contrasts[arm+'-'+reference]=dict(delta_MSE=float(diff.mean()),nominal95=np.quantile(diff[idx].mean(1),[.025,.975]).tolist())
 groups={}
 for field in ['branch','dose','S','A']:
  for value in sorted({str(r[field]) for r in rows}):
   rr=[r for r in rows if str(r[field])==value];groups[field+'='+value]=dict(conditions=len(rr),MSE={a:float(np.mean([(r['margins'][a]-r['truth'])**2 for r in rr])) for a in ALL})
 return dict(blocks=len(blocks),conditions=len(rows),metrics=metrics,contrasts=contrasts,subgroups=groups,native_signal_MSE=float(np.mean([(r['truth']-r['natural_margin'])**2 for r in rows])),physical_suffix_batches=sum(r['physical_suffix_batches'] for r in rows),bootstrap=dict(draws=1999,seed=3110362,unit='lexical_pair',scope='nominal pointwise conditional on frozen historicalpriors and384 reused FITlabels; no family/population guarantee'),scope='C-P primary; C-Direct384 secondary. Learned predictions require native upstream prefixes, not true TESTdelta. Oracle diagnostic not upper bound; zero-change uses original natural margin.')
def main():
 cfg=read(BASE/'config.json');manifest=read(BASE/'input_manifest.json')['files'];out=BASE/'run'
 for p,h in manifest.items():
  if sha(ROOT/p)!=h:raise ValueError('Changed input '+p)
 if (out/'completed.json').exists():
  if outputs(out)!=unwrap(out/'completed.json')['files']:raise ValueError('Completed output changed')
  return
 libs=read(BASE/'library_identity.json')
 for name,v in libs['versions'].items():
  if importlib.metadata.version(name)!=v:raise ValueError('Software version changed')
 for p,h in libs['sources'].items():
  if sha(p)!=h:raise ValueError('Software source changed')
 docs=read(BASE/'corpus.json')['documents'];quals=read(BASE/'qualification.json')['contexts'];ops=schedule(docs,quals);assert ops==read(BASE/'operations.json')
 assert len(ops['operations'])==cfg['maximum_forward'];bundle=unwrap(ROOT/cfg['models']);prior=dict(np.load(ROOT/cfg['prior']));U=array(bundle['U'])
 assert sha(ROOT/cfg['prior'])==bundle['historical_prior_sha256'] and np.array_equal(U,prior['B5_U']) and U.shape==(cfg['dimension'],8)
 identity=dict(model=cfg['model'],code=digest(manifest));rt=BundleRuntime(out,identity,ops)
 for rec in rt.records:
  if rec['event']=='attempt' and not (out/'cache'/(rec['request']+'.json')).exists():raise ValueError('Uncertain previous operation; independent recovery required')
 immutable(out/'loads'/(uuid.uuid4().hex+'.json'),dict(model=cfg['model']));torch.set_num_threads(4);model=load(ROOT/cfg['checkpoint']);adapter=Adapter(model)
 def run(ids,doc,setting,fn):return rt.compute(ids[0],doc,setting,dict(ids=[525,374],orientation='are-minus-is'),fn,extra_operations=ids[1:])
 try:
  # Qualification adopts two already measured FIT contexts, never TEST labels.
  for q in quals:
   natural=adopt(q['natural'])['result'];pref=unpack(adopt(q['prefix'])['result']);target=adopt(q['target']);native=target['result'];s=state(natural);t=state(native)
   for k in ['x20','x21']:np.testing.assert_array_equal(pref[k],t[k])
   settings={k:target['request']['intervention'][k] for k in ['x','m','branch','dose']}
   saved=out/'qualification_forecasts'/(q['id']+'.json')
   if saved.exists():pred={k:array(v)[0] for k,v in unwrap(saved)['arrays'].items()}
   else:
    values=forecasts(phi(pref,s,prior)[None,:],prior,bundle);immutable(saved,dict(arrays={k:packed(v) for k,v in values.items()}));pred={k:v[0] for k,v in values.items()}
   ys=[np.asarray(s['y'].astype(float)+pred[a],np.float32) for a in ['P','C','Direct384']];full=[]
   for arm,y in zip(['P','C','Direct384'],ys):
    Y=s['Y'].copy();Y[-1]=y;z=run(['qual_'+q['id']+'_'+arm+'_full'],q['document'],dict(settings,role='qualification_full',target=y.tolist()),lambda Y=Y:adapter.full(q['document'],settings['x'],settings['m'],Y));full.append(scalar(state(z)))
   ids=['qual_'+q['id']+'_'+a+'_tail' for a in ['P','C','Direct384']];tail=run(ids,q['document'],dict(settings,role='qualification_tail_batch',targets=[y.tolist() for y in ys]),lambda:adapter.margin_many(native,ys))
   error=float(np.max(np.abs(np.array(full)-tail)));immutable(out/'qualification'/(q['id']+'.json'),dict(maxabs=error,full=full,tail=tail,passed=error<=1e-4))
   if error>1e-4:raise ValueError('FIT full/suffix numerical qualification failed')
  natural={};prefixes={};byid={d['id']:d for d in docs}
  for d in docs:natural[d['id']]=run([d['id']+'_natural'],d,dict(role='natural'),lambda d=d:adapter.full(d))
  for nd,d in enumerate(docs):
   s=state(natural[d['id']]);donor=state(natural[d['donor']])
   for c in cells(d):
    x=float(s['X'])+c['dose']*(float(donor['X'])-float(s['X']));m=float(s['M']) if c['branch']=='C' else None;setting=dict(x=x,m=m,branch=c['branch'],dose=c['dose'])
    pref=unpack(run([c['id']+'_prefix'],d,dict(setting,role='target_free_prefix'),lambda d=d,x=x,m=m:adapter.prefix(d,x,m)))
    if any(k in pref for k in ['y','Y','logits']):raise ValueError('TEST target leaked into features')
    prefixes[c['id']]=(setting,pref,phi(pref,s,prior))
   status(out,'TEST_forecast',documents=nd+1,total=len(docs))
  keys=[c['id'] for d in docs for c in cells(d)];freeze=out/'TEST_forecasts.json'
  if freeze.exists():frozen=unwrap(freeze);assert frozen['keys']==keys;predictions={k:array(v) for k,v in frozen['arrays'].items()}
  else:
   predictions=forecasts(np.stack([prefixes[k][2] for k in keys]),prior,bundle);immutable(freeze,dict(keys=keys,arrays={k:packed(v) for k,v in predictions.items()},model_payload_sha256=read(ROOT/cfg['models'])['payload_sha256'],scope='All learned deltas before TEST native targets'))
  freeze_hash=sha(freeze);rowindex={k:i for i,k in enumerate(keys)};rows=[]
  for nd,d in enumerate(docs):
   s=state(natural[d['id']]);natural_margin=scalar(s)
   for c in cells(d):
    dest=out/'rows'/(c['id']+'.json')
    if dest.exists():rows.append(unwrap(dest));continue
    setting,pref,_=prefixes[c['id']];native=run([c['id']+'_target'],d,dict(setting,role='TEST_native_target',forecasts_sha256=freeze_hash),lambda d=d,setting=setting:adapter.full(d,setting['x'],setting['m']));t=state(native)
    for k in ['x19','x20','x21','earlier19','earlier20','earlier21']:np.testing.assert_array_equal(pref[k],t[k])
    np.testing.assert_array_equal(s['Y'][:-1],t['Y'][:-1]);D=t['y']-s['y'];truth=scalar(t);v={a:predictions[a][rowindex[c['id']]] for a in LEARNED};v['oracle_inside']=v['P']+U@(U.T@(D.astype(float)-v['P']))
    margins=dict(zero_change=natural_margin);pending={};aliases={};known={t['y'].tobytes():('native',truth)}
    for arm in ARMS:
     y=np.asarray(s['y'].astype(float)+v[arm],np.float32);key=y.tobytes()
     if key in known:aliases[arm]=known[key]
     else:known[key]=(arm,None);pending[arm]=y
    if pending:
     names=list(pending);ids=[c['id']+'_'+a for a in names]
     result=run(ids,d,dict(setting,role='batched_suffix',targets={a:y.tolist() for a,y in pending.items()},forecasts_sha256=freeze_hash),lambda:adapter.margin_many(native,[pending[a] for a in names]));margins.update(zip(names,result))
    for arm,(source,value) in aliases.items():
     margins[arm]=value if source=='native' else margins[source];immutable(out/'aliases'/(c['id']+'_'+arm+'.json'),dict(source=source,value=margins[arm],charged=0))
    v['zero_change']=np.zeros_like(D,dtype=float)
    row=dict(id=c['id'],block=d['block'],S=d['S'],A=d['A'],branch=c['branch'],dose=c['dose'],truth=truth,natural_margin=natural_margin,margins=margins,vector_L2={a:float(np.sum((v[a]-D)**2)) for a in ALL},coefficient_L2={a:float(np.sum(((v[a]-D)@U)**2)) for a in ALL},native_delta_L2=float(np.sum(D.astype(float)**2)),physical_suffix_batches=int(bool(pending)))
    immutable(dest,row);rows.append(row)
   status(out,'TEST_suffix',documents=nd+1,total=len(docs),conditions=len(rows),logical_attempts=len(rt.attempts),maximum=cfg['maximum_forward'])
  assert len(rows)==4*len(docs) and len({r['block'] for r in rows})==cfg['test_blocks']
  if not (out/'summary.json').exists():immutable(out/'summary.json',summarize(rows))
  else:unwrap(out/'summary.json')
  aliases={p.stem for p in (out/'aliases').glob('*.json')};scheduled={o['id'] for o in ops['operations']};assert not rt.attempts&aliases and rt.attempts|aliases==scheduled
  physical_full_prefix=len(docs)+2*len(rows)+3*len(quals);physical_suffix=sum(r['physical_suffix_batches'] for r in rows)+len(quals)
  immutable(out/'completed.json',dict(complete=True,conditions=len(rows),blocks=cfg['test_blocks'],attempts=len(rt.attempts),aliases=len(aliases),backwards=0,physical_full_prefix_calls=physical_full_prefix,physical_suffix_batches=physical_suffix,physical_forward_calls=physical_full_prefix+physical_suffix,model_load_attempts=len(list((out/'loads').glob('*.json'))),files=outputs(out)));status(out,'completed',conditions=len(rows),logical_attempts=len(rt.attempts))
 finally:adapter.close()
if __name__=='__main__':main()
