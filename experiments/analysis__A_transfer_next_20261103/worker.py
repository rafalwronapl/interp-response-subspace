"""New32-label FIT replicates, oldTEST readout extension; gated launch only."""
from pathlib import Path
import json,time,uuid,numpy as np,torch
from research_checks.gate import digest
from research_checks.persistence import immutable
from analysis.Qwen06_ABC_extension_20261102.bundle_runtime import BundleRuntime
from analysis.A_L21_learned_20261103.common import read,sha,unwrap,array,packed
from analysis.AB_five_experiments_20261103.A_L21.adapter_v4 import state,unpack
from analysis.AB_budget_readout_20261103.worker import cells,phi,outputs,status
from e031.loading import load
from .adapter import Adapter
from .models import train,forecast
BASE=Path(__file__).parent;ROOT=BASE.parents[1]
ARMS=['P']+[f'{m}_{c}' for c in range(3) for m in ['C','same_U','Direct']];BATCH=ARMS+['native','natural']
def schedule(fitdocs,testdocs):
 ops=[]
 for d in fitdocs:
  ops.append(dict(id=d['id']+'_natural',kind='forward'))
  for c in cells(d):
   for role in ['prefix','target']:ops.append(dict(id=c['id']+'_'+role,kind='forward'))
 # Two newFIT conditions, natural/native parity plus fixed sensitivity vector.
 for q in range(2):
  for role in ['full0','full1','full2','tail0','tail1','tail2']:ops.append(dict(id=f'qual{q}_{role}',kind='forward'))
 for d in testdocs:
  for c in cells(d):
   for arm in BATCH:ops.append(dict(id=c['id']+'_'+arm,kind='forward'))
 return dict(operations=ops)
def summarize(rows):
 blocks=sorted({r['block'] for r in rows});idx=np.random.default_rng(3111038).integers(0,len(blocks),(1999,len(blocks)));result={}
 for head in ['is_are','has_have']:
  metrics={}
  for arm in ARMS+['zero']:
   mse=np.array([np.mean([(r['margins'][head][arm]-r['margins'][head]['native'])**2 for r in rows if r['block']==b]) for b in blocks]);metrics[arm]=dict(MSE=float(mse.mean()),block_MSE=mse.tolist())
  diff=np.mean([np.array(metrics[f'C_{c}']['block_MSE'])-metrics[f'Direct_{c}']['block_MSE'] for c in range(3)],0)
  perfit={}
  for c in range(3):
   delta=np.array(metrics[f'C_{c}']['block_MSE'])-metrics[f'Direct_{c}']['block_MSE'];perfit[str(c)]=dict(delta_MSE=float(delta.mean()),nominal95=np.quantile(delta[idx].mean(1),[.025,.975]).tolist())
  groups={}
  for field in ['branch','dose','S','A']:
   for value in sorted({str(r[field]) for r in rows}):
    rr=[r for r in rows if str(r[field])==value];groups[field+'='+value]=dict(conditions=len(rr),MSE={a:float(np.mean([(r['margins'][head][a]-r['margins'][head]['native'])**2 for r in rr])) for a in ARMS+['zero']})
  natural=[r['margins'][head]['zero'] for r in rows]
  result[head]=dict(metrics=metrics,per_fit_C_Direct=perfit,subgroups=groups,natural_margin_range=[min(natural),max(natural)],natural_margin_sign_counts={str(k):int(sum(np.sign(v)==k for v in natural)) for k in [-1,0,1]},mean_C_Direct_delta=float(diff.mean()),nominal95=np.quantile(diff[idx].mean(1),[.025,.975]).tolist(),native_effect_MSE=float(np.mean([(r['margins'][head]['native']-r['margins'][head]['zero'])**2 for r in rows])),native_effect_sign_counts={str(k):int(sum(np.sign(r['margins'][head]['native']-r['margins'][head]['zero'])==k for r in rows)) for k in [-1,0,1]})
 return dict(blocks=len(blocks),conditions=len(rows),readouts=result,primary='has_have mean across3fits C32-Direct32',scope='Posthoc oldTEST,3fresh2blockFITs,conditional priors/fits; is_are descriptive')
def main():
 cfg=read(BASE/'config.json');manifest=read(BASE/'input_manifest.json')['files'];out=BASE/'run'
 for p,h in manifest.items():
  if sha(ROOT/p)!=h:raise ValueError('Input changed '+p)
 identity=dict(model=cfg['model'],code=digest(manifest))
 if (out/'completed.json').exists():
  done=unwrap(out/'completed.json');assert done['identity']==identity and done['files']==outputs(out);return
 corpus=read(BASE/'corpus.json');fitdocs=corpus['documents'];testdocs=corpus['test_documents'];ops=schedule(fitdocs,testdocs);assert ops==read(BASE/'operations.json')
 prior=dict(np.load(ROOT/cfg['prior']));U=prior['B5_U'];rt=BundleRuntime(out,identity,ops)
 for r in rt.records:
  if r['event']=='attempt' and not (out/'cache'/(r['request']+'.json')).exists():raise ValueError('Uncertain prior attempt needs independent recovery')
 immutable(out/'loads'/(uuid.uuid4().hex+'.json'),dict(model=cfg['model']));torch.set_num_threads(4);adapter=Adapter(load(ROOT/cfg['checkpoint']),corpus['heads'])
 def run(ids,d,setting,fn):return rt.compute(ids[0],d,setting,corpus['heads'],fn,extra_operations=ids[1:])
 adoption=read(BASE/'adoption_index.json')['entries']
 def adopt(op):
  e=adoption[op];b=unwrap(ROOT/e['bundle']);cache=unwrap(ROOT/e['cache']);assert digest(cache['request'])==e['request']==b['request'];return cache['result']
 try:
  natural={d['id']:run([d['id']+'_natural'],d,dict(role='newFIT_natural'),lambda d=d:adapter.full(d)) for d in fitdocs};vectors=[];qual=[]
  for d in fitdocs:
   s=state(natural[d['id']]);donor=state(natural[d['donor']])
   for c in cells(d):
    x=float(s['X'])+c['dose']*(float(donor['X'])-float(s['X']));m=float(s['M']) if c['branch']=='C' else None;setting=dict(x=x,m=m,branch=c['branch'],dose=c['dose'])
    pref=unpack(run([c['id']+'_prefix'],d,dict(setting,role='newFIT_prefix'),lambda:adapter.prefix(d,x,m)))
    native=run([c['id']+'_target'],d,dict(setting,role='newFIT_target'),lambda:adapter.full(d,x,m));t=state(native)
    for k in ['x20','x21']:np.testing.assert_array_equal(pref[k],t[k])
    feature=phi(pref,s,prior);P=np.asarray(feature@prior['B5_W']@U.T,np.float32).astype(float);vectors.append(dict(cohort=d['cohort'],block=d['block'],x=feature,D=t['y']-s['y'],P=P))
    if len(qual)<2:qual.append((d,setting,s,native,t))
   status(out,'newFIT',documents=len(vectors)//4)
  modelpath=out/'MODELS.json'
  if modelpath.exists():bundles=unwrap(modelpath)['bundles']
  else:
   bundles=[]
   for cohort in range(3):
    rr=[r for r in vectors if r['cohort']==cohort];bundles.append(train(*[np.stack([r[k] for r in rr]) for k in ['x','D','P']],U,np.array([r['block'] for r in rr])))
   immutable(modelpath,dict(bundles=bundles,fit_labels=96,prior=sha(ROOT/cfg['prior'])))
  # Actual full/suffix parity on newFIT, both heads, two distinct patches.
  for q,(d,setting,s,native,t) in enumerate(qual):
   ys=[t['y'],np.asarray(s['y'],np.float32),np.asarray(t['y']+.01*np.linspace(.5,1.5,len(t['y'])),np.float32)];full=[]
   for i,y in enumerate(ys):
    Y=s['Y'].copy();Y[-1]=y;full.append(run([f'qual{q}_full{i}'],d,dict(setting,role='qualification_full',target=y.tolist()),lambda Y=Y:adapter.full_both(d,setting['x'],setting['m'],Y)))
   tail=run([f'qual{q}_tail{i}' for i in range(3)],d,dict(setting,role='qualification_tail',targets=[y.tolist() for y in ys]),lambda:adapter.both(native,ys));err=float(np.max(np.abs(np.asarray(full)-tail)));sensitivity=np.abs(np.asarray(tail[0])-tail[2]).tolist()
   passed=err<=1e-4 and min(sensitivity)>1e-8
   immutable(out/'qualification'/f'{q}.json',dict(maxabs=err,fixed_sensitivity=sensitivity,native_effect=(np.asarray(tail[0])-tail[1]).tolist(),passed=passed));assert passed
  # Freeze every vector prediction globally before any newTEST readout.
  keys=[];features=[]
  for d in testdocs:
   s=state(adopt(d['id']+'_natural'))
   for c in cells(d):keys.append(c['id']);features.append(phi(unpack(adopt(c['id']+'_prefix')),s,prior))
  freeze=out/'TEST_forecasts.json'
  if not freeze.exists():immutable(freeze,dict(keys=keys,arrays={k:packed(v) for k,v in forecast(np.stack(features),prior,bundles).items()},models_sha256=sha(modelpath),scope='OldTEST new forecasts before new readout measurements'))
  frozen=unwrap(freeze);assert frozen['keys']==keys;pred={k:array(v) for k,v in frozen['arrays'].items()};index={k:i for i,k in enumerate(keys)};rows=[]
  for d in testdocs:
   s=state(adopt(d['id']+'_natural'))
   for c in cells(d):
    dest=out/'rows'/(c['id']+'.json')
    if dest.exists():rows.append(unwrap(dest));continue
    native=adopt(c['id']+'_target');t=state(native);np.testing.assert_array_equal(s['Y'][:-1],t['Y'][:-1]);ys=[np.asarray(s['y'].astype(float)+pred[a][index[c['id']]],np.float32) for a in ARMS]+[t['y'],s['y']]
    measured=run([c['id']+'_'+a for a in BATCH],d,dict(role='oldTEST_both_heads',condition=c,forecasts_sha256=sha(freeze),adopted_target=adoption[c['id']+'_target']['request'],targets=[y.tolist() for y in ys]),lambda:adapter.both(native,ys));margins={head:{a:float(measured[i][j]) for i,a in enumerate(BATCH)} for j,head in enumerate(['is_are','has_have'])}
    for head in margins:margins[head]['zero']=margins[head].pop('natural')
    row=dict(id=c['id'],block=d['block'],S=d['S'],A=d['A'],branch=c['branch'],dose=c['dose'],margins=margins);immutable(dest,row);rows.append(row)
   status(out,'oldTEST_both_readouts',conditions=len(rows))
  immutable(out/'summary.json',summarize(rows));assert len(rt.attempts)==cfg['maximum_forward']
  physical=sum(r['event']=='attempt' for r in rt.records)
  immutable(out/'completed.json',dict(identity=identity,complete=True,conditions=len(rows),attempts=len(rt.attempts),physical_forward_calls=physical,backwards=0,files=outputs(out)));status(out,'completed',conditions=len(rows))
 finally:adapter.close()
if __name__=='__main__':main()


