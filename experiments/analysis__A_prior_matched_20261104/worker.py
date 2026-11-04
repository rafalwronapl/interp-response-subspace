"""Prospective matched-prior experiment; canonical review launcher only."""
from pathlib import Path
import uuid,numpy as np,torch
from research_checks.gate import digest
from research_checks.persistence import immutable
from analysis.Qwen06_ABC_extension_20261102.bundle_runtime import BundleRuntime
from analysis.A_L21_learned_20261103.common import read,sha,unwrap,array,packed
from analysis.AB_five_experiments_20261103.A_L21.adapter_v4 import state,unpack,scalar
from analysis.AB_budget_readout_20261103.worker import cells,phi,outputs,status
from e031.loading import load
from .adapter import Adapter
from .models import train,forecast
BASE=Path(__file__).parent;ROOT=BASE.parents[1]
METHODS=['C','same_U','Direct','Direct_P'];ARMS=['P']+[f'{m}_{c}' for c in range(6) for m in METHODS];BATCH=ARMS+['native','natural']
def chunks(v):return [v[j:j+16] for j in range(0,len(v),16)]
def schedule(fitdocs,testdocs):
 ops=[]
 for docs,roles in [(fitdocs,['prefix','target']),(testdocs,['prefix','target']+BATCH)]:
  for d in docs:
   ops.append(dict(id=d['id']+'_natural',kind='forward'))
   for c in cells(d):
    for role in roles:ops.append(dict(id=c['id']+'_'+role,kind='forward'))
 for q in range(2):
  for role in [f'full{i}' for i in range(3)]+[f'tail{i}' for i in range(27)]:ops.append(dict(id=f'qual{q}_{role}',kind='forward'))
 return dict(operations=ops)
def summarize(rows):
 blocks=sorted({r['block'] for r in rows});idx=np.random.default_rng(4111042).integers(0,len(blocks),(1999,len(blocks)));metrics={}
 for arm in ARMS+['zero']:
  v=np.array([np.mean([(r['margins'][arm]-r['margins']['native'])**2 for r in rows if r['block']==b]) for b in blocks]);metrics[arm]=dict(MSE=float(v.mean()),block_MSE=v.tolist(),vector_L2=float(np.mean([r['vector_L2'][arm] for r in rows])))
 contrasts={}
 for left,right in [('C','Direct_P'),('C','same_U'),('Direct_P','Direct'),('C','Direct'),('same_U','Direct')]:
  per=np.array([np.array(metrics[f'{left}_{c}']['block_MSE'])-metrics[f'{right}_{c}']['block_MSE'] for c in range(6)]);v=per.mean(0);means=per.mean(1)
  contrasts[left+'-'+right]=dict(mean_delta_MSE=float(v.mean()),conditional_block_nominal95=np.quantile(v[idx].mean(1),[.025,.975]).tolist(),per_fit_delta_MSE=means.tolist(),fit_SD=float(means.std(ddof=1)),fit_range=[float(means.min()),float(means.max())],fits_negative=int((means<0).sum()))
 groups={}
 for field in ['branch','dose','S','A']:
  for value in sorted({str(r[field]) for r in rows}):
   rr=[r for r in rows if str(r[field])==value];groups[field+'='+value]=dict(conditions=len(rr),MSE={a:float(np.mean([(r['margins'][a]-r['margins']['native'])**2 for r in rr])) for a in ARMS+['zero']})
 return dict(blocks=len(blocks),conditions=len(rows),metrics=metrics,contrasts=contrasts,subgroups=groups,primary=contrasts['C-Direct_P'],primary_name='mean6FIT C32 minus Direct+P32, block-equal is/are MSE',native_minus_zero_MLP_delta_MSE=float(np.mean([(r['margins']['native']-r['margins']['zero'])**2 for r in rows])),native_minus_zero_MLP_delta_sign_counts={str(k):int(sum(np.sign(r['margins']['native']-r['margins']['zero'])==k for r in rows)) for k in [-1,0,1]},scope='Prospective fresh TEST conditional on historically selected Qwen L21/E041 F1/is-are regime and fixed E036 priors. Six disjoint FIT cohorts give descriptive variability, not adequate population power. Block CI conditions on all six FIT; it omits FIT sampling uncertainty.',bootstrap=dict(draws=1999,seed=4111042,unit='lexical_pair',primary_only=True))
def main():
 cfg=read(BASE/'config.json');manifest=read(BASE/'input_manifest.json')['files'];out=BASE/'run'
 for p,h in manifest.items():
  if sha(ROOT/p)!=h:raise ValueError('Input changed '+p)
 identity=dict(model=cfg['model'],code=digest(manifest))
 if (out/'completed.json').exists():
  done=unwrap(out/'completed.json');assert done['identity']==identity and done['files']==outputs(out);return
 corpus=read(BASE/'corpus.json');fitdocs=corpus['documents'];testdocs=corpus['test_documents'];ops=schedule(fitdocs,testdocs);assert ops==read(BASE/'operations.json')
 assert len(ops['operations'])==cfg['maximum_forward']
 prior=dict(np.load(ROOT/cfg['prior']));U=prior['B5_U'];rt=BundleRuntime(out,identity,ops)
 for r in rt.records:
  if r['event']=='attempt' and not (out/'cache'/(r['request']+'.json')).exists():raise ValueError('Uncertain prior attempt needs independent recovery')
 immutable(out/'loads'/(uuid.uuid4().hex+'.json'),dict(model=cfg['model']));torch.set_num_threads(4);adapter=Adapter(load(ROOT/cfg['checkpoint']),corpus['heads'])
 def run(ids,d,setting,fn):return rt.compute(ids[0],d,setting,dict(ids=[374,525],orientation='are-minus-is'),fn,extra_operations=ids[1:])
 try:
  natural={d['id']:run([d['id']+'_natural'],d,dict(role='FIT_natural'),lambda d=d:adapter.full(d)) for d in fitdocs};vectors=[];qual=[]
  for nd,d in enumerate(fitdocs):
   s=state(natural[d['id']]);donor=state(natural[d['donor']])
   for c in cells(d):
    x=float(s['X'])+c['dose']*(float(donor['X'])-float(s['X']));m=float(s['M']) if c['branch']=='C' else None;setting=dict(x=x,m=m,branch=c['branch'],dose=c['dose'])
    pref=unpack(run([c['id']+'_prefix'],d,dict(setting,role='FIT_prefix'),lambda:adapter.prefix(d,x,m)))
    native=run([c['id']+'_target'],d,dict(setting,role='FIT_target'),lambda:adapter.full(d,x,m));t=state(native)
    for k in ['x19','x20','x21','earlier19','earlier20','earlier21']:np.testing.assert_array_equal(pref[k],t[k])
    np.testing.assert_array_equal(s['Y'][:-1],t['Y'][:-1]);feature=phi(pref,s,prior);P=np.asarray(feature@prior['B5_W']@U.T,np.float32).astype(float)
    vectors.append(dict(cohort=d['cohort'],block=d['block'],x=feature,D=t['y']-s['y'],P=P))
    if len(qual)<2:qual.append((d,setting,s,native,t))
   status(out,'FIT',documents=nd+1,total=len(fitdocs),logical_attempts=len(rt.attempts))
  rawpath=out/'FIT_vectors.json'
  if not rawpath.exists():immutable(rawpath,dict(blocks=[r['block'] for r in vectors],cohorts=[r['cohort'] for r in vectors],arrays={k:packed(np.stack([r[k] for r in vectors])) for k in ['x','D','P']}))
  modelpath=out/'MODELS.json'
  if modelpath.exists():bundles=unwrap(modelpath)['bundles']
  else:
   bundles=[]
   for cohort in range(6):
    rr=[r for r in vectors if r['cohort']==cohort];assert len(rr)==32;bundles.append(train(*[np.stack([r[k] for r in rr]) for k in ['x','D','P']],U,np.array([r['block'] for r in rr])))
   immutable(modelpath,dict(bundles=bundles,fit_labels=len(vectors),prior=sha(ROOT/cfg['prior']),raw_FIT_sha256=sha(rawpath)))
  # Sham/native plus fixed nonzero sensitivity; exact both physical batch shapes.
  for q,(d,setting,s,native,t) in enumerate(qual):
   ys=[t['y'],np.asarray(s['y'],np.float32),np.asarray(t['y']+.01*np.linspace(.5,1.5,len(t['y'])),np.float32)];full=[]
   for i,y in enumerate(ys):
    Y=s['Y'].copy();Y[-1]=y;full.append(run([f'qual{q}_full{i}'],d,dict(setting,role='qualification_full',target=y.tolist()),lambda Y=Y:adapter.full_both(d,setting['x'],setting['m'],Y)))
   padded=[ys[i%3] for i in range(27)];tail=[]
   for start in [0,16]:
    yy=padded[start:start+16];tail+=run([f'qual{q}_tail{i}' for i in range(start,start+len(yy))],d,dict(setting,role='qualification_tail',targets=[y.tolist() for y in yy]),lambda yy=yy:adapter.both(native,yy))
   err=float(np.max(np.abs(np.asarray([full[i%3] for i in range(27)])-tail)));sensitivity=np.abs(np.asarray(tail[0])-tail[2]).tolist();passed=err<=1e-4 and sensitivity[0]>1e-8
   immutable(out/'qualification'/f'{q}.json',dict(maxabs=err,fixed_sensitivity=sensitivity,native_effect=(np.asarray(tail[0])-tail[1]).tolist(),passed=passed,batch_shapes=[16,11]));assert passed
  # All TEST inputs are target-free. Freeze every prediction before first target.
  testnatural={d['id']:run([d['id']+'_natural'],d,dict(role='TEST_natural'),lambda d=d:adapter.full(d)) for d in testdocs};prefixes={};keys=[]
  for nd,d in enumerate(testdocs):
   s=state(testnatural[d['id']]);donor=state(testnatural[d['donor']])
   for c in cells(d):
    x=float(s['X'])+c['dose']*(float(donor['X'])-float(s['X']));m=float(s['M']) if c['branch']=='C' else None;setting=dict(x=x,m=m,branch=c['branch'],dose=c['dose'])
    pref=unpack(run([c['id']+'_prefix'],d,dict(setting,role='target_free_TEST_prefix'),lambda:adapter.prefix(d,x,m)))
    if any(k in pref for k in ['y','Y','logits']):raise ValueError('Target leakage')
    keys.append(c['id']);prefixes[c['id']]=(setting,pref,phi(pref,s,prior))
   status(out,'TEST_prefix',documents=nd+1,total=len(testdocs),logical_attempts=len(rt.attempts))
  freeze=out/'TEST_forecasts.json'
  if not freeze.exists():immutable(freeze,dict(keys=keys,arrays={k:packed(v) for k,v in forecast(np.stack([prefixes[k][2] for k in keys]),prior,bundles).items()},models_sha256=sha(modelpath),scope='All25 full-vector forecasts before any TEST intervention target'))
  frozen=unwrap(freeze);assert frozen['keys']==keys and frozen['models_sha256']==sha(modelpath);pred={k:array(v) for k,v in frozen['arrays'].items()};index={k:i for i,k in enumerate(keys)};rows=[];freezehash=sha(freeze)
  for nd,d in enumerate(testdocs):
   s=state(testnatural[d['id']])
   for c in cells(d):
    dest=out/'rows'/(c['id']+'.json')
    if dest.exists():rows.append(unwrap(dest));continue
    setting,pref,_=prefixes[c['id']];native=run([c['id']+'_target'],d,dict(setting,role='TEST_target',forecasts_sha256=freezehash),lambda:adapter.full(d,setting['x'],setting['m']));t=state(native)
    for k in ['x19','x20','x21','earlier19','earlier20','earlier21']:np.testing.assert_array_equal(pref[k],t[k])
    np.testing.assert_array_equal(s['Y'][:-1],t['Y'][:-1]);D=t['y']-s['y'];ys=[np.asarray(s['y'].astype(float)+pred[a][index[c['id']]],np.float32) for a in ARMS]+[t['y'],s['y']];measured=[]
    for start in [0,16]:
     yy=ys[start:start+16];aa=BATCH[start:start+16];measured+=run([c['id']+'_'+a for a in aa],d,dict(role='TEST_suffix',condition=c,forecasts_sha256=freezehash,targets=[y.tolist() for y in yy]),lambda yy=yy:adapter.both(native,yy))
    margins={a:float(measured[i][0]) for i,a in enumerate(BATCH)};margins['zero']=margins.pop('natural');v={a:pred[a][index[c['id']]] for a in ARMS};v['zero']=np.zeros_like(D)
    row=dict(id=c['id'],block=d['block'],S=d['S'],A=d['A'],branch=c['branch'],dose=c['dose'],margins=margins,natural_prompt_margin=scalar(s),vector_L2={a:float(np.sum((vv-D)**2)) for a,vv in v.items()},native_delta_L2=float(np.sum(D.astype(float)**2)));immutable(dest,row);rows.append(row)
   status(out,'TEST_suffix',documents=nd+1,total=len(testdocs),conditions=len(rows),logical_attempts=len(rt.attempts),maximum=cfg['maximum_forward'])
  if not (out/'summary.json').exists():immutable(out/'summary.json',summarize(rows))
  assert len(rt.attempts)==cfg['maximum_forward'];physical=sum(r['event']=='attempt' for r in rt.records);assert physical==cfg['physical_maximum']
  immutable(out/'completed.json',dict(identity=identity,complete=True,conditions=len(rows),attempts=len(rt.attempts),physical_forward_calls=physical,backwards=0,model_load_attempts=len(list((out/'loads').glob('*.json'))),files=outputs(out)));status(out,'completed',conditions=len(rows))
 finally:adapter.close()
if __name__=='__main__':main()
