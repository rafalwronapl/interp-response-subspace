import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from research_checks.gate import read,sha,digest
from analysis.A_L21_learned_20261103.common import unwrap,array
from analysis.AB_five_experiments_20261103.A_L21.adapter_v4 import state,unpack
from analysis.AB_budget_readout_20261103.worker import phi,cells
from analysis.A_L21_learned_20261103.models import PARAMS,fit,predict
from analysis.A_transfer_next_20261103.models import unpack as model_un
R=Path(__file__).resolve().parents[3];B=R/'analysis/A_transfer_next_20261103';O=B/'run';corpus=read(B/'corpus.json');cfg=read(B/'config.json');prior=dict(np.load(R/cfg['prior']));U=prior['B5_U']
records=[]
for l in (O/'ledger.jsonl').read_text().splitlines():
 e=json.loads(l);assert digest(e['record'])==e['sha256'];records.append(e['record'])
ats={e['operation']:e for e in records if e['event']=='attempt'}
def op(k):
 e=ats[k];c=unwrap(O/'cache'/(e['request']+'.json'));assert digest(c['request'])==e['request'];return c
ad=read(B/'adoption_index.json')['entries']
def adopt(k):return unwrap(R/ad[k]['cache'])['result']
models=unwrap(O/'MODELS.json')['bundles'];fitrows=[]
for d in corpus['documents']:
 s=state(op(d['id']+'_natural')['result'])
 for c in cells(d):
  pref=unpack(op(c['id']+'_prefix')['result']);t=state(op(c['id']+'_target')['result']);x=phi(pref,s,prior);P=np.asarray(x@prior['B5_W']@U.T,np.float32).astype(float);fitrows.append(dict(cohort=d['cohort'],block=d['block'],x=x,D=t['y']-s['y'],P=P))
with threadpool_limits(limits=2):
 for ci in range(3):
  rr=[r for r in fitrows if r['cohort']==ci];x=np.stack([r['x'] for r in rr]);D=np.stack([r['D'] for r in rr]);P=np.stack([r['P'] for r in rr]);blocks=np.array([r['block'] for r in rr])
  for method,y in [('C',(D-P)@U),('same_U',D@U),('Direct',D)]:
   losses=[]
   for param in PARAMS:
    errs=[]
    for b in sorted(set(blocks)):
     va=blocks==b;pr=predict(fit(x[~va],y[~va],param),x[va]);errs.append(float(np.mean(np.sum((y[va]-pr)**2,axis=1))))
    losses.append(float(np.mean(errs)))
   scores=models[ci]['scores'][method];np.testing.assert_allclose(losses,[s['loss'] for s in scores['candidates']],rtol=1e-10,atol=1e-12);assert int(np.argmin(losses))==scores['selected']
forecast_json=json.loads((O/'TEST_forecasts.json').read_text());frozen=unwrap(O/'TEST_forecasts.json');pred={k:array(v) for k,v in frozen['arrays'].items()};idx={k:i for i,k in enumerate(frozen['keys'])};arms=['P']+[f'{m}_{c}' for c in range(3) for m in ['C','same_U','Direct']];rows=[];native_parity=[]
for d in corpus['test_documents']:
 s=state(adopt(d['id']+'_natural'))
 for c in cells(d):
  i=idx[c['id']];x=phi(unpack(adopt(c['id']+'_prefix')),s,prior);P=np.asarray(x@prior['B5_W']@U.T,np.float32).astype(float);np.testing.assert_array_equal(P,pred['P'][i])
  for ci,b in enumerate(models):
   for method,m in b['models'].items():
    raw=predict(model_un(m),x[None,:])[0];expect=P+raw@U.T if method=='C' else raw@U.T if method=='same_U' else raw;np.testing.assert_allclose(expect,pred[f'{method}_{ci}'][i],rtol=1e-9,atol=1e-10)
  t=state(adopt(c['id']+'_target'));ys=[np.asarray(s['y'].astype(float)+pred[a][i],np.float32) for a in arms]+[t['y'],s['y']];cache=op(c['id']+'_P');request=cache['request'];np.testing.assert_array_equal(request['intervention']['targets'],ys);assert request['readout']==corpus['heads'];assert ats[c['id']+'_P']['utc']>=forecast_json['created_utc'];values=np.asarray(cache['result']);assert values.shape==(12,2)
  row=unwrap(O/'rows'/(c['id']+'.json'))
  for j,head in enumerate(['is_are','has_have']):
   for n,a in enumerate(arms+['native','zero']):assert row['margins'][head][a]==values[n,j]
  native_parity.append(abs(values[10,0]-(t['logits'][1]-t['logits'][0])));rows.append(row)
assert max(native_parity)<=1e-4 and len(rows)==1024
summary=unwrap(O/'summary.json');blocks=sorted({r['block'] for r in rows});boot=np.random.default_rng(3111038).integers(0,64,(1999,64));results={}
for head in ['is_are','has_have']:
 ms={}
 for arm in arms+['zero']:
  b=np.array([np.mean([(r['margins'][head][arm]-r['margins'][head]['native'])**2 for r in rows if r['block']==block]) for block in blocks]);ms[arm]=b;assert abs(b.mean()-summary['readouts'][head]['metrics'][arm]['MSE'])<1e-14
 delta=np.mean([ms[f'C_{ci}']-ms[f'Direct_{ci}'] for ci in range(3)],axis=0);interval=np.quantile(delta[boot].mean(1),[.025,.975]);np.testing.assert_array_equal(interval,summary['readouts'][head]['nominal95']);assert delta.mean()==summary['readouts'][head]['mean_C_Direct_delta'];results[head]=dict(delta=float(delta.mean()),nominal95=interval.tolist())
logical=sum(e['logical_forward'] for e in ats.values());assert len(ats)==1248 and logical==12516
out=dict(passed=True,physical_calls=1248,logical_calls=12516,backward=0,conditions=1024,new_fit_labels=96,all_fresh_fit_CV_choices_reproduced=True,all_forecasts_and_suffix_targets_recomputed=True,old_native_isare_maxparity=max(native_parity),results=results,pretrained_calls=0)
(B/'review/independent_completion_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
