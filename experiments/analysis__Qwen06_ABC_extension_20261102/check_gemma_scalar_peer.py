"""Independent point/block/SE/energy validation, no head/bootstrap replay."""
import json,numpy as np
from analysis.C2_fresh_fit_holdout_20261102.worker import ROOT,read,sha,receipt
BASE=ROOT/'analysis/A_gemma270_extension_20261102';stage=BASE/'evaluation_v1';cfg=read(stage/'config.json');out=ROOT/cfg['output'];raw=receipt(out/'result.json');summary=receipt(out/'analysis_result.json');docs={d['id']:d for d in read(ROOT/cfg['corpus'])['documents']};values={};energy={};fits=['r1','r2','r3'];assert len(docs)==2048
assert summary['source_sha256']==dict(result=sha(out/'result.json'),corpus=sha(ROOT/cfg['corpus']),analyzer=sha(BASE/'analyze.py'))
for r in raw['rows']:
 key=r['document'],r['arm'];assert key not in values and r['document'] in docs and r['arm'] in cfg['arms'] and np.isfinite(r['margin']);assert all(r[k]==docs[r['document']][k] for k in ['block','frame','side','q']);values[key]=r['margin']
assert set(values)=={(doc,arm) for doc in docs for arm in cfg['arms']}
for r in raw['energies']:
 key=r['document'],r['fit'];assert key not in energy and r['fit'] in fits and r['document'] in docs;assert all(r[k]==docs[r['document']][k] for k in ['block','frame','side','q']);assert all(np.isfinite(r[k]) and r[k]>=0 for k in ['P','inside','outside']);assert np.isclose(r['P'],r['inside']+r['outside'],rtol=1e-10,atol=1e-7);energy[key]=r
assert len(energy)==6144
for frame in ['T5','T2']:
 selected=[d for d in docs.values() if d['frame']==frame];blocks=sorted({d['block'] for d in selected});s=summary['frames'][frame];assert s['blocks']==256 and s['dependent_records']==3072
 losses={(d['id'],i):{arm:(values[d['id'],arm+'_'+i]-values[d['id'],'native'])**2 for arm in ['P','inside','outside']} for d in selected for i in fits}
 blockvals={b:{arm:np.mean([losses[d['id'],i][arm] for d in selected if d['block']==b for i in fits]) for arm in ['P','inside','outside']} for b in blocks}
 for arm in ['P','inside','outside']:assert np.isclose(s['losses'][arm],np.mean([v[arm] for v in blockvals.values()]),rtol=1e-12,atol=1e-12)
 for record in s['per_block']:
  b=record['block'];assert all(np.isclose(record['losses'][arm],blockvals[b][arm],rtol=1e-12,atol=1e-12) for arm in ['P','inside','outside'])
 for key,other in [('inside_minus_P','P'),('inside_minus_outside','outside')]:
  differences=np.array([blockvals[b]['inside']-blockvals[b][other] for b in blocks]);r=s['contrasts'][key];assert r['status']=='defined' and r['draws']==1999 and r['n']==256;assert np.isclose(r['estimate'],differences.mean(),rtol=1e-12,atol=1e-12) and np.isclose(r['se'],differences.std(ddof=1)/16,rtol=1e-12,atol=1e-12);assert np.isfinite(r['ci']).all() and r['ci'][0]<r['estimate']<r['ci'][1]
 for i in fits:
  for arm in ['P','inside','outside']:
   assert np.isclose(s['per_fit'][i]['losses'][arm],np.mean([losses[d['id'],i][arm] for d in selected]),rtol=1e-12,atol=1e-12)
   assert np.isclose(s['per_fit'][i]['vector_energies'][arm],np.mean([energy[d['id'],i][arm] for d in selected]),rtol=1e-12,atol=1e-7)
primary=read(BASE/'PRIMARY_SUMMARY.json');assert primary['source_sha256']==dict(analysis_result=sha(out/'analysis_result.json'),head_result=sha(out/'result.json'))
for frame in ['T2','T5']:
 assert primary['frames'][frame]=={k:v for k,v in summary['frames'][frame].items() if k!='per_block'}
proof=dict(status='PASS',reviewer_id='/root/sol_qwen06_parallel',independent=True,head_calls=0,bootstrap_draws_repeated=0,checks=['All20480scalarvalues/docarmgrid and6144energyrows finite/metadata/orthogonalidentity','Pointlosses/frame/perFIT/blockmeans andSE independentlyvalidated','Saved1999drawCI metadata/direction/sourcehashes checked; no resampling replay','PRIMARY_SUMMARY values exactsavedanalysis projection; outside pooledbetter butnotallfits'])
(BASE/'scalar_peer_checks_qwen_v1.json').write_text(json.dumps(proof,indent=2)+'\n');print('Gemma scalar/frame/block/perFIT/energy/SE peerchecks PASS;0heads0resamples')
