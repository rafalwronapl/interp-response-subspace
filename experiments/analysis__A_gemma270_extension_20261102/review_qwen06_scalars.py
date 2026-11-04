"""Independent Q06 scalar/frame/block/fit audit; no head/fit/resampling calls."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'analysis/Qwen06_ABC_extension_20261102';ST=B/'evaluation_v2'
def load(p):
 v=json.loads(p.read_text(encoding='utf-8-sig'))
 if 'payload' in v:
  d=hashlib.sha256(json.dumps(v['payload'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest();assert d==v['payload_sha256'];return v['payload']
 return v
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cfg=load(ST/'config.json');out=ROOT/cfg['output'];raw=load(out/'result.json');fore=load(out/'noD_forecast.json');oracle=load(out/'oracle_forecast.json');saved=load(ST/'summary.json');blocksaved=load(ST/'block_summary.json');docs={d['id']:d for d in load(ROOT/cfg['corpus'])['documents'] if d['role']=='HOLDOUT'};assert len(docs)==2048
 assert raw['noD_forecast_sha256']==oracle['noD_forecast_sha256']==sha(out/'noD_forecast.json') and raw['oracle_forecast_sha256']==sha(out/'oracle_forecast.json');assert saved['source_sha256']==sha(out/'result.json');arms=cfg['arms'];m={d:{} for d in docs}
 for r in raw['rows']:
  d=r['document'];assert d in docs and r['arm'] in arms and r['arm'] not in m[d] and np.isfinite(r['margin']);assert all(r[k]==docs[d][k] for k in ['block','frame','side','q']);m[d][r['arm']]=r['margin']
 assert len(raw['rows'])==102400 and all(set(v)==set(arms) for v in m.values());groups=['r1','r2','r3'];fs={};os={};expect={(d,i,j) for d in docs for i in groups for j in groups}
 for records,index in [(fore['records'],fs),(oracle['records'],os)]:
  for r in records:
   key=r['document'],r['i'],r['j'];assert key in expect and key not in index and all(r[k]==docs[key[0]][k] for k in ['block','frame','side','q']);index[key]=r
  assert set(index)==expect
 maxima=0.;primary={}
 for frame in ['T2','T5']:
  for scope,pairs in [('primary_matched_three',[(i,i) for i in groups]),('secondary_all_nine',[(i,j) for i in groups for j in groups])]:
   rows=[]
   for d,mm in m.items():
    if docs[d]['frame']!=frame:continue
    def loss(a):return (mm[a]-mm['native'])**2
    for i,j in pairs:
     f=fs[d,i,j];o=os[d,i,j];P=loss('P_'+i);inside=loss('oracle_'+i+'_'+j);outside=loss('outside_'+i+'_'+j);proj=loss('proj_'+i+'_'+j);score=f['score_projected'];assert f['apply_projected']==(score<0) and f['apply_full']==(f['score_full']<0) and f['score_full']<=0
     row=dict(P=P,inside=inside,outside=outside,projected=proj,full=loss('full_'+i+'_'+j),direct=loss('direct_'+i+'_'+j),gated=proj if score<0 else P,delta_inside_P=inside-P,delta_inside_outside=inside-outside,oracle_correct=float(np.sign(o['score'])==np.sign(inside-P)),noD_correct=float(np.sign(score)==np.sign(proj-P)),help=float(proj<P),harm=float(proj>P),oracle_nonzero=float(o['score']!=0),noD_nonzero=float(score!=0),oracle_correct_nonzero=float(o['score']!=0 and np.sign(o['score'])==np.sign(inside-P)),noD_correct_nonzero=float(score!=0 and np.sign(score)==np.sign(proj-P)),oracle_actual_tie=float(inside==P),noD_actual_tie=float(proj==P),apply_fraction=float(score<0),energy_inside=o['energy_inside'],energy_outside=o['energy_outside']);assert all(np.isfinite(v) for v in row.values());rows.append((d,i,j,row))
   check=saved['summary'][frame+'_'+scope]
   for key,target in check['means'].items():
    value=float(np.mean([r[3][key] for r in rows]));error=abs(value-target);maxima=max(maxima,error);assert np.isclose(value,target,rtol=1e-11,atol=1e-11),(frame,scope,key,value,target)
   for i,j in pairs:
    for key,target in check['per_pair'][i+'_'+j].items():assert np.isclose(np.mean([r[3][key] for r in rows if r[1:3]==(i,j)]),target,rtol=1e-11,atol=1e-11)
   for bs in [v for v in blocksaved if v['frame']==frame and v['scope']==scope]:
    rr=[r[3] for r in rows if docs[r[0]]['block']==bs['block']]
    for key,value in bs.items():
     if key not in ['frame','scope','block']:assert np.isclose(np.mean([v[key] for v in rr]),value,rtol=1e-11,atol=1e-11)
   if scope=='primary_matched_three':
    primary[frame]=check['means'];aa=saved['aligned_A']['frames'][frame];assert aa['blocks']==256
    for arm in ['P','inside','outside']:assert np.isclose(aa['losses'][arm],check['means'][arm],rtol=1e-11,atol=1e-11)
    for contrast in ['inside_minus_P','inside_minus_outside']:
     values=[]
     for block in sorted({d['block'] for d in docs.values()}):
      rr=[r[3] for r in rows if docs[r[0]]['block']==block];key='delta_inside_P' if contrast=='inside_minus_P' else 'delta_inside_outside';values.append(np.mean([v[key] for v in rr]))
     st=aa['contrasts'][contrast];assert st['n']==256 and st['draws']==1999;assert np.isclose(st['estimate'],np.mean(values),rtol=1e-11,atol=1e-11) and np.isclose(st['se'],np.std(values,ddof=1)/16,rtol=1e-11,atol=1e-11);assert st['ci'][0]<=st['ci'][1]
    for fit in groups:
     for arm,target in aa['per_fit'][fit]['losses'].items():assert np.isclose(np.mean([r[3][arm] for r in rows if r[1]==fit]),target,rtol=1e-11,atol=1e-11)
     for arm,target in aa['per_fit'][fit]['vector_energies'].items():
      values=[(o['energy_P'] if arm=='P' else o['energy_outside'] if arm=='inside' else o['energy_inside']) for (d,i,j),o in os.items() if i==j==fit and docs[d]['frame']==frame];assert np.isclose(np.mean(values),target,rtol=1e-11,atol=1e-8)
 proof=dict(status='PASS',reviewer_id='/root/sol_parallel_acceptance',independent=True,margin_rows=102400,forecast_rows_each=18432,all_pair_frame_block_means_checked=True,primary_block_unit=256,primary_dependent_records_per_frame=3072,maximum_mean_error=maxima,primary=primary,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),out/'result.json',out/'noD_forecast.json',out/'oracle_forecast.json',ST/'summary.json',ST/'block_summary.json']},limitations='Saved arithmetic only. All102400margin rows and all forecast grids/pair/frame/block means/SE and residual-energy translation checked. No model/head/refit/SVD or1999resampling rerun; CIendpoints source/seed procedure reused, not independently regenerated. Raw Cenergy fields are errorcomponents, aligned Aenergies are residual norms. Actual full operation-accounting/cache membership and completion snapshot join remain separate.')
 with (B/'scalar_review_sol_gemma_peer_v1.json').open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2)
 print('Q06 allsaved scalar/block/perpair/SE/energy independentPASS',maxima)
if __name__=='__main__':main()
