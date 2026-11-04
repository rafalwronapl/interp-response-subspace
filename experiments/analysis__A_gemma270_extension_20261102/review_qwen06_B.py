"""Independent Q06 B saved-cell, bundle/accounting and adoption audit; zero calls."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'analysis/Qwen06_ABC_extension_20261102';ST=B/'B_v1'
def dg(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def rd(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def wr(p):
 d=p.read_bytes();v=json.loads(d);assert dg(v['payload'])==v['payload_sha256'];return v['payload'],hashlib.sha256(d).hexdigest(),v['created_utc']
def main():
 cfg=rd(ST/'config.json');cc=rd(ROOT/cfg['capture_config']);out=ROOT/cfg['output'];all_docs=rd(ROOT/cfg['corpus'])['documents'];docs={d['id']:d for d in all_docs if d['role']=='HOLDOUT'};fit={d['id']:d for d in all_docs if d['role']=='FIT' and d['cohort']=='r1' and d['split']=='FIT_MAP'};doses=[.05,.1,.25,.5,1.,2.];methods=['CLT','TAYLOR','COMPOSED_LINEAR','FROZEN','RIDGE','MEAN','ZERO'];raw,_,_=wr(out/'result.json');summary,_,_=wr(ST/'summary.json');bs,_,_=wr(ST/'block_summary.json');members=rd(ST/'completion_snapshot_v1.json')['payload']['files'];assert len(raw['rows'])==12288 and len(summary['cells'])==126
 rows={};maxerror=0.
 for r in raw['rows']:
  key=r['document'],r['dose'];assert key not in rows and r['document'] in docs and r['dose'] in doses;assert all(r[k]==docs[key[0]][k] for k in ['block','frame','side','q']);assert set(r['predictions'])==set(methods) and all(np.isfinite(v) for v in r['predictions'].values());rows[key]=r
 assert set(rows)=={(d,c) for d in docs for c in doses}
 blockmap={(v['frame'],v['dose'],v['method'],v['block']):v for v in bs};assert len(blockmap)==126*256
 for cell in summary['cells']:
  rr=[v for v in rows.values() if v['dose']==cell['dose'] and (cell['frame']=='mixture' or v['frame']==cell['frame'])];assert len(rr)==cell['contexts']==(2048 if cell['frame']=='mixture' else 1024);truth=np.array([r['native_effect'] for r in rr]);error=np.array([r['predictions'][cell['method']]-r['native_effect'] for r in rr]);n=float(np.mean(truth**2));m=float(np.mean(error**2));expect=dict(native_effect_RMS=np.sqrt(n),absolute_error_RMS=np.sqrt(m),signed_error=np.mean(error))
  for k,v in expect.items():maxerror=max(maxerror,abs(v-cell[k]));assert np.isclose(v,cell[k],rtol=1e-11,atol=1e-11)
  assert cell['relative_RMS_defined']==(n>0);assert np.isclose(cell['relative_RMS'],np.sqrt(m/n),rtol=1e-11,atol=1e-11) if n else cell['relative_RMS'] is None
  groups={}
  for r in rr:groups.setdefault(r['block'],[]).append(r)
  for block,gr in groups.items():
   v=blockmap[cell['frame'],cell['dose'],cell['method'],block];assert np.isclose(v['native_MSE'],np.mean([r['native_effect']**2 for r in gr]),rtol=1e-11,atol=1e-11) and np.isclose(v['error_MSE'],np.mean([(r['predictions'][cell['method']]-r['native_effect'])**2 for r in gr]),rtol=1e-11,atol=1e-11)
 expected={}
 for d in cc['controls']:o=['control_'+d+'_'+k for k in ['branch','head','branch_plus','branch_minus','branch_center64','head_plus','head_minus']];expected[o[0]]=o
 for d in fit:o=['fit_'+d+'_c'+str(c) for c in doses if c!=1.];expected[o[0]]=o
 for d in docs:o=[d+'__CLT_BASE_ENCODER',d+'__FROZEN_DIRECTION']+[d+'__'+k+'_c'+str(c) for c in doses for k in ['CLT','head_CLT','head_TAYLOR','head_FROZEN','head_MEAN']+(['branch','head_native','head_RIDGE'] if c!=1. else [])];expected[o[0]]=o
 identity,_,_=wr(out/'identity.json');fitseal,fitsha,ftime=wr(out/'FIT_frozen.json');assert raw['FIT_frozen_sha256']==fitsha;seen={};readhashes={};first=[]
 for p in (out/'bundles').glob('*.json'):
  v,h,_=wr(p);key=v['operations'][0];assert key in expected and key not in seen and v['operations']==expected[key] and v['logical_forward']==len(expected[key]);cache=out/'cache'/(v['request']+'.json');record,ch,_=wr(cache);req=record['request'];assert dg(req)==v['request'] and req['model_identity']==identity['model'] and req['code_identity']==identity['code'];assert req['intervention']['logical_operations']==expected[key];seen[key]=v;assert members[p.relative_to(ROOT).as_posix()]==h and members[cache.relative_to(ROOT).as_posix()]==ch
  if '__CLT_BASE_ENCODER' in key:
   d=key.split('__')[0];assert req['document']==dict(document=docs[d]) and req['intervention']['FIT_frozen_sha256']==fitsha;assert len(record['result'])==6
   for r in record['result']:assert {k:v for k,v in rows[d,r['dose']].items() if k not in ['document','block','frame','side','q']}==r
  elif key.startswith('control_'):assert record['result']['passed'] and record['result']['branch_relative']<=1e-4 and record['result']['head_absolute']<=1e-4 and record['result']['JVP_relative']<=1e-5 and record['result']['composed_relative']<=1e-4
  else:assert len(record['result'])==5 and np.asarray(record['result']).shape==(5,1024)
 assert set(seen)==set(expected) and len(seen)==2180
 attempt={};computed={};ids=[]
 for line in (out/'ledger.jsonl').read_text().splitlines():
  q=json.loads(line);r=q['record'];assert dg(r)==q['sha256'];key=r['operation'];assert key in expected and r['coupled_operations']==expected[key] and r['logical_forward']==len(expected[key]) and r['request']==seen[key]['request'];target=attempt if r['event']=='attempt' else computed;assert key not in target;target[key]=r
  if r['event']=='attempt':ids.extend(r['coupled_operations']);assert r['kind']=='forward'
  if r['event']=='attempt' and '__CLT_BASE_ENCODER' in key:first.append(r['utc'])
 assert set(attempt)==set(computed)==set(expected) and len(ids)==len(set(ids))==96924 and ftime<min(first)
 old,_,_=wr(ROOT/cfg['aligned_evaluation_result']);oldm={}
 for r in old['rows']:
  if r['arm'] in ['native','P_r1']:oldm.setdefault(r['document'],{})[r['arm']]=r['margin']
 for d in docs:assert abs((rows[d,1.]['native_effect']-rows[d,1.]['predictions']['RIDGE'])-(oldm[d]['native']-oldm[d]['P_r1']))<1e-10
 controls,_,_=wr(out/'controls.json');assert len(controls)==4 and all(c['passed'] for c in controls);seal,_,_=wr(out/'stage_completed.json');assert seal['forward']==96924 and seal['backward']==seal['LLM_loads']==0 and seal['adopted_unit_native_and_ridge_heads']==4096
 proof=dict(status='PASS',reviewer_id='/root/sol_parallel_acceptance',independent=True,cells=126,records=12288,block_cells=32256,bundles=2180,logical_attempts=96924,logical_computed=96924,adopted_unit_heads=4096,maximum_scalar_error=maxerror,FIT_frozen_before_new_TEST=True,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),out/'result.json',ST/'summary.json',ST/'block_summary.json']},limitations='All saved cells/blockmeans and every cache/bundle/request/ledger logicaloperation checked; no model/head/JVP/refit/SVD/CI replay. Unitc1native/RIDGE are inherited/exposedA/C; algebraicadoption checked. DeclaredlocalMLPinput27→CLTcomponent27/readout only; not fullCLTgraph/globalartifactquality or independentfreshBconfirmation.')
 with (B/'B_output_review_sol_gemma_peer_v1.json').open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2)
 print('Q06 B ALL126cells/2180bundles/96924logicalF independentPASS',maxerror)
if __name__=='__main__':main()
