"""Finish independent Q06 capture checks from prior immutable-prefix index."""
import json,hashlib
from pathlib import Path
import numpy as np
from analysis.A_gemma270_extension_20261102.review_qwen06_capture import ROOT,B,ST,DEST,read,digest,payload,run

def main():
 run();cfg=read(ST/'config.json');out=ROOT/cfg['output'];proof=read(DEST/'capture_index_v1.json');identity=proof['identity'];ops={x['id'] for x in read(ROOT/cfg['schedule'])['operations']};assert len(ops)==2844 and set(proof['records'])==ops
 pre=read(ST/'pre_run_snapshot_v3.json');contract=read(ST/'contract.json');sources={n:pre['payload']['files'][n] for n in contract['tracked_files']};assert identity['code']==digest(dict(config=pre['payload']['files'][(ST/'config.json').relative_to(ROOT).as_posix()],sources=sources));assert identity['model']==pre['payload']['files'][cfg['model_identity']]
 ledger=[]
 for line in (out/'ledger.jsonl').read_text().splitlines():
  v=json.loads(line);assert digest(v['record'])==v['sha256'];ledger.append(v['record'])
 attempts=[r for r in ledger if r['event']=='attempt'];computed=[r for r in ledger if r['event']=='computed'];assert len(attempts)==len(computed)==2844 and {r['operation'] for r in attempts}=={r['operation'] for r in computed}==ops
 assert all(r['kind']=='forward' and r['request']==proof['records'][r['operation']]['request'] for r in attempts);assert all(r['coupled_operations']==[r['operation']] and r['request']==proof['records'][r['operation']]['request'] for r in computed)
 def result(op):return payload(ROOT/proof['records'][op]['cache'])[0]['result']
 docs=read(ROOT/cfg['corpus'])['documents'];byid={d['id']:d for d in docs};assert len(docs)==len(byid)==2816 and sum(d['role']=='HOLDOUT' for d in docs)==2048
 for d in docs:assert d['donor'] in byid and all(d[k]==byid[d['donor']][k] for k in ['block','frame','q']) and d['side']!=byid[d['donor']]['side']
 controls,_=payload(out/'controls.json');assert len(controls)==4 and {c['document'] for c in controls}==set(cfg['controls']);checks=[]
 for c in controls:
  doc=c['document'];r=result('capture_'+doc);donor=result('capture_'+byid[doc]['donor']);vs={k:result('control_'+doc+'_'+k) for k in ['sham_input','sham_output','full_head','donor_input','positive','negative','suffix']}
  parity=max(abs(vs[k]['margin'][0]-r['margin']) for k in ['sham_input','sham_output','full_head','negative']);parity=max(parity,abs(vs['suffix']-r['margin']));branch=float(np.linalg.norm(np.asarray(vs['donor_input']['y'][0])-donor['y'])/np.linalg.norm(donor['y']));sensitivity=abs(vs['positive']['margin'][0]-r['margin']);assert parity<=1e-4 and branch<=1e-4 and sensitivity>=1e-6 and np.linalg.norm(vs['negative']['patch_before'])>0 and c['passed']
  for k,v in dict(parity=parity,branch=branch,sensitivity=sensitivity).items():assert abs(c[k]-v)<=1e-12
  checks.append(dict(document=doc,parity=parity,branch=branch,sensitivity=sensitivity))
 seal,_=payload(out/'stage_completed.json');assert seal['identity']==identity and seal['forward']==2844 and seal['captures']==2816 and seal['backward']==0 and seal['LLM_loads']==1
 resultproof=dict(status='PASS',reviewer_id='/root/sol_parallel_acceptance',independent=True,operations=2844,natural_captures=2816,controls=checks,ledger_attempts=len(attempts),ledger_computed=len(computed),identity=identity,source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},index_sha256=hashlib.sha256((DEST/'capture_index_v1.json').read_bytes()).hexdigest(),limitations='All saved request/cache/token/vector/ledger/control records checked once incrementally. No checkpoint rehash, model/head calls, refit/SVD or statistical resampling. Actual official completion membership and exit0 join still required before formal approval.')
 with (DEST/'capture_checks_v1.json').open('x',encoding='utf-8') as f:json.dump(resultproof,f,indent=2)
 print('Q06 independent capture checks PASS')
if __name__=='__main__':main()
