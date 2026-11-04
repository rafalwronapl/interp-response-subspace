"""Independent coupled logical-accounting/cache-byte audit for Q06 heads."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'analysis/Qwen06_ABC_extension_20261102';ST=B/'evaluation_v2'
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def wrapped(p):
 data=p.read_bytes();v=json.loads(data);assert digest(v['payload'])==v['payload_sha256'];return v['payload'],hashlib.sha256(data).hexdigest(),v['created_utc']
def main():
 cfg=read(ST/'config.json');out=ROOT/cfg['output'];s=read(ST/'completion_snapshot_v1.json');members=s['payload']['files'];identity,_,_=wrapped(out/'identity.json');pre=read(ST/'pre_run_snapshot_v1.json');contract=read(ST/'contract.json');expected=digest(dict(config=pre['payload']['files'][(ST/'config.json').relative_to(ROOT).as_posix()],inputs=pre['payload']['files'][(ST/'input_manifest.json').relative_to(ROOT).as_posix()],sources={n:pre['payload']['files'][n] for n in contract['tracked_files'] if n.endswith('.py')}));assert identity==dict(model=digest(cfg['assets']),code=expected)
 docs={d['id']:d for d in read(ROOT/cfg['corpus'])['documents'] if d['role']=='HOLDOUT'};raw,_,_=wrapped(out/'result.json');rows={d:{} for d in docs}
 for r in raw['rows']:assert r['arm'] not in rows[r['document']];rows[r['document']][r['arm']]=r['margin']
 noD,nh,nt=wrapped(out/'noD_forecast.json');oracle,oh,ot=wrapped(out/'oracle_forecast.json');nc={r['document']:r['vector_sha256'] for r in noD['contexts']};oc={r['document']:r['vector_sha256'] for r in oracle['contexts']};assert len(nc)==len(oc)==2048 and nt<ot
 index={};op_union=set();keys=set()
 for path in (out/'bundles').glob('*.json'):
  bundle,bh,_=wrapped(path);operations=bundle['operations'];doc=operations[0].rsplit('__',1)[0];assert doc in docs and doc not in index;expected_ops=[doc+'__'+a for a in cfg['arms']];assert operations==expected_ops and bundle['logical_forward']==50 and not op_union.intersection(operations);op_union.update(operations);key=bundle['request'];assert key not in keys;keys.add(key);cache=out/'cache'/(key+'.json');v,ch,_=wrapped(cache);request=v['request'];assert set(request)=={'model_identity','code_identity','document','intervention','readout'} and digest(request)==key
  assert request['model_identity']==identity['model'] and request['code_identity']==identity['code'];assert request['document']==dict(document=docs[doc],context_sha256=nc[doc],oracle_context_sha256=oc[doc]);assert request['intervention']==dict(arms=cfg['arms'],noD_forecast_sha256=nh,oracle_forecast_sha256=oh,logical_operations=expected_ops);assert request['readout']==dict(answer_ids=cfg['assets']['answer_ids'],orientation='is-minus-are');assert set(v['result'])==set(cfg['arms']) and all(np.isfinite(m) for m in v['result'].values());assert v['result']==rows[doc]
  assert members[path.relative_to(ROOT).as_posix()]==bh and members[cache.relative_to(ROOT).as_posix()]==ch;index[doc]=dict(request=key,operations=operations,bundle_sha256=bh,cache_sha256=ch)
 assert len(index)==2048 and len(op_union)==102400 and {p.stem for p in (out/'cache').glob('*.json')}==keys and op_union=={r['id'] for r in read(ROOT/cfg['schedule'])['operations']}
 attempts={};computed={};timestamps=[]
 for line in (out/'ledger.jsonl').read_text().splitlines():
  v=json.loads(line);r=v['record'];assert digest(r)==v['sha256'];ops=r['coupled_operations'];doc=ops[0].rsplit('__',1)[0];assert r['request']==index[doc]['request'] and ops==index[doc]['operations'] and r['logical_forward']==50 and r['operation']==ops[0];dest=attempts if r['event']=='attempt' else computed;assert r['event'] in ['attempt','computed'] and doc not in dest;dest[doc]=r
  if r['event']=='attempt':assert r['kind']=='forward';timestamps.append(r['utc'])
 assert set(attempts)==set(computed)==set(docs) and ot<min(timestamps);assert all(attempts[d]['utc']<=computed[d]['utc'] for d in docs)
 for name,h in noD['fit_hashes'].items():assert members[(out/name).relative_to(ROOT).as_posix()]==h
 for group in ['r1','r2','r3']:
  u,_,_=wrapped(out/'fits'/('U_'+group+'.json'));U=np.asarray(u['U']);assert U.shape==(1024,8) and np.allclose(U.T@U,np.eye(8),atol=1e-10)
 seal,_,_=wrapped(out/'stage_completed.json');assert seal['forward']==102400 and seal['backward']==0 and seal['LLM_loads']==0 and seal['unique_gradient_points']==6144 and seal['analytic_gradient_evaluations']==18432
 proof=dict(status='PASS',reviewer_id='/root/sol_parallel_acceptance',independent=True,batches=2048,logical_attempts=102400,logical_computed=102400,cache_bundle_bytes_joined=4096,all_cached_head_values_equal_saved_rows=True,noD_seal=nt,oracle_seal=ot,first_finite_head_attempt=min(timestamps),all24fit_receipt_hashes_sealed_before_heads=True,U0_orthonormality=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations='All2048bundle/cache request identities and actualbytes/logicaloperation expansion/results/ledger chronology checked, no native/head calls or repeatedbootstrap. All24FIT receipt hashes joined, U0orthonormalitychecked; ridge fits not re-solved/SVD not rerun. Fullcapturedata/source/selector and scalar/perblock/pair/SE proofs reused separately. No claim that oracle and noD differ only by removingD: correctionvectors andscores both differ.')
 with (B/'bundle_output_review_sol_gemma_peer_v1.json').open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2)
 print('Q06 actualALL2048bundles/102400logicalops independentPASS')
if __name__=='__main__':main()
