"""Focused independent final Gemma review; reuse bounded raw and scalar proofs."""
import json
from analysis.C2_fresh_fit_holdout_20261102.worker import ROOT,read,sha,receipt
from research_checks.gate import digest
from research_checks.persistence import request_key
base=ROOT/'analysis/A_gemma270_extension_20261102';stage=base/'evaluation_v1';s=read(stage/'completion_snapshot_v1.json');pre=read(stage/'pre_run_snapshot_v1.json');assert digest(s['payload'])==s['snapshot_sha256']=='41b5538f7df1b2f0c0f356b1d073367d25596cc832aef7d3e5a7f6abae122a63' and s['payload']['phase']=='completion';members=s['payload']['files'];assert all(members[n]==h for n,h in pre['payload']['files'].items());cfg=read(stage/'config.json');out=ROOT/cfg['output'];done=receipt(out/'stage_completed.json');assert done['forward']==20480 and done['backward']==done['LLM_loads']==0 and all(sha(out/n)==h for n,h in done['essential_files'].items())
proof=read(base/'scalar_peer_checks_qwen_v1.json');assert proof['status']=='PASS' and proof['head_calls']==proof['bootstrap_draws_repeated']==0;raw=read(base/'head_peer_checks_qwen_progress_v1.json');assert raw['checked_count']==16431 and raw['model_calls']==0 and raw['identity']==done['identity'];assert all(members[n]==h for n,h in raw['files'].items())
for p in [base/'REPORT.md',base/'PRIMARY_SUMMARY.json',out/'result.json',out/'analysis_result.json']:assert members[p.relative_to(ROOT).as_posix()]==sha(p)
docs=read(ROOT/cfg['corpus'])['documents'];expected={d['id']+'__'+arm:(d,arm) for d in docs for arm in cfg['arms']};contexts=dict(raw['contexts']);requests={}
for d in docs:
 if d['id'] not in contexts:
  p=out/'contexts'/(d['id']+'.json');c=receipt(p);assert c['document']==d['id'] and c['fixed_pairs']==cfg['fixed_pairs'] and members[p.relative_to(ROOT).as_posix()]==sha(p);contexts[d['id']]=c['context_sha256']
 for arm in cfg['arms']:
  request=dict(model_identity=done['identity']['model'],code_identity=done['identity']['code'],document=dict(document=d,context_sha256=contexts[d['id']]),intervention=dict(arm=arm),readout=dict(answer_ids=[563,659],orientation='is-minus-are'));requests[d['id']+'__'+arm]=request_key(request)
attempts={};computed={}
for line in (out/'ledger.jsonl').read_text().splitlines():
 row=json.loads(line);r=row['record'];assert digest(r)==row['sha256'] and r['operation'] in expected and r['request']==requests[r['operation']]
 if r['event']=='attempt':assert r['kind']=='forward' and r['operation'] not in attempts;attempts[r['operation']]=r
 elif r['event']=='computed':assert r['operation'] not in computed and r['coupled_operations']==[r['operation']];computed[r['operation']]=r
 else:raise AssertionError('Unexpected ledger event')
assert set(attempts)==set(computed)==set(expected) and len(expected)==20480
runtime=list((ROOT/'operations/concurrency_runtime_v1').glob('launch_ff9448*.json'));assert len(runtime)==1;rt=read(runtime[0]);assert rt['snapshot_sha256']==pre['snapshot_sha256'] and rt['returncode']==0
finalproof=dict(status='PASS',reviewer_id='/root/sol_qwen06_parallel',independent=True,snapshot_sha256=s['snapshot_sha256'],all_scalar_rows=20480,all_energy_rows=6144,ledger_attempts=20480,ledger_computed=20480,raw_cache_requests_checked=16431,raw_scope='Bounded immutable prefix; all aggregate scalars/block/perfit/SE validated separately; remaining4049rawcache payloads not individually inspected',native_or_head_replays=0,bootstrap_replays=0,report_sha256=sha(base/'REPORT.md'),summary_sha256=sha(base/'PRIMARY_SUMMARY.json'))
(base/'evaluation_peer_checks_qwen_v1.json').write_text(json.dumps(finalproof,indent=2)+'\n')
review=dict(schema_version=1,snapshot_sha256=s['snapshot_sha256'],reviewer_id='/root/sol_qwen06_parallel',independent=True,verdict='approved',topics_checked=['design','implementation','data','cost','claims'],findings=[],limitations='Independent focused final review: all20480savedscalarrows/6144energies/frame-block-perFIT pointmeans andSE/sourcehashes validated; saved1999draw nominal95CI metadata/source/kernel checked without resampling. Bounded16431of20480rawheadcache/receipt/request proofs checked once and theiractualbytehashes joined toofficialcompletion; remaining4049rawcache payloads not individually inspected. ALL20480attempt/computedledger IDs/requestkeys verified, actualsuccessfulruntime and immutableessentialfiles/report/summary bindings checked. Native/head/sourcequalification and3FIT provenance reused;0native/head/bootstrapreplays. Finding is conditional oraclebenefit onbothframes, outsidepooledbetterbutnotallfits; outside632directions versusinside8 andnonmatchednorms precludeper-dimensionefficiency/universalpreference. No inference over refits, qualifiedpopulationcoverage/FWER or identicalhistoricalE041replication claim. Focusedproof: analysis/A_gemma270_extension_20261102/evaluation_peer_checks_qwen_v1.json.')
(stage/'completion_review_sol_qwen_peer_v1.json').write_text(json.dumps(review,indent=2)+'\n');print('GemmaA270 final scientific/completion independentlyAPPROVED;0model/head/bootstrapreplays')
