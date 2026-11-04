"""Offline completion packet; no model, fit or resampling calls."""
from pathlib import Path
import json,numpy as np
from research_checks.gate import read,sha,digest,snapshot,create
BASE=Path(__file__).parent;ROOT=BASE.parents[2];REL=BASE.relative_to(ROOT).as_posix()
def unwrap(p):
 v=read(p);assert v['payload_sha256']==digest(v['payload']);return v['payload']
def main():
 out=BASE/'main_run';completed=unwrap(out/'completed.json');assert completed['complete']
 actual={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['completed.json','STATUS.json','STATUS.tmp']}
 assert actual==completed['files']
 rows=[unwrap(p) for p in sorted((out/'rows').glob('*.json'))];assert len(rows)==1024
 summary=unwrap(out/'summary.json');blocks=sorted({r['block'] for r in rows});assert len(blocks)==64
 for arm,result in summary['metrics'].items():
  per=[]
  for b in blocks:
   vals=[(r['margins'][arm]-r['truth'])**2 for r in rows if r['block']==b and r['margins'][arm] is not None]
   per.append(float(np.mean(vals)) if vals else None)
  assert np.allclose(np.asarray(per,float),np.asarray(result['block_MSE'],float),equal_nan=True,atol=1e-12,rtol=1e-12)
  if all(v is not None for v in per):assert abs(np.mean(per)-result['MSE'])<1e-12
 ledger=[json.loads(line) for line in (out/'ledger.jsonl').read_text().splitlines()]
 for v in ledger:assert v['sha256']==digest(v['record'])
 attempts=[op for v in ledger if v['record']['event']=='attempt' for op in v['record']['coupled_operations']]
 aliases={p.stem for p in (out/'aliases').glob('*.json')};skips={p.stem for p in (out/'skips').glob('*.json')}
 planned={o['id'] for o in read(BASE/'main_schedule.json')['operations']}
 assert len(attempts)==len(set(attempts))==completed['attempts']
 assert set(attempts)|aliases|skips==planned and not set(attempts)&aliases and not set(attempts)&skips and not aliases&skips
 audit=dict(ok=True,files=len(actual),rows=len(rows),blocks=len(blocks),attempts=len(attempts),aliases=len(aliases),skips=len(skips),no_model_fit_bootstrap=True,scope='All completionhashes/rowmean aggregates/ledger coverage; bootstrap not repeated; independent review separate')
 create(BASE/'completion_audit.json',audit)
 inputs=[out/p for p in actual]+[out/'completed.json',BASE/'main_contract.json',BASE/'main_snapshot.json',BASE/'main_review.json',BASE/'completion_audit.json',BASE/'REPORT.txt',BASE/'parallel/started.json',BASE/'parallel/exit.json',BASE/'parallel/review_v2.json',BASE/'parallel/snapshot_v2.json']
 create(BASE/'completion_inputs.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in inputs}))
 create(BASE/'completion_contract.json',dict(schema_version=1,id='A_L21_fresh64_ranks_completion',kind='maintenance',primary_agent='/root/a_l21_five_campaign',purpose='Completion of two authorized A experiments; original strictfit and separate newfits/ranks',tracked_files=[REL+'/complete.py'],tracked_trees=[],input_manifests=[REL+'/completion_inputs.json'],gates=[dict(id=k,category=k,evidence=REL+'/completion_audit.json',pointer='/ok',op='eq',expected=True,reason='Completed immutable record verification, no new native calls') for k in ['hooks','data','pilot','accounting','restart']],circuits=[],structural_scope='Actual full nativeL19/L20 intervention and exactqualifiedL22-27 suffix',entrypoint=None,schedule=None,budget=None))
 create(BASE/'completion_snapshot.json',snapshot(ROOT,REL+'/completion_contract.json','completion'))
 print(audit)
if __name__=='__main__':main()
