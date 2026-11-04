"""Bind actual completed outputs and launch receipt for independent review."""
import argparse
from pathlib import Path
from analysis.C2_fresh_fit_holdout_20261102.worker import ROOT,read,sha,put,receipt
BASE=Path(__file__).parent
def main(name,snapshot,review):
 stage=BASE/name;pre=read(ROOT/snapshot);assert pre['payload']['phase']=='pre_run';out=ROOT/read(stage/'config.json')['output'];done=receipt(out/'stage_completed.json')
 if name=='capture_base_v2':assert done['forward']==2844 and done['captures']==2816 and all(r['passed'] for r in receipt(out/'controls.json'))
 else:assert done['forward']==(96924 if name=='B_v1' else 102400) and done['LLM_loads']==0 and (stage/'summary.json').exists()
 matches=[]
 for p in (ROOT/'operations').glob('research_launch_*.json'):
  v=read(p)
  if v.get('snapshot_sha256')==pre['snapshot_sha256'] and v.get('returncode')==0:matches.append(p)
 # Parallel-runtime launches, when used for later stages, retain their own
 # immutable receipt namespace and bind the same scientific snapshot.
 for p in (ROOT/'operations/concurrency_runtime_v1').glob('launch_*.json'):
  v=read(p)
  if v.get('snapshot_sha256')==pre['snapshot_sha256'] and v.get('returncode')==0:matches.append(p)
 assert len(matches)==1,'Need unique successful actual launch receipt'
 files=dict(pre['payload']['files'])
 for p in [ROOT/snapshot,ROOT/review,matches[0],Path(__file__)]:files[p.relative_to(ROOT).as_posix()]=sha(p)
 for p in out.rglob('*'):
  if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha(p)
 for p in [stage/'summary.json',stage/'block_summary.json']:
  if p.exists():files[p.relative_to(ROOT).as_posix()]=sha(p)
 put(stage/'completion_manifest_v1.json',dict(files=files));contract=read(stage/'contract.json');contract['id']+='_COMPLETION';contract['purpose']='Closure of actual QBase completed stage; no scientific computations';contract['input_manifests']=[(stage/'completion_manifest_v1.json').relative_to(ROOT).as_posix()];put(stage/'completion_contract_v1.json',contract);print(name,'completion metadata ready')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--review',required=True);a=p.parse_args();main(a.stage,a.snapshot,a.review)
