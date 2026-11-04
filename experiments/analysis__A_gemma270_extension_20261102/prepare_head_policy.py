"""Pin one reviewed head snapshot beside the unchanged exact Q06 capture."""
import argparse
from pathlib import Path
from analysis.C2_fresh_fit_holdout_20261102.worker import read,put
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent
def build():
 stage=BASE/'head_parallel_policy_v1';assert not stage.exists();stage.mkdir();old=BASE/'parallel_policy_v1';snap=BASE/'evaluation_v1/pre_run_snapshot_v1.json';review=BASE/'evaluation_v1/pre_run_review_sol_qwen_peer_v1.json';s=read(snap);r=read(review);assert r['verdict']=='approved' and r['independent'] is True and r['snapshot_sha256']==s['snapshot_sha256']
 policy=read(old/'policy.json');policy['parallel_native']['stage_snapshot_sha256']=s['snapshot_sha256'];put(stage/'policy.json',policy);put(stage/'evidence.json',dict(ready=True,scope='ONLYexact frozen head snapshot replaces completed capture snapshot; nofullmodel/noFIT; same wrapper/Qchain/locks,0maintenancecalls. Headworker includes prospectively fixed1999scalaranalysis afterimmutableheadresults.'))
 c=read(old/'contract.json');c['id']='A270_HEAD_BESIDE_EXACT_Q06_V1';c['purpose']='Only exact head snapshot pin; unchanged wrapper, locks and Q native identity.';replace={'analysis/A_gemma270_extension_20261102/parallel_policy_v1/':'analysis/A_gemma270_extension_20261102/head_parallel_policy_v1/','analysis/A_gemma270_extension_20261102/capture_v1/pre_run_snapshot_v1.json':'analysis/A_gemma270_extension_20261102/evaluation_v1/pre_run_snapshot_v1.json','analysis/A_gemma270_extension_20261102/capture_v1/pre_run_review_resume_v1.json':'analysis/A_gemma270_extension_20261102/evaluation_v1/pre_run_review_sol_qwen_peer_v1.json'}
 def rewrite(n):
  for a,b in replace.items():n=n.replace(a,b)
  return n
 c['tracked_files']=[rewrite(n) for n in c['tracked_files']]+[Path(__file__).resolve().relative_to(ROOT).as_posix(),(old/'snapshot_v1.json').relative_to(ROOT).as_posix(),(old/'review_resume_v1.json').relative_to(ROOT).as_posix()]
 for g in c['gates']:g['evidence']=rewrite(g['evidence']);g['reason']='Exact new head snapshot reviewed; unchanged lock/legacy identity implementation, no new maintenancecalls.'
 put(stage/'contract.json',c)
if __name__=='__main__':build()
