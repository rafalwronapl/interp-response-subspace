"""Bounded passive queue: no lock mutation, no silent retry of a model attempt."""
from pathlib import Path
import json,time,subprocess,sys
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
COMPLETE=ROOT/'analysis/AB_five_experiments_20261103/B_replication/run/completed.json'
LOCK=ROOT/'operations/research_worker.lock'
B_RECEIPT=ROOT/'operations/research_launch_535e861ebdaf431fbad27c82fb5d6588.json'
start=time.monotonic()
def status(stage,**kw):
 p=BASE/'queue_status_v4.tmp';p.write_text(json.dumps(dict(stage=stage,elapsed=time.monotonic()-start,**kw)));p.replace(BASE/'queue_status_v4.json')
while time.monotonic()-start<12*3600:
 if B_RECEIPT.exists():
  result=json.loads(B_RECEIPT.read_text())
  if result['returncode']!=0:
   status('stopped_B_failed',returncode=result['returncode']);sys.exit(2)
 if COMPLETE.exists() and not LOCK.exists():break
 if B_RECEIPT.exists() and not COMPLETE.exists():
  status('stopped_B_missing_completion');sys.exit(2)
 status('waiting_B_completion_and_free_lock',B_completed=COMPLETE.exists(),lock_exists=LOCK.exists())
 time.sleep(15)
else:
 status('bounded_timeout_worker_untouched');sys.exit(2)
status('launching_pilot_v4')
cmd=[sys.executable,'-m','research_checks','launch','--snapshot','analysis/AB_five_experiments_20261103/A_L21/pilot_snapshot_v4.json','--review','analysis/AB_five_experiments_20261103/A_L21/pilot_review_v4.json']
with (BASE/'pilot_v4_launch.log').open('w') as stdout,(BASE/'pilot_v4_launch.err').open('w') as stderr:
 p=subprocess.run(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
status('pilot_v4_returned',returncode=p.returncode,completed=(BASE/'pilot_run_v4/completed.json').exists())
sys.exit(p.returncode)
