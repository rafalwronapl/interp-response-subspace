"""Canonical approved main launcher with durable actual process return code."""
from pathlib import Path
import json,subprocess,sys
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
cmd=[sys.executable,'-m','research_checks','launch','--snapshot','analysis/AB_five_experiments_20261103/A_L21/main_snapshot.json','--review','analysis/AB_five_experiments_20261103/A_L21/main_review.json']
with (BASE/'main_launch.log').open('w') as stdout,(BASE/'main_launch.err').open('w') as stderr:
 p=subprocess.run(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
(BASE/'main_launch_exit.json').write_text(json.dumps(dict(returncode=p.returncode,completed=(BASE/'main_run/completed.json').exists())))
sys.exit(p.returncode)
