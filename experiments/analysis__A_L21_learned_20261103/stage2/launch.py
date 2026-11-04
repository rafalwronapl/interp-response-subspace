"""Canonical reviewed launch, detached by caller, with durable exit status."""
from pathlib import Path
import json,subprocess,sys
BASE=Path(__file__).parent;ROOT=BASE.parents[2];REL=BASE.relative_to(ROOT).as_posix()
def main():
 cmd=[sys.executable,'-m','research_checks','launch','--snapshot',REL+'/pre_run_snapshot.json','--review',REL+'/pre_run_review.json']
 with (BASE/'launch.log').open('w') as stdout,(BASE/'launch.err').open('w') as stderr:
  p=subprocess.run(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
 (BASE/'launch_exit.json').write_text(json.dumps(dict(returncode=p.returncode,completed=(BASE/'run/completed.json').exists()))+'\n')
 return p.returncode
if __name__=='__main__':sys.exit(main())
