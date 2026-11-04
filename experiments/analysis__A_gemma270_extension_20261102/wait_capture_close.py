"""Bounded metadata-only close-on-exit helper; no capture or head callback."""
import json,subprocess,sys,time
from pathlib import Path
BASE=Path('analysis/A_gemma270_extension_20261102');runtime='operations/concurrency_runtime_v1/launch_99c578af4050440c9b9270443b92b8e5.json'
for _ in range(240):
 p=Path(runtime)
 if p.exists():
  rt=json.loads(p.read_text(encoding='utf-8-sig'))
  if 'returncode' in rt:
   assert rt['returncode']==0,rt
   subprocess.run([sys.executable,'-X','utf8','-m','analysis.A_gemma270_extension_20261102.close_stage','--stage','capture_v1','--runtime',runtime],check=True)
   with (BASE/'capture_completion_snapshot_stdout_v1.log').open('x',encoding='utf-8') as out:
    subprocess.run([sys.executable,'-X','utf8','-m','research_checks','snapshot',str(BASE/'capture_v1/completion_contract_v1.json'),'--phase','completion','--out',str(BASE/'capture_v1/completion_snapshot_v1.json')],stdout=out,check=True)
   print('CAPTURE_COMPLETION_SNAPSHOT_READY',flush=True);break
 time.sleep(15)
else:raise TimeoutError('Bounded60minute metadata watcher expired, no native changes')
