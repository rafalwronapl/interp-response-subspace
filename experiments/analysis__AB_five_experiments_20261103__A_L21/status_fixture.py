import json
from pathlib import Path
from . import worker_safe as w
def main():
 count=[0];old=w.ORIGINAL_REPLACE;pause=w.time.sleep
 def locked(a,b):count[0]+=1;raise PermissionError('synthetic')
 w.ORIGINAL_REPLACE=locked;w.time.sleep=lambda _:None
 try:
  base=w.worker.BASE/'main_run';w.replace(base/'STATUS.tmp',base/'STATUS.json');assert count[0]==5
  try:w.replace(base/'cache/a.tmp',base/'cache/a.json')
  except PermissionError:pass
  else:raise AssertionError('Nonstatus failures must propagate')
 finally:w.ORIGINAL_REPLACE=old;w.time.sleep=pause
 (Path(__file__).parent/'status_fixture.json').write_text(json.dumps(dict(ok=True,trained_calls=0,status_retries=5,nonstatus_failure_propagates=True)))
if __name__=='__main__':main()
