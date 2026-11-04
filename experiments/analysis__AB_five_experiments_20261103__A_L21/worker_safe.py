"""Keep advisory Windows status locks from interrupting scientific measurements."""
from pathlib import Path
import time
from . import worker_v4 as worker
ORIGINAL_REPLACE=Path.replace
def replace(path,target):
    if path.resolve()!=worker.BASE/'main_run/STATUS.tmp' or Path(target).resolve()!=worker.BASE/'main_run/STATUS.json':return ORIGINAL_REPLACE(path,target)
    for delay in [0,.01,.03,.1,.2]:
        if delay:time.sleep(delay)
        try:return ORIGINAL_REPLACE(path,target)
        except PermissionError:pass
    return Path(target)
def main():
    Path.replace=replace
    try:worker.main()
    finally:Path.replace=ORIGINAL_REPLACE
if __name__=='__main__':main()
