"""Child-process gate: validate this interpreter before importing any worker."""
import argparse
import json
from pathlib import Path
import runpy
import sys

from .gate import contract, verify_review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--review", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    try:
        snap = verify_review(root, args.snapshot, args.review, "pre_run")
        obj = contract(root, snap["payload"]["contract"])
    except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        print(json.dumps(dict(ok=False, blocked_before_worker_import=str(exc)), ensure_ascii=True))
        return 2
    entry = obj["entrypoint"]
    sys.argv = [entry["module"], *entry["args"]]
    runpy.run_module(entry["module"], run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
