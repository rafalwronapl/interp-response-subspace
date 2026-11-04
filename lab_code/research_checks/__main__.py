import argparse
import json
import sys
from pathlib import Path

from .gate import create, launch, preflight, snapshot, verify_review
from .reachability import analyze
from .gate import read


def main():
    parser = argparse.ArgumentParser(description="Fail-closed research design/review guard")
    sub = parser.add_subparsers(dest="command", required=True)
    circuit = sub.add_parser("circuit")
    circuit.add_argument("path")
    check = sub.add_parser("check")
    check.add_argument("contract")
    freeze = sub.add_parser("snapshot")
    freeze.add_argument("contract")
    freeze.add_argument("--phase", choices=("pre_run", "completion"), required=True)
    freeze.add_argument("--out", required=True)
    for command in ("verify", "launch"):
        p = sub.add_parser(command)
        p.add_argument("--snapshot", required=True)
        p.add_argument("--review", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    try:
        if args.command == "circuit":
            result = analyze(read(args.path))
        elif args.command == "check":
            result = preflight(root, args.contract)
        elif args.command == "snapshot":
            result = snapshot(root, args.contract, args.phase)
            create(root/args.out, result)
        elif args.command == "verify":
            snap = verify_review(root, args.snapshot, args.review)
            result = dict(ok=True, snapshot_sha256=snap["snapshot_sha256"], phase=snap["payload"]["phase"])
        else:
            return launch(root, args.snapshot, args.review)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result.get("ok", True) else 2
    except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        print(json.dumps(dict(ok=False, blocked=str(exc)), ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
