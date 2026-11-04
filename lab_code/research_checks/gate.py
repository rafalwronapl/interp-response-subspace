"""Hash-bound, fail-closed pre-run checks and independent review receipts.

This is a workflow guard, not a sandbox or authenticated agent identity service.
The orchestrating agent must obtain the review from an actual separate agent.
"""
import hashlib
from contextvars import ContextVar
from functools import wraps
from importlib import metadata
import json
import math
import platform
import re
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .reachability import analyze, integer

REQUIRED_CATEGORIES = {"hooks", "data", "pilot", "accounting", "restart"}
REVIEW_TOPICS = {"design", "implementation", "data", "cost", "claims"}

# A fresh content-hash map belongs to exactly one public snapshot/preflight/
# manifest invocation. Nested calls share it; successive verify_review calls,
# lock boundaries and child processes do not. No mtime or persistent cache.
_HASHES = ContextVar("research_checks_hashes", default=None)


def _verification_scope(function, *, fresh=False):
    @wraps(function)
    def scoped(*args, **kwargs):
        if not fresh and _HASHES.get() is not None:
            return function(*args, **kwargs)
        token = _HASHES.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _HASHES.reset(token)
    return scoped


def _fresh_verification_scope(function):
    return _verification_scope(function, fresh=True)


def read(path):
    file = Path(path).resolve()
    data = file.read_bytes()
    hashes = _HASHES.get()
    if hashes is not None:
        actual = hashlib.sha256(data).hexdigest()
        if file in hashes and hashes[file] != actual:
            raise ValueError("Input changed within verification: " + str(file))
        # Bind parsed metadata to the exact bytes actually read, rather than a
        # separate later file read that could describe another JSON revision.
        hashes[file] = actual
    return json.loads(data.decode("utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sha(path):
    file = Path(path).resolve()
    hashes = _HASHES.get()
    if hashes is None:
        return _file_sha(file)
    if file not in hashes:
        hashes[file] = _file_sha(file)
    return hashes[file]


def environment_identity():
    """Detect ordinary interpreter/package upgrades without importing ML libraries.

Distribution versions are not hashes of every installed binary. Workers still
pin model/dtype/backend and validate their numerical controls.
"""
    executable = Path(sys.executable).resolve()
    distributions = []
    for distribution in metadata.distributions():
        package_metadata = distribution.metadata
        # The standard version property reads metadata['Version'] again.
        # Reuse this invocation's observed metadata, preserving custom overrides.
        version = (package_metadata['Version']
                   if getattr(type(distribution), 'version', None) is metadata.Distribution.version
                   else distribution.version)
        distributions.append((package_metadata.get('Name', ''), version))
    distributions.sort()
    flag_names = ("debug", "inspect", "interactive", "optimize", "dont_write_bytecode",
                  "no_user_site", "no_site", "ignore_environment", "verbose",
                  "bytes_warning", "quiet", "hash_randomization", "isolated",
                  "dev_mode", "utf8_mode", "warn_default_encoding", "safe_path",
                  "int_max_str_digits")
    return dict(executable=str(executable), executable_sha256=sha(executable),
                python=sys.version, implementation=sys.implementation.name,
                prefix=str(Path(sys.prefix).resolve()), platform=sys.platform,
                machine=platform.machine(), optimize=sys.flags.optimize,
                flags={k: getattr(sys.flags, k) for k in flag_names if hasattr(sys.flags, k)},
                distributions=distributions)


def local(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Expected a nonempty workspace-relative path")
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Path escapes workspace: " + name)
    return result


def create(path, obj):
    """Receipts are write-once; do not rewrite timestamps on resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def pointer(value, path):
    if path == "":
        return value
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("Use a JSON pointer, e.g. /metrics/MAE")
    for token in path[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def compare(actual, op, expected):
    if op == "eq":
        return type(actual) is type(expected) and actual == expected
    if op not in {"le", "ge"}:
        raise ValueError("Unsupported comparison")
    for value in (actual, expected):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Threshold comparison requires finite numbers, not booleans")
    return actual <= expected if op == "le" else actual >= expected


def contract(root, path):
    obj = read(local(root, path))
    required = {"schema_version", "id", "kind", "primary_agent", "purpose",
                "tracked_files", "tracked_trees", "input_manifests", "gates", "circuits",
                "structural_scope", "entrypoint", "schedule", "budget"}
    if set(obj) != required or obj["schema_version"] != 1:
        raise ValueError("Unsupported/incomplete contract fields")
    if obj["kind"] not in {"maintenance", "experiment"}:
        raise ValueError("Invalid work kind")
    for field in ("id", "primary_agent", "purpose", "structural_scope"):
        if not isinstance(obj[field], str) or not obj[field].strip():
            raise ValueError("Missing " + field)
    if not all(isinstance(obj[k], list) for k in ("tracked_files", "tracked_trees", "input_manifests")):
        raise ValueError("Tracked inputs must be lists")
    if not isinstance(obj["gates"], list) or not obj["gates"]:
        raise ValueError("No evidence gates")
    if not isinstance(obj["circuits"], list):
        raise ValueError("circuits must be a list")
    if obj["kind"] == "maintenance":
        if any(obj[x] is not None for x in ("entrypoint", "schedule", "budget")):
            raise ValueError("Maintenance cannot launch an experiment")
    else:
        entry = obj["entrypoint"]
        if (not isinstance(entry, dict) or set(entry) != {"module", "args"}
                or not re.fullmatch(r"[a-zA-Z_]\w*(\.[a-zA-Z_]\w*)*", entry["module"])
                or not isinstance(entry["args"], list)
                or not all(isinstance(a, str) for a in entry["args"])):
            raise ValueError("Invalid Python module entrypoint")
        module = entry["module"].replace(".", "/")
        if not (local(root, module+".py").is_file() or local(root, module+"/__main__.py").is_file()):
            raise ValueError("Worker must be a local, tracked Python module")
        if not isinstance(obj["budget"], dict) or set(obj["budget"]) != {"forward", "backward"}:
            raise ValueError("Explicit operation caps required")
        for kind, cap in obj["budget"].items():
            integer(cap, kind+" cap")
    return obj


@_verification_scope
def preflight(root, path):
    root = Path(root).resolve()
    obj = contract(root, path)
    outcomes = []
    ids = set()
    categories = set()
    for gate in obj["gates"]:
        if set(gate) != {"id", "category", "evidence", "pointer", "op", "expected", "reason"}:
            raise ValueError("Invalid evidence gate fields")
        if not gate["id"] or gate["id"] in ids or not gate["reason"]:
            raise ValueError("Duplicate/missing gate id or reason")
        ids.add(gate["id"])
        categories.add(gate["category"])
        actual = pointer(read(local(root, gate["evidence"])), gate["pointer"])
        outcomes.append(dict(id=gate["id"], ok=compare(actual, gate["op"], gate["expected"]),
                             actual=actual, expected=gate["expected"]))
    for item in obj["circuits"]:
        if set(item) != {"id", "spec"} or item["id"] in ids:
            raise ValueError("Invalid/duplicate circuit id")
        ids.add(item["id"])
        result = analyze(read(local(root, item["spec"])))
        outcomes.append(dict(id=item["id"], **result))
    if obj["kind"] == "experiment":
        if not REQUIRED_CATEGORIES <= categories:
            raise ValueError("Missing required evidence categories: " + str(REQUIRED_CATEGORIES-categories))
        schedule = read(local(root, obj["schedule"]))
        if set(schedule) != {"operations"} or not isinstance(schedule["operations"], list):
            raise ValueError("Concrete operation schedule required")
        if not schedule["operations"] and any(obj["budget"].values()):
            raise ValueError("Empty schedule requires zero forward/backward caps")
        counts = Counter()
        labels = set()
        for op in schedule["operations"]:
            if set(op) != {"id", "kind"} or not isinstance(op["id"], str) or not op["id"]:
                raise ValueError("Invalid scheduled operation")
            if op["id"] in labels or op["kind"] not in {"forward", "backward"}:
                raise ValueError("Duplicate operation or unsupported kind")
            labels.add(op["id"])
            counts[op["kind"]] += 1
        for kind, cap in obj["budget"].items():
            outcomes.append(dict(id="scheduled_"+kind, ok=counts[kind] <= cap,
                                 actual=counts[kind], expected_maximum=cap))
    manifest(root, path)
    return dict(ok=all(x["ok"] for x in outcomes), checks=outcomes)


@_verification_scope
def manifest(root, path):
    root = Path(root).resolve()
    obj = contract(root, path)
    names = {path, "AGENTS.md", "plans/RESEARCH_REVIEW_POLICY.md"}
    names.update(obj["tracked_files"])
    names.update(g["evidence"] for g in obj["gates"])
    names.update(c["spec"] for c in obj["circuits"])
    # A pinned manifest alone does not bind the current bytes of its members.
    # Resolve and verify every declared input before allowing a reviewed launch.
    for name in obj["input_manifests"]:
        declared = read(local(root, name))
        if not isinstance(declared, dict) or set(declared) != {"files"} or not declared["files"]:
            raise ValueError("Input manifest must contain a nonempty files mapping")
        for member, expected in declared["files"].items():
            if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
                raise ValueError("Invalid input SHA256: " + member)
            if sha(local(root, member)) != expected:
                raise ValueError("Input manifest member changed: " + member)
            names.add(member)
        names.add(name)
    if obj["schedule"]:
        names.add(obj["schedule"])
    trees = list(obj["tracked_trees"]) + ["research_checks"]
    for tree in trees:
        folder = local(root, tree)
        if not folder.is_dir():
            raise ValueError("Missing tracked source directory: " + tree)
        for file in sorted(folder.rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                names.add(file.relative_to(root).as_posix())
    result = {}
    for name in sorted(names):
        file = local(root, name)
        if not file.is_file():
            raise ValueError("Missing tracked file: " + name)
        result[file.relative_to(root).as_posix()] = sha(file)
    if obj["entrypoint"]:
        module = obj["entrypoint"]["module"].replace(".", "/")
        if not any(p in result for p in (module+".py", module+"/__main__.py")):
            raise ValueError("Worker is outside the reviewed manifest")
    return result


@_verification_scope
def snapshot(root, path, phase):
    if phase not in {"pre_run", "completion"}:
        raise ValueError("Invalid review phase")
    obj = contract(root, path)
    checks = preflight(root, path)
    if phase == "pre_run" and (obj["kind"] != "experiment" or not checks["ok"]):
        raise ValueError("Pre-run gates failed or work is not an experiment")
    payload = dict(schema_version=1, contract=path, phase=phase,
                   primary_agent=obj["primary_agent"], files=manifest(root, path),
                   environment=environment_identity(), preflight=checks)
    return dict(payload=payload, snapshot_sha256=digest(payload),
                created_utc=datetime.now(timezone.utc).isoformat())


@_fresh_verification_scope
def verify_review(root, snapshot_path, review_path, required_phase=None):
    root = Path(root).resolve()
    snap = read(local(root, snapshot_path))
    payload = snap["payload"]
    if snap["snapshot_sha256"] != digest(payload):
        raise ValueError("Snapshot checksum mismatch")
    if required_phase is not None and payload["phase"] != required_phase:
        raise ValueError("Wrong review phase")
    current = snapshot(root, payload["contract"], payload["phase"])
    if current["snapshot_sha256"] != snap["snapshot_sha256"]:
        raise ValueError("Review is stale: source, evidence, contract or file set changed")
    review = read(local(root, review_path))
    required = {"schema_version", "snapshot_sha256", "reviewer_id", "independent",
                "verdict", "topics_checked", "findings", "limitations"}
    if set(review) != required or review["schema_version"] != 1:
        raise ValueError("Incomplete review receipt")
    if review["snapshot_sha256"] != snap["snapshot_sha256"]:
        raise ValueError("Reviewer examined another snapshot")
    if (review["independent"] is not True or not isinstance(review["reviewer_id"], str)
            or not review["reviewer_id"].strip() or review["reviewer_id"] == payload["primary_agent"]):
        raise ValueError("Independent reviewer required")
    if review["verdict"] != "approved" or set(review["topics_checked"]) != REVIEW_TOPICS:
        raise ValueError("Independent review is incomplete or requests changes")
    if not isinstance(review["limitations"], str) or not review["limitations"].strip():
        raise ValueError("Review limitations must be stated")
    if not isinstance(review["findings"], list):
        raise ValueError("findings must be a list")
    for finding in review["findings"]:
        if set(finding) != {"id", "severity", "status", "evidence", "resolution"}:
            raise ValueError("Invalid review finding")
        if finding["severity"] not in {"blocker", "high", "medium", "low"}:
            raise ValueError("Invalid finding severity")
        if finding["status"] not in {"open", "resolved", "accepted"} or not finding["evidence"]:
            raise ValueError("Invalid finding disposition")
        if finding["severity"] in {"blocker", "high"} and finding["status"] != "resolved":
            raise ValueError("Unresolved serious finding")
        if finding["status"] == "open" or not finding["resolution"]:
            raise ValueError("Approved review requires a recorded disposition for every finding")
    return snap


def launch(root, snapshot_path, review_path):
    root = Path(root).resolve()
    # No worker import or model allocation happens until all checks succeed.
    snap = verify_review(root, snapshot_path, review_path, "pre_run")
    obj = contract(root, snap["payload"]["contract"])
    lock = root / "operations/research_worker.lock"
    token = uuid.uuid4().hex
    create(lock, dict(token=token, snapshot_sha256=snap["snapshot_sha256"],
                      purpose=obj["purpose"]))
    try:
        # Recheck after obtaining exclusivity.
        verify_review(root, snapshot_path, review_path, "pre_run")
        # A subprocess does not necessarily inherit the parent's interpreter
        # flags. Verify IN the real child before importing the expensive worker.
        result = subprocess.run([sys.executable, "-m", "research_checks._worker",
                                 "--snapshot", snapshot_path, "--review", review_path],
                                cwd=root, check=False)
        create(root / f"operations/research_launch_{token}.json",
               dict(snapshot_sha256=snap["snapshot_sha256"], returncode=result.returncode,
                    review_sha256=sha(local(root, review_path)),
                    completed_utc=datetime.now(timezone.utc).isoformat()))
        return result.returncode
    finally:
        if lock.exists() and read(lock).get("token") == token:
            lock.unlink()
