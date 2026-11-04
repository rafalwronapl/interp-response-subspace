"""Content-bound JSON cache and idempotent receipts for future workers.

The caller supplies the full request (including model AND dependency identity).
These helpers cannot discover missing semantic dependencies for the caller.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import tempfile

from .gate import digest, read


def immutable(path, payload):
    """Publish once without exposing a half-written file; identical resume is a no-op."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = digest(payload)

    def existing():
        obj = read(path)
        if obj["payload_sha256"] != digest(obj["payload"]) or obj["payload_sha256"] != expected:
            raise ValueError("Immutable receipt differs: " + str(path))
        return obj

    if path.exists():
        return existing()
    obj = dict(payload=payload, payload_sha256=expected,
               created_utc=datetime.now(timezone.utc).isoformat())
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(obj, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic publish, no overwrite. NTFS and ordinary local Linux FS support
        # hard links; unsupported filesystems fail closed rather than weakening it.
        try:
            os.link(temporary, path)
        except FileExistsError:
            return existing()
        return obj
    finally:
        Path(temporary).unlink(missing_ok=True)


def request_key(request):
    required = {"model_identity", "code_identity", "document", "intervention", "readout"}
    if set(request) != required or any(request[k] is None for k in required):
        raise ValueError("Cache request needs complete model/code/document/intervention/readout identity")
    if not request["model_identity"] or not request["code_identity"]:
        raise ValueError("Empty model/code identity")
    if json.loads(json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False)) != request:
        raise ValueError("Cache request must use JSON-native metadata that survives roundtrip")
    return digest(request)


def cached(folder, request, compute):
    key = request_key(request)
    path = Path(folder)/(key+".json")
    if path.exists():
        obj = read(path)
        if obj["payload_sha256"] != digest(obj["payload"]):
            raise ValueError("Cache payload corrupted")
        payload = obj["payload"]
        if digest(payload["request"]) != key or payload["request"] != request:
            raise ValueError("Cache request/provenance mismatch")
        return payload["result"]
    result = compute()
    return immutable(path, dict(request=request, result=result))["payload"]["result"]
