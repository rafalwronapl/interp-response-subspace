"""Small pure-NumPy helpers; no native model imports."""
from pathlib import Path
import base64,hashlib,json
import numpy as np
from research_checks.gate import digest
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unwrap(p):
    v=read(p)
    if digest(v['payload'])!=v['payload_sha256']:raise ValueError('Corrupt immutable source '+str(p))
    return v['payload']
def array(v):
    a=np.frombuffer(base64.b64decode(v['data']),dtype=np.dtype(v['dtype'])).reshape(v['shape']).copy()
    if not np.isfinite(a).all():raise ValueError('Nonfinite tensor')
    return a
def packed(a):
    a=np.ascontiguousarray(a)
    return dict(dtype=a.dtype.str,shape=list(a.shape),data=base64.b64encode(a.tobytes()).decode())
def rel(p):return Path(p).relative_to(ROOT).as_posix()
