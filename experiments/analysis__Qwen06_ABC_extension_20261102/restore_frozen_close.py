"""Preserve accepted closure source; put new B accounting in separate helper."""
from pathlib import Path
import hashlib,json
base=Path(__file__).parent;p=base/'close_stage.py';blob=p.read_bytes();(base/'close_b_stage.py').write_bytes(blob)
s=json.loads((base/'evaluation_v2/completion_snapshot_v1.json').read_text(encoding='utf-8'));name=p.relative_to(base.parents[1]).as_posix();expected=s['payload']['files'][name]
candidates=[blob]+[blob.replace(b'96924',v) for v in [b'96920',b'92824',b'92816']]
found=[b for b in candidates if hashlib.sha256(b).hexdigest()==expected];assert len(found)==1,'No exact frozen-source restoration candidate';p.write_bytes(found[0]);assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
print('Accepted Q closure source byte-exact restored; new B helper separate')
