"""Pre-freeze metadata correction: bind explicit --config output namespace."""
from pathlib import Path
base=Path(__file__).parent
for name in ['prepare_evaluation.py','prepare_b.py']:
 p=base/name;t=p.read_text(encoding='utf-8');old="args=[rel+'/config.json']";assert old in t;t=t.replace(old,"args=['--config',rel+'/config.json']");p.write_text(t,encoding='utf-8')
