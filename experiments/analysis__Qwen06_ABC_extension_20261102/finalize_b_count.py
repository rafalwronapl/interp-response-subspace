"""Final pre-run B control-center count; no experiment or frozen-file edits."""
from pathlib import Path
base=Path(__file__).parent
for name in ['prepare_b.py','B_PROTOCOL.txt']:
 p=base/name;t=p.read_text(encoding='utf-8').replace('forward=96920','forward=96924').replace('96920logicalF','96924logicalF').replace('10892nativebranches','10896nativebranches');p.write_text(t,encoding='utf-8')
