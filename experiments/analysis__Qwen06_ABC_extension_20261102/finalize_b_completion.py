"""Bind saved documentary B outputs; no model or inference work."""
import json
from pathlib import Path
from analysis.C2_fresh_fit_holdout_20261102.worker import ROOT,read,sha
BASE=Path(__file__).parent
stage=BASE/'B_v1'
manifest=read(stage/'completion_manifest_v1.json')
for p in [BASE/'B_REPORT.md',BASE/'B_PRIMARY_SUMMARY.json',BASE/'render_b_report.py',Path(__file__)]:
    manifest['files'][p.relative_to(ROOT).as_posix()]=sha(p)
(stage/'completion_manifest_v1.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('B saved reports bound to completion metadata; zero scientific calls')
