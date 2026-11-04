"""Add documentary results and failed-zero-call provenance before snapshot."""
import json
from pathlib import Path
from analysis.C2_fresh_fit_holdout_20261102.worker import ROOT,read,sha
BASE=Path(__file__).parent;stage=BASE/'evaluation_v2';manifest=read(stage/'completion_manifest_v1.json');files=manifest['files']
for p in [BASE/'A_C_REPORT.md',BASE/'A_C_PRIMARY_SUMMARY.json',BASE/'render_head_report.py',Path(__file__),BASE/'evaluation_v1/launch_v1.err',BASE/'evaluation_v1/launch_v1.log',BASE/'evaluation_v1/pre_run_snapshot_v1.json',BASE/'evaluation_v1/pre_run_review_sol_gemma_peer_v1.json',BASE/'evaluation_v1/run/identity.json',BASE/'evaluation_v1/run/schedule.json']:
 files[p.relative_to(ROOT).as_posix()]=sha(p)
(stage/'completion_manifest_v1.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Actualheadcompletionmetadatafinalized, reports+0-call-v1provenance bound')
