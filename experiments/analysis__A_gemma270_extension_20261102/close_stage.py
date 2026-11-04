"""Metadata-only closure, preserving all durable attempts and provenance."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def create(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main(stage_name,runtime):
 stage=BASE/stage_name;pre=read(stage/'pre_run_snapshot_v1.json');contract=read(stage/'contract.json');rp=ROOT/runtime;rt=read(rp);assert rt.get('returncode')==0 and rt['snapshot_sha256']==pre['snapshot_sha256']
 cfg=read(stage/'config.json');out=ROOT/cfg['output'];seal=read(out/'stage_completed.json')['payload'];expected=2048 if stage_name=='capture_v1' else 20480
 assert seal['forward']==expected and seal['backward']==0;scheduled={x['id'] for x in read(ROOT/cfg['schedule'])['operations']};assert len(scheduled)==expected and {p.stem for p in (out/'completed').glob('*.json')}==scheduled
 if stage_name=='capture_v1':
  docs=read(ROOT/cfg['corpus'])['documents'];assert len(docs)==2048;create(stage/'exposure_consumed_v1.json',dict(model=cfg['model'],documents=docs,corpus_sha256=sha(ROOT/cfg['corpus']),completed_stage_sha256=sha(out/'stage_completed.json'),pre_run_snapshot_sha256=pre['snapshot_sha256'],scope='Fresh extension responses consumed; no future freshness claim.'))
 files=dict(pre['payload']['files'])
 for p in stage.rglob('*'):
  if p.is_file() and not p.name.startswith('completion_'):files[p.relative_to(ROOT).as_posix()]=sha(p)
 files[rp.relative_to(ROOT).as_posix()]=sha(rp);files[Path(__file__).resolve().relative_to(ROOT).as_posix()]=sha(Path(__file__))
 args=read(stage/'launch_receipt_v1.json')['args']
 for flag in ['--policy','--policy-snapshot','--policy-review']:
  if flag in args:
   p=ROOT/args[args.index(flag)+1];files[p.relative_to(ROOT).as_posix()]=sha(p)
   if flag=='--policy-snapshot':files.update(read(p)['payload']['files'])
 create(stage/'completion_summary_v1.json',dict(forward=expected,backward=0,LLM_loads=int(stage_name=='capture_v1'),runtime_returncode=0,new_calls_by_closure=0,independent_review='pending'))
 files[(stage/'completion_summary_v1.json').relative_to(ROOT).as_posix()]=sha(stage/'completion_summary_v1.json');create(stage/'completion_outputs_manifest_v1.json',dict(files=files));contract['id']+='_COMPLETION';contract['input_manifests'].append((stage/'completion_outputs_manifest_v1.json').relative_to(ROOT).as_posix());create(stage/'completion_contract_v1.json',contract)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['capture_v1','evaluation_v1'],required=True);p.add_argument('--runtime',required=True);a=p.parse_args();main(a.stage,a.runtime)
