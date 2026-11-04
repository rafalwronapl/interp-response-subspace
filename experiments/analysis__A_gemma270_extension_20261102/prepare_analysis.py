"""Literal zero-model frozen A270 block analysis packet."""
import argparse,runpy
from pathlib import Path
from research_checks.gate import digest
from analysis.C2_fresh_fit_holdout_20261102.worker import read,sha,receipt,put
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent
def build(snapshot,review):
 s=read(ROOT/snapshot);r=read(ROOT/review);p=s['payload'];assert (ROOT/snapshot).parent.resolve()==(BASE/'evaluation_v1').resolve();assert digest(p)==s['snapshot_sha256'] and p['phase']=='completion'
 assert r['snapshot_sha256']==s['snapshot_sha256'] and r['independent'] is True and r['verdict']=='approved' and r['reviewer_id']!=p['primary_agent'];assert set(r['topics_checked'])=={'design','implementation','data','cost','claims'};assert all(f['status']!='open' for f in r['findings'])
 cfg=read(BASE/'evaluation_v1/config.json');out=ROOT/cfg['output'];seal=receipt(out/'stage_completed.json');assert seal['forward']==20480 and seal['backward']==0 and seal['LLM_loads']==0
 files={};pre=read(BASE/'capture_v1/pre_run_snapshot_v1.json')
 for file in [out/'result.json',out/'stage_completed.json',ROOT/cfg['corpus'],BASE/'analyze.py',BASE/'PROTOCOL.md',BASE/'evaluation_v1/config.json']:
  n=file.relative_to(ROOT).as_posix();h=sha(file);assert p['files'][n]==h
  if n in pre['payload']['files']:assert pre['payload']['files'][n]==h
  files[n]=h
 assert seal['essential_files']['result.json']==files[(out/'result.json').relative_to(ROOT).as_posix()]
 for n in [snapshot,review]:files[n]=sha(ROOT/n)
 stage=BASE/'analysis_v1';assert not stage.exists();stage.mkdir();rel=stage.relative_to(ROOT).as_posix();put(stage/'operations.json',dict(operations=[]));files[rel+'/operations.json']=sha(stage/'operations.json');put(stage/'input_manifest.json',dict(files=files));put(stage/'evidence.json',dict(accepted_heads=True,forward=0,backward=0,draws=1999,block_unit=256))
 helper='analysis/AB_confirmatory_execution_20261101/close_short_pilot_imports_v1.py';seeds=[BASE.relative_to(ROOT).as_posix()+'/'+n for n in ['analyze.py','prepare_analysis.py','PROTOCOL.md','analyzer_fixture_root_v1.json']]+[helper];tracked=sorted(set(seeds)|set(runpy.run_path(str(ROOT/helper))['closure'](seeds)))
 put(stage/'contract.json',dict(schema_version=1,id='A_G270_EXTENSION_BLOCK_ANALYSIS_V1',kind='experiment',primary_agent='/root/sol_parallel_acceptance',purpose='Frozen primaryT5 inside-P; block256, fixed3fits averagewithinblock,1999draws nominalCI; descriptiveT2/outside, no p/family/coveragepromise.',tracked_files=tracked,tracked_trees=[],input_manifests=[rel+'/input_manifest.json'],gates=[dict(id=k,category=k,evidence=rel+'/evidence.json',pointer='/accepted_heads',op='eq',expected=True,reason='Independent accepted exact head closure and prospectively fixed analyzer/corpus.') for k in ['hooks','data','pilot','accounting','restart']],circuits=[],structural_scope='Pure saved-margin scalar analysis, no model/native/readout/refit calls.',entrypoint=dict(module='analysis.A_gemma270_extension_20261102.analyze',args=['--result',cfg['output']+'/result.json','--corpus',cfg['corpus'],'--out',rel+'/result.json']),schedule=rel+'/operations.json',budget=dict(forward=0,backward=0)))
 print(rel)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--review',required=True);a=p.parse_args();build(a.snapshot,a.review)
