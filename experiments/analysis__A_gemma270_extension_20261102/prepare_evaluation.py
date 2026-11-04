"""Bind accepted fresh captures to literal fixed-fit head packet."""
import argparse,runpy
from pathlib import Path
from research_checks.gate import digest
from analysis.C2_fresh_fit_holdout_20261102.worker import read,sha,receipt,put
from analysis.A_gemma270_extension_20261102.evaluate import arms
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent
PIN='c6d59249300d09c75008e298be87535e3365f6932bd5bbaa0b903d1f18b9c831'
def build(snapshot,review):
 cap=BASE/'capture_v1';s=read(ROOT/snapshot);r=read(ROOT/review);p=s['payload'];assert (ROOT/snapshot).parent.resolve()==cap.resolve();assert digest(p)==s['snapshot_sha256'] and p['phase']=='completion'
 assert r['snapshot_sha256']==s['snapshot_sha256'] and r['independent'] is True and r['verdict']=='approved' and r['reviewer_id']!=p['primary_agent'];assert set(r['topics_checked'])=={'design','implementation','data','cost','claims'}
 assert all(f['status']!='open' and (f['severity'] not in ['high','blocker'] or f['status']=='resolved') for f in r['findings'])
 pre=read(cap/'pre_run_snapshot_v1.json');assert digest(pre['payload'])==pre['snapshot_sha256']==PIN;manifest={};members=p['files'];cfg=read(cap/'config.json')
 def bind(file):
  file=Path(file);n=file.relative_to(ROOT).as_posix();h=sha(file);assert members[n]==h,n
  if n in pre['payload']['files']:assert pre['payload']['files'][n]==h,n
  manifest[n]=h
 for file in [cap/'config.json',ROOT/cfg['corpus'],ROOT/cfg['tokens'],ROOT/cfg['model_identity'],ROOT/cfg['schedule'],ROOT/cfg['asset_path'],ROOT/cfg['output']/'stage_completed.json',BASE/'analyze.py',BASE/'PROTOCOL.md']:bind(file)
 seal=receipt(ROOT/cfg['output']/'stage_completed.json');assert seal['forward']==2048 and seal['captures']==2048 and seal['backward']==0
 docs=read(ROOT/cfg['corpus'])['documents'];assert len(docs)==2048
 for d in docs:
  cp=ROOT/cfg['output']/'completed'/('capture_'+d['id']+'.json');bind(cp);bind(ROOT/cfg['output']/'cache'/(receipt(cp)['request']+'.json'))
 for pair in cfg['fixed_pairs']:
  for k in ['P','U']:bind(ROOT/pair[k]);assert sha(ROOT/pair[k])==pair[k+'_sha256']
 for n in [snapshot,review]:manifest[n]=sha(ROOT/n)
 stage=BASE/'evaluation_v1';assert not stage.exists();stage.mkdir();rel=stage.relative_to(ROOT).as_posix()
 evaluation=dict(capture_config='analysis/A_gemma270_extension_20261102/capture_v1/config.json',corpus=cfg['corpus'],fixed_pairs=cfg['fixed_pairs'],arms=arms(),output=rel+'/run',schedule=rel+'/operations.json')
 put(stage/'config.json',evaluation);put(stage/'operations.json',dict(operations=[dict(id=d['id']+'__'+a,kind='forward') for d in docs for a in arms()]))
 for n in ['config.json','operations.json']:manifest[rel+'/'+n]=sha(stage/n)
 put(stage/'input_manifest.json',dict(files=manifest));put(stage/'evidence.json',dict(accepted_capture=True,forward=20480,backward=0,no_new_FIT=True,LLM_loads=0))
 helper='analysis/AB_confirmatory_execution_20261101/close_short_pilot_imports_v1.py';seeds=[BASE.relative_to(ROOT).as_posix()+'/'+n for n in ['evaluate.py','prepare_evaluation.py','analyze.py','PROTOCOL.md','head_fixture_v1.py','head_fixture_v5.json']]+[helper];tracked=sorted(set(seeds)|set(runpy.run_path(str(ROOT/helper))['closure'](seeds)))
 put(stage/'contract.json',dict(schema_version=1,id='A_G270_EXTENSION_FIXED_HEADS_V1',kind='experiment',primary_agent='/root/sol_parallel_acceptance',purpose='All fresh256blocks T2/T5, three fixed matched fits; primaryT5 inside-P, outside secondary, no fitting or selection; frozen1999 block analysis only after sealed head outcomes.',tracked_files=tracked,tracked_trees=[],input_manifests=[rel+'/input_manifest.json'],gates=[dict(id=k,category=k,evidence=rel+'/evidence.json',pointer='/accepted_capture',op='eq',expected=True,reason='Actual independently accepted capture and bound current consumed files; fixed fits and qualified head arithmetic.') for k in ['hooks','data','pilot','accounting','restart']],circuits=[],structural_scope='Exact qualified lastL17 Gemma offset RMSNorm/two-head rows; no model loading, no refit.',entrypoint=dict(module='analysis.A_gemma270_extension_20261102.evaluate',args=['--config',rel+'/config.json']),schedule=rel+'/operations.json',budget=dict(forward=20480,backward=0)))
 print(rel)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--review',required=True);a=p.parse_args();build(a.snapshot,a.review)
