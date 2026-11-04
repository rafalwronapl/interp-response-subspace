"""Prepare separately gated compact head stage after accepted QBase capture."""
import json,runpy
from pathlib import Path
from analysis.C2_fresh_fit_holdout_20261102.worker import ROOT,read,sha,put,receipt
from analysis.Qwen06_ABC_extension_20261102.evaluate import expected_arms
from analysis.Qwen06_ABC_extension_20261102.accepted import accepted as accept
BASE=Path(__file__).parent
def main(snapshot,review):
 accepted=accept(snapshot,review);assert accepted['snapshot_sha256']=='eccfa58d9e7ca5514358ed7f660b472449c0cdd52b33d34a35acb97d9f382906'
 cap=BASE/'capture_base_v2';cfg=read(cap/'config.json');done=receipt(ROOT/cfg['output']/'stage_completed.json');assert done['forward']==2844 and done['captures']==2816
 stage=BASE/'evaluation_v1';stage.mkdir();(stage/'run').mkdir();rel=stage.relative_to(ROOT).as_posix();docs=read(ROOT/cfg['corpus'])['documents'];hold=[d for d in docs if d['role']=='HOLDOUT'];arms=expected_arms();assert len(arms)==50 and len(hold)==2048
 ops=[dict(id=d['id']+'__'+arm,kind='forward') for d in hold for arm in arms];assert len(ops)==102400
 evaluation=dict(model='qwen06_base',capture_stage=cap.relative_to(ROOT).as_posix(),corpus=cfg['corpus'],assets=cfg['assets'],asset_path=cfg['asset_path'],arms=arms,output=rel+'/run',physical_output=str((stage/'run').resolve()),schedule=rel+'/operations.json',recipe=dict(MAP=128,U=128,rank=8,lambda_value=.01,residual_n=128,pooled_direct_n=256),primary='T5 average3matchedpairs,256blocks;T2and9crossessecondary')
 put(stage/'config.json',evaluation);put(stage/'operations.json',dict(operations=ops))
 manifest={};members=accepted['payload']['files']
 consumed=[cap/'config.json',ROOT/cfg['corpus'],ROOT/cfg['tokens'],ROOT/cfg['model_identity'],ROOT/cfg['asset_path']]+list((ROOT/cfg['output']).rglob('*.json'))
 for p in consumed:
  name=p.relative_to(ROOT).as_posix();h=sha(p);assert members[name]==h,name;manifest[name]=h
 for p in [ROOT/snapshot,ROOT/review,stage/'config.json',stage/'operations.json',BASE/'evaluation_fixture_v1.json',BASE/'evaluation_fixtures.py',BASE/'bundle_fixture_v1.json',BASE/'bundle_fixtures.py',BASE/'scoring_fixture_v1.json',BASE/'scoring_fixtures.py',BASE/'prepare_evaluation_source.py',BASE/'compact_evaluation_source.py',BASE/'ANALYSIS_ALIGNMENT_BEFORE_HEAD.txt']:manifest[p.relative_to(ROOT).as_posix()]=sha(p)
 put(stage/'input_manifest.json',dict(files=manifest));put(stage/'evidence.json',dict(accepted_capture=True,forward=102400,backward=0,LLM_loads=0,unique_gradient_points=6144,analytic_gradient_evaluations=18432,retained_fp64_vector_hashes=True,primary3matchedfits=True,all9crosses=True,points=2048,arms=50))
 helper='analysis/AB_confirmatory_execution_20261101/close_short_pilot_imports_v1.py';seeds=[BASE.relative_to(ROOT).as_posix()+'/'+n for n in ['evaluate.py','score.py','evaluation_worker.py','prepare_evaluation.py','PROTOCOL.txt']]+[helper];tracked=sorted(set(seeds)|set(runpy.run_path(str(ROOT/helper))['closure'](seeds)))
 put(stage/'contract.json',dict(schema_version=1,id='Qwen06_ABC_evaluation_v1',kind='experiment',primary_agent='/root/sol_qwen06_parallel',purpose='FixedA/C50armheadonlyevaluation,compactexactFP64vectorhashes,nodeletionofarms;forecastsealedbeforeoracleDthenfiniteoutcomes;descriptivefixedfitblockintervals.',tracked_files=tracked,tracked_trees=[],input_manifests=[rel+'/input_manifest.json'],gates=[dict(id=k,category=k,evidence=rel+'/evidence.json',pointer='/accepted_capture',op='eq',expected=True,reason='Acceptedactualcapture,allconsumedcacheinputsbound,exactnoDfeatureboundaryandvectorhashregeneration reviewed before measurement.') for k in ['hooks','data','pilot','accounting','restart']],circuits=[],structural_scope='QualifiednativeQBasefinalL27R/x/y,actualfinalnormand2headrows;noLLMreload.',entrypoint=dict(module='analysis.Qwen06_ABC_extension_20261102.evaluation_worker',args=['--config',rel+'/config.json']),schedule=rel+'/operations.json',budget=dict(forward=102400,backward=0)))
 print(rel)
if __name__=='__main__':
 import sys
 main(*sys.argv[1:])
