"""Actual B packet assembled after accepted aligned A/C evidence."""
import sys,runpy
from pathlib import Path
from analysis.Qwen06_ABC_extension_20261102.accepted import accepted as accept
from analysis.C2_fresh_fit_holdout_20261102.worker import ROOT,read,sha,put,receipt
from analysis.Qwen06_ABC_extension_20261102.b_worker import DOSES,holdout_operations
BASE=Path(__file__).parent
def main(snapshot,review):
 accepted=accept(snapshot,review);ev=BASE/'evaluation_v2';evcfg=read(ev/'config.json');assert accepted['payload']['contract']==(ev/'completion_contract_v1.json').relative_to(ROOT).as_posix();result=receipt(ROOT/evcfg['output']/'result.json');assert len(result['rows'])==102400
 cap=BASE/'capture_base_v2';cfg=read(cap/'config.json');docs=read(ROOT/cfg['corpus'])['documents'];fit=[d for d in docs if d['role']=='FIT' and d['cohort']=='r1' and d['split']=='FIT_MAP'];hold=[d for d in docs if d['role']=='HOLDOUT'];stage=BASE/'B_v1';stage.mkdir();(stage/'run').mkdir();rel=stage.relative_to(ROOT).as_posix()
 operations=[dict(id='control_'+doc+'_'+k,kind='forward') for doc in cfg['controls'] for k in ['branch','head','branch_plus','branch_minus','branch_center64','head_plus','head_minus']]+[dict(id='fit_'+d['id']+'_c'+str(c),kind='forward') for d in fit for c in DOSES if c!=1.]+[dict(id=op,kind='forward') for d in hold for op in holdout_operations(d['id'])];assert len(operations)==96924
 bcfg=dict(capture_config=(cap/'config.json').relative_to(ROOT).as_posix(),corpus=cfg['corpus'],aligned_evaluation_result=(ROOT/evcfg['output']/'result.json').relative_to(ROOT).as_posix(),unit_dose_fit=(ROOT/evcfg['output']/'fits/P_r1.json').relative_to(ROOT).as_posix(),output=rel+'/run',schedule=rel+'/operations.json',doses=DOSES,scope='Last27publicCLTcomponent,onefullinputoperator,conceptualBextension')
 put(stage/'config.json',bcfg);put(stage/'operations.json',dict(operations=operations));files={};members=accepted['payload']['files']
 consumed=[cap/'config.json',ev/'config.json',ROOT/cfg['corpus'],ROOT/cfg['tokens'],ROOT/cfg['model_identity'],ROOT/cfg['asset_path'],ROOT/bcfg['aligned_evaluation_result'],ROOT/bcfg['unit_dose_fit']]+list((ROOT/cfg['output']).rglob('*.json'))
 for p in consumed:
  name=p.relative_to(ROOT).as_posix();assert name in members;files[name]=members[name]
 upstream=ROOT/'runs/E017/upstream.json';meta=read(upstream);assert meta['sha']=='5301dca1909eccac1ecedd3d58da60097056db8f'
 native_identity=read(ROOT/cfg['model_identity']);native_file='model/model.safetensors';expected_native=native_identity['files'][str(ROOT/native_file)];files[native_file]=expected_native
 for name in ['clt_cache/e017/W_enc_27.safetensors','clt_cache/e033/W_dec_27.safetensors']:
  p=ROOT/name;entry=next(x for x in meta['siblings'] if x['rfilename']==p.name);assert p.stat().st_size==entry['size'] and sha(p)==entry['lfs']['sha256'];files[name]=entry['lfs']['sha256']
 for p in [ROOT/snapshot,ROOT/review,upstream,stage/'config.json',stage/'operations.json',BASE/'B_PROTOCOL.txt',BASE/'b_fixture_v2.json',BASE/'bundle_fixture_v1.json',BASE/'bundle_fixtures.py',BASE/'b_scoring_fixture_v2.json',BASE/'b_scoring_fixtures_v2.py']:files[p.relative_to(ROOT).as_posix()]=sha(p)
 put(stage/'input_manifest.json',dict(files=files));put(stage/'evidence.json',dict(prepared=True,forward=96924,backward=0,full_model_loads=0,new_native_branch=10896,new_native_heads=69644,CLT_endpoint_delta_predictions=12288,CLT_base_encodings=2048,FROZEN_direction_predictions=2048,analytic_JVP=2052,adopted_native_and_ridge_unit_heads=4096,heldout_B_outcomes_used_for_fit=False,one_full_input_operator=True))
 helper='analysis/AB_confirmatory_execution_20261101/close_short_pilot_imports_v1.py';seeds=[BASE.relative_to(ROOT).as_posix()+'/'+n for n in ['b_worker.py','b_math.py','b_score.py','b_driver.py','b_fixtures_v2.py','prepare_b.py','B_PROTOCOL.txt']]+[helper];tracked=sorted(set(seeds)|set(runpy.run_path(str(ROOT/helper))['closure'](seeds)))
 put(stage/'contract.json',dict(schema_version=1,id='Qwen06_B_last27_component_v1',kind='experiment',primary_agent='/root/sol_qwen06_parallel',purpose='CompleteboundedpublicCLTinput-dosefidelitycurve,7methods6dosesT2T5+mixture,exactnativeSwiGLU+headTaylor,FIT-onlyperdosebaselines,nooutcometuning.',tracked_files=tracked,tracked_trees=[],input_manifests=[rel+'/input_manifest.json'],gates=[dict(id=k,category=k,evidence=rel+'/evidence.json',pointer='/prepared',op='eq',expected=True,reason='FrozenqualifiedQBasecaptures,exactpublicCLTsource27target27headers/identity,chargedbranch/composedJVPtechnicalcontrols beforeTEST,allmethods/dosesretained.') for k in ['hooks','data','pilot','accounting','restart']],circuits=[],structural_scope='NativefinalL27SwiGLUweights plusactualfinalnorm/twoheadrows; no earlier-layerhead-only shortcut, no fullmodelreload.',entrypoint=dict(module='analysis.Qwen06_ABC_extension_20261102.b_driver',args=['--config',rel+'/config.json']),schedule=rel+'/operations.json',budget=dict(forward=96924,backward=0)))
 print(rel)
if __name__=='__main__':main(*sys.argv[1:])
