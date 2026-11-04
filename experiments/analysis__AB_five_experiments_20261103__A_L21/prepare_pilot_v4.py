"""No trained calls; freeze local source/model identity and 35 logical qualification calls."""
from pathlib import Path
import json,sys,runpy
from research_checks.gate import sha,read
BASE=Path(__file__).parent;ROOT=BASE.parents[2];REL=BASE.relative_to(ROOT).as_posix()
def put(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
 old=read(ROOT/'analysis/E041_execution/identity.json')['model_identity']['native']
 ds=read(ROOT/'analysis/E041_execution/dataset.json')['documents'];doc=next(d for d in ds if d['frame']==1)
 cfg=dict(model=old,document=doc)
 assert __import__('json').loads((BASE/'pilot_config.json').read_text())==cfg
 schedule=dict(operations=[dict(id=s,kind='forward') for s in (['natural','prefix','native']+[f'full_{i}' for i in range(16)]+[f'tail_{i}' for i in range(16)])])
 put(BASE/'pilot_schedule_v4.json',schedule)
 seeds=[REL+'/'+n for n in ['pilot_v4.py','adapter_v4.py','batch_fixture.py','fixtures_v3.py','pipeline_fixture_v4.py','math.py']]
 helper=ROOT/'analysis/AB_confirmatory_execution_20261101/close_short_pilot_imports_v1.py'
 tracked=sorted(set(seeds)|set(runpy.run_path(str(helper))['closure'](seeds)))
 paths=[BASE/'pilot_config.json',BASE/'pilot_schedule_v4.json',BASE/'fixtures_v3.json',BASE/'batch_fixture.json',BASE/'pipeline_fixture_v4.json',ROOT/'analysis/E041_execution/identity.json',ROOT/'analysis/E041_execution/dataset.json']+list((ROOT/'model').glob('*'))
 inputs={p.relative_to(ROOT).as_posix():sha(p) for p in paths if p.is_file()}
 assert inputs['model/model.safetensors']==old['weights_sha256']
 put(BASE/'pilot_inputs_v4.json',dict(files=inputs))
 contract=dict(schema_version=1,id='A_L21_20261103_pilot_v4',kind='experiment',primary_agent='/root/a_l21_five_campaign',purpose='Technical exact-tail qualification for literal E041 L21 replication; no outcome selection',tracked_files=tracked,tracked_trees=[],input_manifests=[REL+'/pilot_inputs_v4.json'],gates=[dict(id=k,category=k,evidence=REL+'/fixtures_v3.json',pointer='/'+k,op='eq',expected=True,reason='Random-weight fixture plus 35 logical charged pretrained qualification calls') for k in ['hooks','data','pilot','accounting','restart']],circuits=[],structural_scope='Native neuron19 and20 interventions, target L21, exact dynamic L22-27 suffix. Parity/sham/sensitivity before scale.',entrypoint=dict(module='analysis.AB_five_experiments_20261103.A_L21.pilot_v4',args=[]),schedule=REL+'/pilot_schedule_v4.json',budget=dict(forward=35,backward=0))
 contract['gates'].append(dict(id='batch_equivalence',category='hooks',evidence=REL+'/batch_fixture.json',pointer='/ok',op='eq',expected=True,reason='16 independent batch rows vs full and singleton on random native model'))
 put(BASE/'pilot_contract_v4.json',contract)
 print('Prepared35F,0B,1load; no trained calls during preparation')
if __name__=='__main__':main()
