"""Canonical contract and transitive code/data/model identity before native use."""
from pathlib import Path
import json,runpy
from datetime import datetime,timezone
from research_checks.gate import read,sha
from .worker import schedule
BASE=Path(__file__).parent;ROOT=BASE.parents[1];REL=BASE.relative_to(ROOT).as_posix()
def put(n,v):(BASE/n).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
 fixture=read(BASE/'fixture.json');assert fixture['worker_sha256']==sha(BASE/'worker.py') and fixture['models_sha256']==sha(BASE/'models.py')
 corpus=read(BASE/'corpus.json');ops=schedule(corpus['documents'],corpus['test_documents']);put('operations.json',ops);count=len(ops['operations']);assert count==30444
 old=ROOT/'analysis/AB_budget_readout_20261103';cfg=read(old/'config.json');put('config.json',dict(model=cfg['model'],checkpoint=cfg['checkpoint'],prior=cfg['prior'],maximum_forward=count,physical_maximum=4794,new_fit_labels=192,new_test_blocks=64,planned_model_loads=1))
 paths={ROOT/p for p in read(old/'input_manifest.json')['files']};paths.update(ROOT/p for p in read(BASE/'freshness_inputs.json')['files'])
 paths.update(BASE/n for n in ['PROTOCOL.txt','corpus.json','freshness_inputs.json','data_checks.json','config.json','operations.json','fixture.json'])
 paths.update(ROOT/p for p in ['analysis/uwagi_zewnetrzne_20261104/review_method.md','analysis/uwagi_zewnetrzne_20261104/source_manifest.json','analysis/A13_review_checks_20261104/findings.txt','analysis/A_prior_matched_20261104/review_registry_before.txt'])
 seeds=[REL+'/'+p.name for p in BASE.glob('*.py')];helper=ROOT/'analysis/AB_confirmatory_execution_20261101/close_short_pilot_imports_v1.py';tracked=sorted(set(seeds)|set(runpy.run_path(str(helper))['closure'](seeds)));paths.update(ROOT/p for p in tracked)
 freeze=BASE/'freeze.json'
 if not freeze.exists():put('freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),status='prepared_unrun',scope='6freshFIT32/commonfreshTEST64;matchedprior4arms'))
 paths.add(freeze);put('input_manifest.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}))
 gates=[dict(id=k,category=k,evidence=REL+'/fixture.json',pointer='/'+k,op='eq',expected=True,reason='Actualworker synthetic globalforecastfreeze/crash/restart/seals/exactaccounting and nativeadaptertoy checks') for k in ['hooks','data','pilot','accounting','restart']]
 gates.append(dict(id='tokenization_freshness',category='data',evidence=REL+'/data_checks.json',pointer='/ok',op='eq',expected=True,reason='Localactualtokenization,76newpairs excluded cumulativeexposure'))
 put('contract.json',dict(schema_version=1,id='A_PRIOR_MATCHED_6FIT32_FRESHTEST64_20261104',kind='experiment',primary_agent='/root/a_prior_experiment',purpose='Prospective missing full-vector residualridge comparator; six freshFIT32, fresh64blockTEST, historicalregimeconditional',tracked_files=tracked,tracked_trees=[],input_manifests=[REL+'/input_manifest.json'],gates=gates,circuits=[],structural_scope='Exact historical native prefix/MLP21patch and L22-27suffix, no reducedcircuit; actualFIT full/suffix parity16and11batch and fixed nonzero sensitivity beforeTEST',entrypoint=dict(module='analysis.A_prior_matched_20261104.worker',args=[]),schedule=REL+'/operations.json',budget=dict(forward=count,backward=0)))
 print(dict(logical=count,physical=4794,files=len(paths),ready_for_independent_review=True))
if __name__=='__main__':main()
