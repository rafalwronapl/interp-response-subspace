"""Freeze actual old-cache adoption, source closure and canonical launch contract."""
from pathlib import Path
import json,runpy
from datetime import datetime,timezone
from research_checks.gate import read,sha
from .worker import schedule
BASE=Path(__file__).parent;ROOT=BASE.parents[1];OLD=ROOT/'analysis/AB_budget_readout_20261103';REL=BASE.relative_to(ROOT).as_posix()
def put(n,v):(BASE/n).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
 assert read(BASE/'fixture.json')['worker_sha256']==sha(BASE/'worker.py')
 assert read(OLD/'review/completion_review.json')['verdict']=='approved'
 corpus=read(BASE/'corpus.json');ops=schedule(corpus['documents'],corpus['test_documents']);put('operations.json',ops);count=len(ops['operations']);assert count==12516
 cfg=read(OLD/'config.json');put('config.json',dict(model=cfg['model'],checkpoint=cfg['checkpoint'],prior=cfg['prior'],maximum_forward=count,physical_maximum=1248,new_fit_labels=96,old_test_blocks=64))
 # Preserve old complete model/library/prior provenance in addition to actual raw adoption.
 paths={ROOT/p for p in read(OLD/'input_manifest.json')['files']}
 paths.update(ROOT/p for p in read(BASE/'adoption_manifest.json')['files'])
 paths.update(ROOT/p for p in read(BASE/'freshness_inputs.json')['files'])
 paths.update(BASE/n for n in ['PROTOCOL.txt','corpus.json','freshness_inputs.json','data_checks.json','adoption_index.json','adoption_manifest.json','config.json','operations.json','fixture.json'])
 paths.update(OLD/n for n in ['run/completed.json','run/ledger.jsonl','review/completion_review.json','completion_snapshot.json','input_manifest.json','library_identity.json'])
 seeds=[REL+'/'+p.name for p in BASE.glob('*.py')];helper=ROOT/'analysis/AB_confirmatory_execution_20261101/close_short_pilot_imports_v1.py';tracked=sorted(set(seeds)|set(runpy.run_path(str(helper))['closure'](seeds)));paths.update(ROOT/p for p in tracked)
 # A timestamped immutable freeze receipt is part of the snapshot, not an outcome.
 freeze=BASE/'freeze.json'
 if not freeze.exists():put('freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),status='prepared_unrun',scope='oldTEST newreadout;96freshFITvectorlabels'))
 paths.add(freeze);put('input_manifest.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}))
 gates=[dict(id=k,category=k,evidence=REL+'/fixture.json',pointer='/'+k,op='eq',expected=True,reason='Actualworker synthetic model path,globalfreeze,crashresume,seals,12batch accounting') for k in ['hooks','data','pilot','accounting','restart']]
 gates.append(dict(id='tokenization_adoption',category='data',evidence=REL+'/data_checks.json',pointer='/ok',op='eq',expected=True,reason='Allactualprompts single-token is/are/has/have;2304oldrecords;6newFITblocks'))
 put('contract.json',dict(schema_version=1,id='A_NEW_READOUT_3FIT32_20261103',kind='experiment',primary_agent='/root',purpose='New grammatical has/have readout and3freshFIT32 stability on old64blockTEST;posthoc',tracked_files=tracked,tracked_trees=[],input_manifests=[REL+'/input_manifest.json'],gates=gates,circuits=[],structural_scope='Qualified exact historical native prefix/MLP21 patch/L22-27suffix;newfourhead projection;newFIT full/suffix parity and fixednonzerohead sensitivity before oldTEST measurements',entrypoint=dict(module='analysis.A_transfer_next_20261103.worker',args=[]),schedule=REL+'/operations.json',budget=dict(forward=count,backward=0)))
 print(dict(logical=count,physical=1248,files=len(paths),ready_for_independent_review=True))
if __name__=='__main__':main()
