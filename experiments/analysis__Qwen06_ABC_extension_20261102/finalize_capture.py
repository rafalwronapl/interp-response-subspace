"""Finalize actual pre-run evidence before first snapshot; no model calls."""
import json,runpy,tempfile
from pathlib import Path
from analysis.C2_fresh_fit_holdout_20261102.worker import read,sha
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;stage=BASE/'capture_base_v2'
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
 cfg=read(stage/'config.json');cfg['physical_output']=str((ROOT/cfg['output']).resolve());assert Path(cfg['physical_output']).is_relative_to(ROOT);write(stage/'config.json',cfg)
 from ab_confirmatory_20261101.runtime import Runtime
 with tempfile.TemporaryDirectory() as tmp:
  out=Path(tmp);calls=[];ident=dict(model='synthetic',code='fixture');schedule=dict(operations=[dict(id='one',kind='forward'),dict(id='two',kind='forward')])
  rt=Runtime(out,ident,schedule)
  def call():calls.append(1);return {'v':3.}
  a=rt.compute('one',dict(document='d'),dict(mode='natural'),dict(answer_ids=[374,525]),call)
  paths=list(out.rglob('*.json'));before={str(p):sha(p) for p in paths}
  rt=Runtime(out,ident,schedule);b=rt.compute('one',dict(document='d'),dict(mode='natural'),dict(answer_ids=[374,525]),call)
  assert a==b and len(calls)==1 and all(sha(Path(p))==h for p,h in before.items())
 fixture=dict(status='PASS',model_loads=0,forward=0,tests=['Runtime validcache avoids repeat computation','JSONreceipts byte-identical onresume','OriginalE041checkpointidentity compared actualbytes in prepare'],source_sha256=sha(BASE/'capture.py'))
 write(stage/'fixtures.json',fixture)
 manifest=read(stage/'input_manifest.json')['files'];ident=read(stage/'model_identity.json');manifest[str((stage/'config.json').relative_to(ROOT)).replace('\\','/')]=sha(stage/'config.json')
 for name,h in ident['files'].items():manifest[Path(name).relative_to(ROOT).as_posix()]=h
 for p in [ROOT/'analysis/E041_execution/identity.json',stage/'fixtures.json',ROOT/'analysis/A_gemma270_extension_20261102/freshness_audit.json',ROOT/'analysis/A_gemma270_extension_20261102/prepare_corpus.py']:
  manifest[p.relative_to(ROOT).as_posix()]=sha(p)
 audit=read(ROOT/'analysis/A_gemma270_extension_20261102/freshness_audit.json')
 for name,h in audit['source_hashes'].items():assert sha(ROOT/name)==h;manifest[name]=h
 write(stage/'input_manifest.json',dict(files=manifest))
 c=read(stage/'contract.json');c['tracked_files']=sorted(set(c['tracked_files'])|{str((BASE/'finalize_capture.py').relative_to(ROOT)).replace('\\','/')});write(stage/'contract.json',c)
 print('Finalized capture identity, exposure audit, cache-resume fixture;0modelcalls')
if __name__=='__main__':main()
