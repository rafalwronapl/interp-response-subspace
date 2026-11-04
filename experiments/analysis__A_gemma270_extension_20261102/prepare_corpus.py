"""Literal model-independent fresh lexical corpus; no model calls."""
import json,re,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
MASTER=ROOT/'analysis/AB_confirmatory_execution_20261101/corpus/corpus_v1.json'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 data=read(MASTER);blocks=[b for b in data['blocks'] if b['study']=='A'];byblock={b['id']:b for b in blocks};seen=set(read(ROOT/'analysis/AB_confirmatory_execution_20261101/corpus/observed_words_v2.json'));sources={MASTER.relative_to(ROOT).as_posix():sha(MASTER)}
 # Actual immutable capture filenames, not roadmap/draft allocations.
 command=['rg','--files','analysis','-g','capture_A_*.json'];r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8');assert r.returncode in [0,1];paths=r.stdout.splitlines();captured_blocks=set()
 for name in paths:
  m=re.search(r'capture_(A_.+)_(?:F0|F1|T0|T2|T5|T6|T7)_s[01]_q[01]\.json$',name)
  if m:captured_blocks.add(m.group(1))
 exposed=set(seen)
 for b in blocks:
  if b['id'] in captured_blocks:
   for w in b['words']:exposed.update([w.lower(),w.lower()+'s'])
 manifest_paths=subprocess.run(['rg','--files','analysis','-g','*exposure_consumed*.json'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8');exposure_files=[]
 for name in manifest_paths.stdout.splitlines():
  p=ROOT/name
  try:v=read(p)
  except Exception:continue
  sources[p.relative_to(ROOT).as_posix()]=sha(p);exposure_files.append(p.relative_to(ROOT).as_posix())
  for d in v.get('documents',[]):exposed.update(re.findall('[a-z]+',d.get('text','').lower()))
 selected=[];rejected=[]
 for b in blocks:
  if not b['cohort'].startswith('r') or int(b['cohort'][1:])<6:continue
  if b['id'] in captured_blocks:rejected.append(dict(block=b['id'],reason='actual capture receipt exists'));continue
  forms=set()
  for rec in b['lexical_records']:
   for w in [rec['word'],*rec.get('all_noun_aliases',[])]:forms.update([w.lower(),w.lower()+'s'])
  overlap=sorted(forms&exposed)
  if overlap:rejected.append(dict(block=b['id'],reason='exposed lexical form or alias',overlap=overlap));continue
  selected.append(b)
  if len(selected)==256:break
 assert len(selected)==256,'Insufficient independent fresh blocks'
 words=[w for b in selected for w in b['words']];assert len(words)==len(set(words))==512
 docs=[];newblocks=[]
 for index,b in enumerate(selected):
  block=f'A_EXT_SHARED_TEST_{index:04d}';newblocks.append(dict(id=block,index=index,source_block=b['id'],source_cohort=b['cohort'],source_split=b['split'],words=b['words'],lexical_records=b['lexical_records']))
  for frame in ['T2','T5']:
   for side in [0,1]:
    for q in [0,1]:
     docs.append(dict(id=f'{block}_{frame}_s{side}_q{q}',donor=f'{block}_{frame}_s{1-side}_q{q}',block=block,study='A_EXTENSION',split='FRESH_TEST',index=index,frame=frame,side=side,q=q,text=data['frames']['A'][frame].format(s=b['words'][0]+('s' if side else ''),a=b['words'][1]+('s' if q else '')),source_block=b['id']))
 assert len(docs)==2048 and len({d['id'] for d in docs})==2048;lookup={d['id']:d for d in docs};assert all(lookup[d['donor']]['donor']==d['id'] for d in docs)
 corpus=dict(status='FROZEN_BEFORE_NEW_RESPONSES',blocks=newblocks,documents=docs,selection='First256 original reserved A lexical blocks from cohort>=r6 passing actual exposure and canonical/alias-form exclusion; no model outcomes inspected',master_sha256=sha(MASTER),frames={k:data['frames']['A'][k] for k in ['T2','T5']})
 with (OUT/'corpus.json').open('x',encoding='utf-8') as f:json.dump(corpus,f,indent=2)
 audit=dict(master_sha256=sha(MASTER),corpus_sha256=sha(OUT/'corpus.json'),fresh_blocks=256,fresh_headwords=512,documents=2048,capture_filename_command=command,capture_paths_checked=len(paths),previously_captured_master_blocks=len(captured_blocks),selected_source_capture_overlap=sorted({b['id'] for b in selected}&captured_blocks),selected_words=words,rejected_candidates=rejected,exposure_manifests=exposure_files,source_hashes=sources,historical_word_screen='Original qualified pools exclude historical lemmas/plurals/aliases; current actual exposures checked above',model_specific_caveat='Global reserved-word exposure audit is stronger than target-only filename absence, but cannot establish absence from pretraining; data is machine-defined WordNet count-noun population')
 (OUT/'freshness_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(dict(corpus_sha256=audit['corpus_sha256'],blocks=256,headwords=512,docs=2048,first_source=selected[0]['id'],last_source=selected[-1]['id'],rejected=len(rejected),capture_paths_checked=len(paths))))
if __name__=='__main__':main()
