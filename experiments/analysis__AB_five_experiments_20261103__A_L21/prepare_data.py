"""Freeze novel unordered lexical pairs from text-only historical inventories."""
from pathlib import Path
import json,re,itertools,numpy as np
from research_checks.gate import sha,read
from transformers import AutoTokenizer
BASE=Path(__file__).parent;ROOT=BASE.parents[2]
NOUNS='violinist pianist sailor carpenter baker florist geologist astronomer librarian cashier firefighter mechanic dancer photographer tailor barber jeweler watchmaker biologist physicist accountant secretary novelist editor surgeon farmer ranger bouncer bartender dentist receptionist beekeeper weaver shoemaker blacksmith architect artist astronaut auditor banker builder butcher chemist chef cleaner clerk coach collector composer conductor cook counselor courier decorator designer detective director diver doctor driver electrician engineer explorer fisherman gardener guard historian hunter inspector instructor interpreter inventor journalist judge lawyer lecturer locksmith magician manager mason miner musician officer painter pharmacist plumber poet porter professor programmer psychologist researcher roofer sculptor singer soldier teacher technician translator tutor undertaker veterinarian waiter welder writer'.split()
def strings(x):
 if isinstance(x,str):yield x
 elif isinstance(x,list):
  for z in x:yield from strings(z)
 elif isinstance(x,dict):
  for z in x.values():yield from strings(z)
def put(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def plural(word):return {'fisherman':'fishermen','secretary':'secretaries','coach':'coaches'}.get(word,word+'s')
def main():
 oldmanifest=ROOT/'analysis/AB_confirmatory_execution_20261101/corpus/historical_input_manifest_v2.json'
 paths={ROOT/p for p in read(oldmanifest)['files'] if Path(p).suffix=='.json' and any(w in Path(p).name.lower() for w in ['corpus','dataset','document','prompt','stimuli'])}
 paths.update([oldmanifest,ROOT/'analysis/AB_confirmatory_execution_20261101/corpus/corpus_v1.json',ROOT/'analysis/A_gemma270_extension_20261102/corpus.json',ROOT/'analysis/Qwen06_ABC_extension_20261102/capture_base_v2/corpus.json'])
 # Include newer top-level analysis corpus/dataset files, without scanning raw caches.
 for folder in (ROOT/'analysis').iterdir():
  if folder.is_dir() and folder.name!='AB_five_experiments_20261103':
   for pattern in ['*corpus*.json','*dataset*.json','*documents*.json']:
    paths.update(folder.glob(pattern))
 paths={p for p in paths if p.is_file()}
 put(BASE/'freshness_inputs.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}))
 roots={w:w for w in NOUNS};roots.update({plural(w):w for w in NOUNS})
 used=set();prompts=set()
 for path in sorted(paths):
  for text in strings(read(path)):
   if len(text)>1000:continue
   prompts.add(text);present=sorted({roots[w] for w in re.findall(r'[a-z]+',text.lower()) if w in roots})
   used.update(itertools.combinations(present,2))
 candidates=list(itertools.combinations(sorted(NOUNS),2));rng=np.random.default_rng(3110321);rng.shuffle(candidates)
 chosen=[p for p in candidates if p not in used][:88]
 assert len(chosen)==88
 tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True);docs=[]
 for b,pair in enumerate(chosen):
  split='TEST' if b>=24 else 'FIT'+str(b//8)
  for s,a in itertools.product([0,1],repeat=2):
   subject=plural(pair[0]) if s else pair[0];attractor=plural(pair[1]) if a else pair[1]
   prompt=f'The {subject} that the {attractor} can see'
   assert prompt not in prompts
   ids=[151643]+tok.encode(prompt,add_special_tokens=False)
   for word,answer in [(' is',374),(' are',525)]:assert tok.encode(prompt+word,add_special_tokens=False)==ids[1:]+[answer]
   docs.append(dict(id=f'L21_{b:03d}_{s}_{a}',block=b,split=split,frame=1,S=s,A=a,prompt=prompt,ids=ids,donor=f'L21_{b:03d}_{1-s}_{a}'))
 assert not any(re.search(r'\b(coachs|fishermans|secretarys)\b',d['prompt']) for d in docs)
 put(BASE/'corpus.json',dict(documents=docs,pairs=chosen,seed=3110321,scope='Fresh unordered noun pairs and exact prompts against bound textual inventory; individual nouns may recur; no pretraining novelty claim.'))
 put(BASE/'data_checks.json',dict(ok=True,documents=len(docs),test_blocks=64,fit_blocks=24,unique_pairs=88,historical_pair_overlap=0,historical_prompt_overlap=0,history_files=len(paths),trained_calls=0))
 print('Frozen352documents,64TEST+24FITblocks')
if __name__=='__main__':main()
