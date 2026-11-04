"""Freeze fresh pairs and tokenizer; no pretrained model calls."""
from pathlib import Path
import itertools,json,re,numpy as np
from transformers import AutoTokenizer
from research_checks.gate import read,sha
from analysis.AB_five_experiments_20261103.A_L21.prepare_data import NOUNS,plural,strings
BASE=Path(__file__).parent;ROOT=BASE.parents[2];OLD=ROOT/'analysis/AB_five_experiments_20261103/A_L21'
def put(name,value):(BASE/name).write_text(json.dumps(value,indent=2)+'\n')
def main():
 paths={ROOT/p for p in read(OLD/'freshness_inputs.json')['files']}
 paths.add(OLD/'corpus.json')
 paths.update(p for folder in (ROOT/'analysis').iterdir() if folder.is_dir() and folder.name!='A_L21_learned_20261103' for pat in ['*corpus*.json','*dataset*.json','*documents*.json'] for p in folder.glob(pat))
 paths.update((ROOT/'analysis/AB_five_experiments_20261103').glob('*/corpus*.json'))
 paths={p for p in paths if p.is_file()};roots={w:w for w in NOUNS};roots.update({plural(w):w for w in NOUNS});used=set();prompts=set()
 for path in sorted(paths):
  for text in strings(read(path)):
   if len(text)>1000:continue
   prompts.add(text);present=sorted({roots[w] for w in re.findall('[a-z]+',text.lower()) if w in roots});used.update(itertools.combinations(present,2))
 oldpairs={tuple(sorted(p)) for p in read(OLD/'corpus.json')['pairs']};used.update(oldpairs)
 candidates=list(itertools.combinations(sorted(NOUNS),2));np.random.default_rng(3110361).shuffle(candidates);chosen=[x for x in candidates if x not in used][:64];assert len(chosen)==64
 tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True);docs=[]
 for b,pair in enumerate(chosen):
  for s,a in itertools.product([0,1],repeat=2):
   prompt=f"The {plural(pair[0]) if s else pair[0]} that the {plural(pair[1]) if a else pair[1]} can see"
   assert prompt not in prompts and not re.search(r'\b(coachs|fishermans|secretarys)\b',prompt)
   ids=[151643]+tok.encode(prompt,add_special_tokens=False)
   for word,answer in [(' is',374),(' are',525)]:assert tok.encode(prompt+word,add_special_tokens=False)==ids[1:]+[answer]
   docs.append(dict(id=f'LC_{b:03d}_{s}_{a}',block=b,split='TEST',frame=1,S=s,A=a,prompt=prompt,ids=ids,donor=f'LC_{b:03d}_{1-s}_{a}'))
 assert not set(chosen)&oldpairs
 put('freshness_inputs.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}))
 put('corpus.json',dict(documents=docs,pairs=chosen,seed=3110361,scope='Pair and exact-prompt disjointness against recorded exposure; individual words may recur; no pretraining novelty.'))
 put('data_checks.json',dict(ok=True,documents=256,blocks=64,conditions=1024,prior_campaign_pairs_excluded=len(oldpairs),history_files=len(paths),pair_overlap=0,prompt_overlap=0,trained_calls=0))
 print('Frozen64freshpairs/256documents; prior88campaign pairs excluded')
if __name__=='__main__':main()
