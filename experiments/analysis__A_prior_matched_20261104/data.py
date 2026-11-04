"""Freeze new lexical pairs against cumulative recorded history; tokenizer only."""
from pathlib import Path
import itertools,json,re,subprocess,numpy as np
from transformers import AutoTokenizer
from research_checks.gate import read,sha
from analysis.AB_five_experiments_20261103.A_L21.prepare_data import NOUNS,plural,strings
BASE=Path(__file__).parent;ROOT=BASE.parents[1]
def put(name,value):
 p=BASE/name;t=json.dumps(value,indent=2)+'\n'
 if p.exists() and p.read_text(encoding='utf-8')!=t:raise ValueError('Frozen preparation differs '+name)
 if not p.exists():p.write_text(t,encoding='utf-8')
def main():
 paths={ROOT/p for p in read(ROOT/'analysis/A_transfer_next_20261103/freshness_inputs.json')['files']}
 listing=subprocess.run(['rg','--files','analysis','runs','data','-g','*corpus*.json','-g','*dataset*.json','-g','*documents*.json'],cwd=ROOT,capture_output=True,text=True,check=True)
 paths.update(ROOT/p for p in listing.stdout.splitlines() if BASE not in (ROOT/p).parents)
 paths={p for p in paths if p.is_file()};roots={w:w for w in NOUNS};roots.update({plural(w):w for w in NOUNS});used=set();prompts=set()
 for p in sorted(paths):
  for text in strings(read(p)):
   if len(text)>1000:continue
   prompts.add(text);present=sorted({roots[w] for w in re.findall('[a-z]+',text.lower()) if w in roots});used.update(itertools.combinations(present,2))
 choices=list(itertools.combinations(sorted(NOUNS),2));np.random.default_rng(4111041).shuffle(choices);chosen=[p for p in choices if p not in used][:76];assert len(chosen)==76
 tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True);docs=[];heads={word:tok.encode(word,add_special_tokens=False) for word in [' is',' are',' has',' have']};assert all(len(v)==1 for v in heads.values())
 for b,pair in enumerate(chosen):
  prefix='APF' if b<12 else 'APT';block=b if b<12 else b-12
  for s,a in itertools.product([0,1],repeat=2):
   prompt=f'The {plural(pair[0]) if s else pair[0]} that the {plural(pair[1]) if a else pair[1]} can see';assert prompt not in prompts
   ids=[151643]+tok.encode(prompt,add_special_tokens=False)
   for word,token in heads.items():assert tok.encode(prompt+word,add_special_tokens=False)==ids[1:]+token
   docs.append(dict(id=f'{prefix}_{block:02d}_{s}_{a}',block=block,cohort=b//2 if b<12 else None,split='FIT_NEW' if b<12 else 'TEST_NEW',frame=1,S=s,A=a,prompt=prompt,ids=ids,donor=f'{prefix}_{block:02d}_{1-s}_{a}'))
 put('freshness_inputs.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}))
 put('corpus.json',dict(documents=[d for d in docs if d['split']=='FIT_NEW'],test_documents=[d for d in docs if d['split']=='TEST_NEW'],pairs=chosen,heads={k:v[0] for k,v in heads.items()},seed=4111041))
 put('data_checks.json',dict(ok=True,new_fit_blocks=12,new_fit_cohorts=6,labels_per_cohort=32,new_fit_prompts=48,new_test_blocks=64,new_test_prompts=256,history_files=len(paths),historical_pairs=len(used),fresh_pairs=76,pretrained_calls=0,excludes_previous_3FIT_and_oldTEST=True,scope='Unordered pair exclusion across cumulative inventoried corpora; no worldwide dataset guarantee'))
 print(read(BASE/'data_checks.json'))
if __name__=='__main__':main()
