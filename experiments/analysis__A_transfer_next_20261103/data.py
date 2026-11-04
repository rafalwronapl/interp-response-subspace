"""Local tokenizer and hash-bound history inventory; no pretrained forward."""
from pathlib import Path
import itertools,json,re,subprocess,numpy as np
from transformers import AutoTokenizer
from research_checks.gate import read,sha
from analysis.AB_five_experiments_20261103.A_L21.prepare_data import NOUNS,plural,strings
BASE=Path(__file__).parent;ROOT=BASE.parents[1];OLD=ROOT/'analysis/AB_budget_readout_20261103'
def put(n,v):
 p=BASE/n;t=json.dumps(v,indent=2)+'\n'
 if p.exists() and p.read_text(encoding='utf-8')!=t:raise ValueError('Frozen preparation differs '+n)
 if not p.exists():p.write_text(t,encoding='utf-8')
def main():
 paths={ROOT/p for p in read(OLD/'freshness_inputs.json')['files']};paths.add(OLD/'corpus.json')
 listing=subprocess.run(['rg','--files','analysis','-g','*corpus*.json','-g','*dataset*.json','-g','*documents*.json'],cwd=ROOT,capture_output=True,text=True,check=True)
 paths.update(ROOT/p for p in listing.stdout.splitlines() if BASE not in (ROOT/p).parents and 'fixture_' not in p)
 paths={p for p in paths if p.is_file()};roots={w:w for w in NOUNS};roots.update({plural(w):w for w in NOUNS});used=set();prompts=set()
 for p in sorted(paths):
  for text in strings(read(p)):
   if len(text)>1000:continue
   prompts.add(text);present=sorted({roots[w] for w in re.findall('[a-z]+',text.lower()) if w in roots});used.update(itertools.combinations(present,2))
 choices=list(itertools.combinations(sorted(NOUNS),2));np.random.default_rng(3111037).shuffle(choices);chosen=[p for p in choices if p not in used][:6];assert len(chosen)==6
 tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True);docs=[];heads={word:tok.encode(word,add_special_tokens=False) for word in [' is',' are',' has',' have']};assert all(len(v)==1 for v in heads.values())
 for b,pair in enumerate(chosen):
  for s,a in itertools.product([0,1],repeat=2):
   prompt=f'The {plural(pair[0]) if s else pair[0]} that the {plural(pair[1]) if a else pair[1]} can see';assert prompt not in prompts
   ids=[151643]+tok.encode(prompt,add_special_tokens=False)
   docs.append(dict(id=f'ATF_{b:02d}_{s}_{a}',block=b,cohort=b//2,split='FIT_NEW',frame=1,S=s,A=a,prompt=prompt,ids=ids,donor=f'ATF_{b:02d}_{1-s}_{a}'))
 olddocs=read(OLD/'corpus.json')['documents']
 for d in docs+olddocs:
  for word,token in heads.items():assert tok.encode(d['prompt']+word,add_special_tokens=False)==d['ids'][1:]+token
 put('freshness_inputs.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}))
 put('corpus.json',dict(documents=docs,pairs=chosen,test_documents=olddocs,heads={k:v[0] for k,v in heads.items()},seed=3111037))
 # Read immutable receipts directly: no result-based search or model call.
 entries={};adopted={};missing=[]
 for d in olddocs:
  for suffix in ['natural']+[f'{br}_{dose}_{role}' for br in ['B','C'] for dose in [.5,1.] for role in ['prefix','target']]:
   operation=d['id']+'_'+suffix;bundle=OLD/'run/bundles'/f'{operation}.json'
   if not bundle.exists():missing.append(operation);continue
   b=read(bundle)['payload'];cache=OLD/'run/cache'/(b['request']+'.json');assert cache.exists()
   entries[operation]=dict(bundle=bundle.relative_to(ROOT).as_posix(),cache=cache.relative_to(ROOT).as_posix(),request=b['request']);adopted[bundle.relative_to(ROOT).as_posix()]=sha(bundle);adopted[cache.relative_to(ROOT).as_posix()]=sha(cache)
 assert not missing and len(entries)==2304
 put('adoption_index.json',dict(entries=entries));put('adoption_manifest.json',dict(files=adopted))
 put('data_checks.json',dict(ok=True,new_fit_blocks=6,new_fit_cohorts=3,labels_per_cohort=32,new_fit_prompts=24,old_test_blocks=64,old_test_prompts=256,adopted_records=len(entries),adopted_files=len(adopted),heads={k:v[0] for k,v in heads.items()},pretrained_calls=0,scope='Fresh FIT recorded unordered pairs; OLD TEST posthoc; new grammatical readout only'))
 print(dict(fit_blocks=6,adopted=len(entries),heads=heads,pretrained_calls=0))
if __name__=='__main__':main()
