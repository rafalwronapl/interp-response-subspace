"""Freeze a genuinely new recorded-pair cohort using local tokenizer only."""
from pathlib import Path
import itertools,json,re,subprocess,numpy as np
from transformers import AutoTokenizer
from research_checks.gate import read,sha
from analysis.AB_five_experiments_20261103.A_L21.prepare_data import NOUNS,plural,strings
BASE=Path(__file__).parent;ROOT=BASE.parents[1];PREVIOUS=ROOT/'analysis/A_L21_learned_20261103/stage2'
def put(name,value):
    p=BASE/name
    text=json.dumps(value,indent=2)+'\n'
    if p.exists() and p.read_text()!=text:raise ValueError('Frozen preparation differs '+str(p))
    if not p.exists():p.write_text(text,encoding='utf-8')
def main():
    paths={ROOT/p for p in read(PREVIOUS/'freshness_inputs.json')['files']}
    paths.update([PREVIOUS/'corpus.json',PREVIOUS/'freshness_inputs.json'])
    listing=subprocess.run(['rg','--files','analysis','-g','*corpus*.json','-g','*dataset*.json','-g','*documents*.json'],cwd=ROOT,capture_output=True,text=True,check=True)
    paths.update(p for p in (ROOT/name for name in listing.stdout.splitlines()) if BASE not in p.parents and not any('fixture_' in a for a in p.parts))
    paths={p for p in paths if p.is_file()};roots={w:w for w in NOUNS};roots.update({plural(w):w for w in NOUNS});used=set();prompts=set()
    for path in sorted(paths):
        for text in strings(read(path)):
            if len(text)>1000:continue
            prompts.add(text);present=sorted({roots[w] for w in re.findall('[a-z]+',text.lower()) if w in roots});used.update(itertools.combinations(present,2))
    candidates=list(itertools.combinations(sorted(NOUNS),2));np.random.default_rng(3111031).shuffle(candidates)
    chosen=[x for x in candidates if x not in used][:64];assert len(chosen)==64
    tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True);docs=[]
    for b,pair in enumerate(chosen):
        for s,a in itertools.product([0,1],repeat=2):
            prompt=f"The {plural(pair[0]) if s else pair[0]} that the {plural(pair[1]) if a else pair[1]} can see"
            assert prompt not in prompts and not re.search(r'\b(coachs|fishermans|secretarys)\b',prompt)
            ids=[151643]+tok.encode(prompt,add_special_tokens=False)
            for word,answer in [(' is',374),(' are',525)]:assert tok.encode(prompt+word,add_special_tokens=False)==ids[1:]+[answer]
            docs.append(dict(id=f'BR_{b:03d}_{s}_{a}',block=b,split='TEST',frame=1,S=s,A=a,prompt=prompt,ids=ids,donor=f'BR_{b:03d}_{1-s}_{a}'))
    assert not set(chosen)&used
    put('freshness_inputs.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}))
    put('corpus.json',dict(documents=docs,pairs=chosen,seed=3111031,scope='Recorded historical unordered-pair/prompt disjointness; individual nouns and prompt family recur'))
    put('data_checks.json',dict(ok=True,documents=256,blocks=64,conditions=1024,history_files=len(paths),prior_observed_pairs=len(used),pair_overlap=0,prompt_overlap=0,pretrained_model_calls=0))
    print(dict(new_pairs=64,history_files=len(paths),prior_observed_pairs=len(used)))
if __name__=='__main__':main()
