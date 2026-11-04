import json,re,itertools
from pathlib import Path
from transformers import AutoTokenizer
from research_checks.gate import read,sha
from analysis.AB_five_experiments_20261103.A_L21.prepare_data import NOUNS,plural,strings
R=Path(__file__).resolve().parents[3];B=R/'analysis/A_transfer_next_20261103';c=read(B/'corpus.json');ds=c['documents'];old=read(R/'analysis/AB_budget_readout_20261103/corpus.json')['documents'];assert c['test_documents']==old
assert len(ds)==24 and len(c['pairs'])==len(set(map(tuple,c['pairs'])))==6
roots={w:w for w in NOUNS};roots.update({plural(w):w for w in NOUNS});used=set()
for p,h in read(B/'freshness_inputs.json')['files'].items():
 assert sha(R/p)==h
 for t in strings(read(R/p)):
  if len(t)<=1000:
   words=sorted({roots[w] for w in re.findall('[a-z]+',t.lower()) if w in roots});used.update(itertools.combinations(words,2))
assert not set(map(tuple,c['pairs']))&used
tok=AutoTokenizer.from_pretrained(R/'model',local_files_only=True)
for d in ds+old:
 for w,i in c['heads'].items():assert tok.encode(d['prompt']+w,add_special_tokens=False)==d['ids'][1:]+[i]
mf=read(B/'input_manifest.json')['files'];ad=read(B/'adoption_index.json')['entries'];assert len(ad)==2304
for op,e in ad.items():
 assert e['bundle'] in mf and e['cache'] in mf
 assert read(R/e['bundle'])['payload']['request']==e['request']
 assert Path(e['cache']).stem==e['request']
assert len(read(B/'operations.json')['operations'])==12516
out=dict(passed=True,new_fit_cohorts=3,new_fit_blocks=6,new_fit_labels=96,old_test_blocks=64,adopted_records=2304,contextual_token_checks=1120,exposure_files=len(read(B/'freshness_inputs.json')['files']),logical_forward=12516,physical_forward=1248,scope='Old TEST posthoc; three whole-block2fold FITs; no pretrained calls; historical completed raw cache independently canonical verified before this review')
(B/'review/independent_pre_run_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
