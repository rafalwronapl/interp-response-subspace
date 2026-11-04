"""Literal choices; no selection by activation or model performance."""
from .common import ROOT

PAIRS=[('locksmith','barrel'),('gardener','basket'),('violinist','lantern'),('sailor','anchor'),
       ('baker','oven'),('dentist','cabinet'),('painter','easel'),('courier','parcel'),
       ('plumber','faucet'),('farmer','tractor'),('dancer','mirror'),('librarian','bookcase'),
       ('tailor','needle'),('drummer','cymbal'),('driver','garage'),('jeweler','bracelet'),
       ('sculptor','pedestal'),('ranger','cabin'),('brewer','bottle'),('welder','helmet'),
       ('potter','vessel'),('florist','ribbon'),('carpenter','beam'),('pilot','hangar'),
       ('miner','tunnel'),('archer','target'),('juggler','hoop'),('weaver','loom')]
FRAMES=['The {subject} beside the {attractor}','The {subject} next to the {attractor}']

def build():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    documents=[]
    for block,(subject,attractor) in enumerate(PAIRS+[('cook','pan')]):
        split='fit' if block<12 else 'select' if block<16 else 'test' if block<28 else 'pilot'
        for frame,template in enumerate(FRAMES):
            for s in (0,1):
                for a in (0,1):
                    text=template.format(subject=subject+'s'*s,attractor=attractor+'s'*a)
                    ids=[151643]+tok.encode(text,add_special_tokens=False)
                    if len(ids)>40:raise ValueError('Sequence cap')
                    for word,t in [(' is',374),(' are',525)]:
                        if tok.encode(text+word,add_special_tokens=False)!=ids[1:]+[t]:raise ValueError('Token boundary')
                    documents.append(dict(id=f'E036_{block:02d}_F{frame}_S{s}_A{a}',block=block,split=split,frame=frame,S=s,A=a,
                         subject=subject,attractor=attractor,prompt=text,ids=ids,tokens=tok.convert_ids_to_tokens(ids)))
    for d in documents:
        donor=[r for r in documents if (r['block'],r['frame'],r['A'],r['S'])==(d['block'],d['frame'],d['A'],1-d['S'])]
        if len(donor)!=1:raise ValueError('Donor')
        d['donor']=donor[0]['id'];d['donor_length_equal']=len(d['ids'])==len(donor[0]['ids'])
    if len(documents)!=232 or len({d['prompt'] for d in documents})!=232:raise ValueError('Counts')
    lemmas=[w for pair in PAIRS+[('cook','pan')] for w in pair]
    if len(set(lemmas))!=58:raise ValueError('Lemma overlap')
    return dict(documents=documents,lemmas=lemmas,prefix=151643,
                alignment='Scalars at each prompt last token; donor length may differ and is recorded. No cross-prompt positional tensor copying.')

def conditions(doc):
    for branch in ('B','C'):
        for dose in (.5,1.):
            yield dict(id=doc['id']+'_'+branch+('_half' if dose==.5 else '_one'),branch=branch,dose=dose)

def operations(documents,stage):
    splits={'pilot':['pilot'],'development':['fit','select'],'test':['test']}[stage]
    cats={}
    def add(label,category):
        if label in cats:raise ValueError('Duplicate ID')
        cats[label]=category
    for d in documents:
        if d['split'] not in splits:continue
        add(d['id']+'_natural','network');add(d['id']+'_natural_clt','clt')
        for c in conditions(d):
            for suffix,category in [('prefix','network'),('clt','clt'),('target','target'),('jvp','jvp')]:add(c['id']+'_'+suffix,category)
        if stage=='pilot':
            for suffix in ('sham','post','native_full','negative','force','forceprefix','poison'):
                add(d['id']+'_'+suffix,'network')
            add(d['id']+'_local','target')
            for i in range(3):
                for sign in ('p','m'):add(d['id']+f'_fd{i}_{sign}','target')
            add(d['id']+'_jvp64','jvp')
    return cats
