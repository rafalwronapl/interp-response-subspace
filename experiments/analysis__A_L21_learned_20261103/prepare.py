"""Freeze only previously approved FIT sources; exclude TEST targets entirely."""
from pathlib import Path
import json
from research_checks.gate import digest
from research_checks.persistence import immutable
from .common import ROOT,BASE,read,sha,rel,unwrap
OLD=ROOT/'analysis/AB_five_experiments_20261103/A_L21'
def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise ValueError('Already prepared '+str(p))
    p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
    pinned=unwrap(OLD/'main_run/completed.json')['files'];manifest={};index=dict(natural={},cells=[])
    corpus=read(OLD/'corpus.json')['documents'];docs=[d for d in corpus if d['split'].startswith('FIT')]
    assert len(docs)==96
    def bind(p,known=None):
        name=rel(p);h=sha(p)
        if known is not None and h!=known:raise ValueError('Changed accepted evidence '+name)
        manifest[name]=h;return name
    def source(op,d):
        receipt=OLD/'main_run/bundles'/f'{op}.json';r=unwrap(receipt);cache=OLD/'main_run/cache'/f"{r['request']}.json"
        assert r['operations']==[op]
        for p in [receipt,cache]:bind(p,pinned[p.relative_to(OLD/'main_run').as_posix()])
        payload=unwrap(cache);q=payload['request'];assert q['document']==d and q['intervention']['logical_operations']==[op]
        return dict(receipt=rel(receipt),cache=rel(cache),document=d)
    for d in docs:
        index['natural'][d['id']]=source(d['id']+'_natural',d)
        for branch in ['B','C']:
            for dose in [.5,1.]:
                tag=d['id']+'_'+branch+'_'+str(dose)
                index['cells'].append(dict(id=tag,document=d,branch=branch,dose=dose,prefix=source(tag+'_prefix',d),target=source(tag+'_target',d)))
    for p in [OLD/'main_run/completed.json',OLD/'completion_review.json',OLD/'corpus.json',ROOT/'plans/A_L21_LEARNED_20261103/PLAN.txt',ROOT/'plans/A_L21_LEARNED_20261103/review/plan_review_v1.json']:bind(p)
    prior=bind(ROOT/'runs/E036/development/predictors.npz')
    put(BASE/'source_index.json',index);bind(BASE/'source_index.json')
    put(BASE/'config.json',dict(prior=prior,threads=2,scope='stage1 only, FIT vector labels; no model calls'));bind(BASE/'config.json')
    for n in ['common.py','models.py','stage1.py','fixtures.py','fixtures.json','prepare.py']:bind(BASE/n)
    for n in ['research_checks/gate.py','research_checks/persistence.py','research_checks/reachability.py','research_checks/__init__.py']:bind(ROOT/n)
    put(BASE/'input_manifest.json',dict(files=manifest))
    put(BASE/'operations.json',dict(operations=[]))
    put(BASE/'contract.json',dict(schema_version=1,id='A_L21_LEARNED_STAGE1_20261103',kind='experiment',primary_agent='/root',
        purpose='Previously captured FIT-only nested coefficient/vector feasibility and384-label model freeze. Zero native/TC evaluations; no readout utility claim.',
        tracked_files=[rel(BASE/n) for n in ['stage1.py','models.py','common.py','prepare.py','fixtures.py','fixtures.json']],tracked_trees=[],input_manifests=[rel(BASE/'input_manifest.json')],
        gates=[dict(id=k,category=k,evidence=rel(BASE/'fixtures.json'),pointer='/passed',op='eq',expected=True,reason='Pure NumPy regression and group-separated FIT sources; actual inputs/implementation reviewed independently.') for k in ['hooks','data','pilot','accounting','restart']],
        circuits=[],structural_scope='No native-model computation or circuit; FIT-only supervised algebra, distinct from future held-out readout stage.',
        entrypoint=dict(module='analysis.A_L21_learned_20261103.stage1',args=[]),schedule=rel(BASE/'operations.json'),budget=dict(forward=0,backward=0)))
    print(dict(FIT_documents=len(docs),FIT_conditions=len(index['cells']),bound_files=len(manifest)))
if __name__=='__main__':main()
