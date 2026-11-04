"""Raw-first derivative/forecast/precision and ledger reconstruction without loading weights."""
from datetime import datetime
import numpy as np
from research_checks.gate import read
from research_checks.persistence import request_key
from e012.common import save,rows
from e024.common import payload,sha_array
from .common import ROOT,OUT,PREP,CONFIG,Measurements,ledger_summary
from .worker import perform


class Replay(Measurements):
    def __init__(self,stage):super().__init__(None,stage);self.seen=set()
    def run(self,doc,label,reference=None,**settings):
        refs={k:sha_array(np.asarray(v)) for k,v in sorted(reference.items())} if reference is not None else None
        request=dict(model_identity=self.identity['model_identity'],code_identity=self.identity['code_identity'],document=doc,
            intervention=dict(stage=self.stage,label=label,settings=settings,reference=refs),readout=doc.get('readout_ids',[374,525]))
        key=request_key(request);ix=payload(OUT/'index'/(label+'.json'));raw=payload(OUT/'raw'/(key+'.json'))
        if ix!=dict(request_key=key,request=request,result=raw['result']) or raw['request']!=request:raise ValueError('Raw request mismatch')
        rec=raw['result'];arrays=self.arrays(rec);self.seen.add(label)
        if rec['margin']!=float(np.float32(rec['scores'][0])-np.float32(rec['scores'][1])):raise ValueError('Scalar readout mismatch')
        if rec['readout']!=doc.get('readout_ids',[374,525]):raise ValueError('Wrong readout IDs')
        if reference is not None:
            if str(arrays['prefix_hash'])!=str(reference['prefix_hash']):raise ValueError('Earlier positions differ')
            expected_units={}
            if settings.get('t') is not None:expected_units['U']=float(np.float32(reference['U'])+np.float32(settings['t'])*np.float32(settings['delta']))
            if settings.get('branch','b')=='c':expected_units['W']=float(reference['W'])
            for u in ('U','W'):
                if settings.get(u+'_override') is not None:expected_units[u]=float(np.float32(settings[u+'_override']))
            actual=[a for a in rec['audits'] if a['kind']=='unit']
            if len(actual)!=len(expected_units):raise ValueError('Unit audit count')
            for u,v in expected_units.items():
                aa=[a for a in actual if a['unit']==u]
                if len(aa)!=1 or aa[0]['replacement']!=v or not aa[0]['outside_exact'] or aa[0]['position']!=len(doc['ids'])-1:raise ValueError('Unit audit value')
            clamp=settings.get('clamp_mlp20',True);aa=[a for a in rec['audits'] if a['kind']=='MLP20']
            if clamp:
                np.testing.assert_array_equal(arrays['MLP20_after'],reference['MLP20_after'])
                if len(aa)!=1 or not aa[0]['outside_exact'] or aa[0]['target_sha256']!=sha_array(np.asarray(reference['MLP20_after'])):raise ValueError('MLP20 audit')
            elif aa:raise ValueError('Unexpected MLP20 clamp')
        elif rec['audits']:raise ValueError('Unexpected natural patch')
        if settings.get('derivatives') and not np.isfinite([rec['prime'],rec['second']]).all():raise ValueError('Bad derivatives')
        return rec,arrays


def audit(stage):
    original=payload(OUT/('pilot_results.json' if stage=='pilot' else 'results.json'));m=Replay(stage)
    computed=perform(m,stage,read(PREP/'dataset.json'),persist=False)
    if computed!=original:raise ValueError('Reconstructed stage differs')
    sc=read(PREP/f'schedule_{stage}.json');ops=sc['operations']
    if original['status']=='numerically_unresolved':
        n=sum(sc['phase_lengths'][k] for k in sc['phase_order'][:-1]);ops=ops[:n]
    expected={o['id'] for o in ops};forward={o['id'] for o in ops if o['kind']=='forward'}
    if m.seen!=forward:raise ValueError('Raw forward coverage')
    events=list(rows(OUT/'ledger.jsonl'));res={x['id']:x for x in events if x['event']=='reserve'};fin={x['id']:x for x in events if x['event']=='finish'}
    if set(res)!=set(fin):raise ValueError('Unfinished reservations')
    rr={k:x for k,x in res.items() if x['stage']==stage}
    if {x['label'] for k,x in rr.items() if fin[k]['ok']}!=expected:raise ValueError('Successful operation coverage')
    freeze=datetime.fromisoformat(read(OUT/f'{stage}_forecasts.json')['created_utc'])
    pre=sc['phase_lengths']['natural']+sc['phase_lengths']['forecasts'];before={o['id'] for o in sc['operations'][:pre]}
    for k,x in rr.items():
        if x['label'] not in expected:raise ValueError('Unscheduled operation')
        if x['label'] in before:
            if fin[k]['ok'] and datetime.fromisoformat(fin[k]['utc'])>freeze:raise ValueError('Forecast after freeze')
        elif datetime.fromisoformat(x['utc'])<freeze:raise ValueError('Nonzero measurement before freeze')
    summary=ledger_summary();cfg=read(CONFIG)['budget']
    for kind in ('forward','backward'):
        if summary['kinds'][kind]>cfg['hard_'+kind+'_cap']:raise ValueError('Global count cap')
        for s,d in summary['stages'].items():
            if d[kind]>cfg['stage_caps'][s][kind]:raise ValueError('Stage count cap')
    if summary['model_seconds']>cfg['max_model_seconds'] or any(d['model_seconds']>cfg['stage_model_seconds_caps'][s] for s,d in summary['stages'].items()):raise ValueError('Time cap')
    output=dict(ok=True,stage=stage,raw_requests=len(forward),operations=len(expected),whole_stage_chronology=True,precision_recomputed=True,model_rerun=False,
        ledger=dict(attempts=len(rr),forward=sum(x['kind']=='forward' for x in rr.values()),backward=sum(x['kind']=='backward' for x in rr.values()),model_seconds=sum(fin[k]['seconds'] for k in rr)))
    save(OUT/f'audit_{stage}.json',output);return original,output


def report():
    pilot,_=audit('pilot');main=None
    if (OUT/'results.json').exists():main,_=audit('main')
    result=main or pilot;summary=dict(complete=bool(main or not pilot['ready']),pilot_ready=pilot['ready'],main_started=bool(main),
        status=result['status'],fresh_documents=128 if main and main['rows'] else 0,**ledger_summary())
    if main and main['rows']:summary['decision']=main['metrics']['status']
    save(OUT/'summary.json',summary)
    diagnostics={}
    for dose in (.5,1.):
        rr=[r for r in result['rows'] if r['dose']==dose]
        if not rr:continue
        groups=[('all',rr)]+[(f'{f}_S{s}_A{a}',[r for r in rr if (r['frame'],r['S'],r['A'])==(f,s,a)]) for f in ('F0','F1') for s in (0,1) for a in (0,1)]
        groups += [(f'{f}_S{s}',[r for r in rr if (r['frame'],r['S'])==(f,s)]) for f in ('F0','F1') for s in (0,1)]
        groups += [(f'block_{g}',[r for r in rr if r['group']==g]) for g in sorted({r['group'] for r in rr})]
        diagnostics[str(dose)]={}
        for name,cc in groups:
            C=np.array([r['C'] for r in cc]);dd=np.array([r['delta'] for r in cc]);entry=dict(n=len(cc),G=float(abs(C).mean()),sign_C={k:int(sum(fn(C))) for k,fn in [('negative',lambda v:v<0),('zero',lambda v:v==0),('positive',lambda v:v>0)]},small_C=int(sum(abs(C)<.01)),sign_delta=dict(negative=int(sum(dd<0)),zero=int(sum(dd==0)),positive=int(sum(dd>0))),methods={})
            for method in cc[0]['predictions']:
                pred=np.array([r['predictions'][method] for r in cc]);e=abs(pred-C)
                entry['methods'][method]=dict(MAE=float(e.mean()),bias=float((pred-C).mean()),q95=float(np.quantile(e,.95)),maximum=float(e.max()))
            entry['gate_up_regimes']={u:{sg+su:sum(('+' if r['gate'][u]>=0 else '-')==sg and ('+' if r['up'][u]>=0 else '-')==su for r in cc) for sg in ('-','+') for su in ('-','+')} for u in ('U','W')}
            entry.update(natural_correct=sum(r['natural_correct'] for r in cc),natural_incorrect=sum(not r['natural_correct'] for r in cc),
                delta_range=[float(dd.min()),float(dd.max())],
                gate_ranges={u:[min(r['gate'][u] for r in cc),max(r['gate'][u] for r in cc)] for u in ('U','W')},
                up_ranges={u:[min(r['up'][u] for r in cc),max(r['up'][u] for r in cc)] for u in ('U','W')})
            diagnostics[str(dose)][name]=entry
    events=list(rows(OUT/'ledger.jsonl'));res={e['id']:e for e in events if e['event']=='reserve'};fin={e['id']:e for e in events if e['event']=='finish'}
    costs={k:dict(forward=0,backward=0,model_seconds=0.) for k in ('natural','forecast','controls','outcomes')}
    for key,event in res.items():
        label=event['label']
        category='natural' if label.endswith('_natural') else 'forecast' if label.endswith(('_zero','_prime','_second')) else 'outcomes' if label.endswith(('_half','_one')) else 'controls'
        costs[category][event['kind']]+=1
        if key in fin:costs[category]['model_seconds']+=fin[key]['seconds']
    diagnostics['actual_cost_split']=costs
    save(OUT/'diagnostics.json',diagnostics)
    lines=['# E031 Transfer lokalnej prognozy do1.7B','',f"Status: {summary['status']}; pilot ready={summary['pilot_ready']}; nowe zdania={summary['fresh_documents']}.",'',
        f"Forwardy={summary['kinds']['forward']}, backwardy={summary['kinds']['backward']}, czas modelu={summary['model_seconds']:.3f}s.",'']
    if summary.get('decision'):lines.append('Decyzja dla dawki1: '+summary['decision'])
    lines += ['', 'Wynik dotyczy ustalonego U19/W21 przy naturalnym restore MLP20. Sam sukces na dawce0.5 nie potwierdza dawki1. Opisowe przedziały16 bloków nie gwarantują mocy. Wynik negatywny nie jest podstawą do zmiany neuronów lub progów w E031.',
        '', 'Pochodne są liczone w FP32 i wymagają uprzednio określonej kontroli FD. numerically_unresolved oznacza brak testu transferu. P2 jest droższy od bezpośredniego pomiaru kontrastu; nie wykazujemy oszczędności.',
        '', '[Plan](../../plans/E031_PLAN.md) · [Diagnostyka](diagnostics.json)']
    (OUT/'PODSUMOWANIE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(summary)


if __name__=='__main__':report()
