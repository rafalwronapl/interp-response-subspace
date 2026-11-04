"""Materialize new literal evaluation adapter; preserve all old source files."""
from pathlib import Path
from analysis.C2_fresh_fit_holdout_20261102.worker import ROOT
BASE=Path(__file__).parent
t=(ROOT/'analysis/C3_noD_more_models_20261102/evaluate.py').read_text(encoding='utf-8')
t=t.replace("assert cfg['model'] in ['llama','gemma270']", "assert cfg['model']=='qwen06_base'")
t=t.replace("assert shutil.disk_usage(stage).free>1073741824", "assert str(out.resolve())==cfg['physical_output'];assert shutil.disk_usage(out.resolve()).free>1073741824")
t=t.replace("assert len(holdout)==128", "assert len(holdout)==2048")
t=t.replace("analytic_gradients=384", "analytic_gradients=6144")
t=t.replace("==1152", "==18432")
t=t.replace("forward=5248", "forward=102400")
t=t.replace("for kind in ['direct','full','proj','oracle']", "for kind in ['direct','full','proj','oracle','outside']")
needle="c=(res@us[j])@us[j].T;values['oracle_'+i+'_'+j]=p+c;score=float((J@c)**2-2*(J@res)*(J@c));oracle_records.append(dict(document=d['id'],block=d['block'],side=d['side'],q=d['q'],i=i,j=j,score=score))"
replacement="c=(res@us[j])@us[j].T;values['oracle_'+i+'_'+j]=p+c;values['outside_'+i+'_'+j]=D-c;score=float((J@c)**2-2*(J@res)*(J@c));oracle_records.append(dict(document=d['id'],block=d['block'],frame=d['frame'],side=d['side'],q=d['q'],i=i,j=j,score=score,energy_inside=float(c@c),energy_outside=float((res-c)@(res-c))))"
assert needle in t;t=t.replace(needle,replacement)
t=t.replace("for k in ['block','side','q']", "for k in ['block','frame','side','q']")
t=t.replace("block=d['block'],side=d['side'],q=d['q'],arm=arm", "block=d['block'],frame=d['frame'],side=d['side'],q=d['q'],arm=arm")
(BASE/'evaluate.py').write_text(t,encoding='utf-8')
print('Literal Q06evaluation adapter materialized,50arms102400head0loads')
if __name__=='__main__':pass
