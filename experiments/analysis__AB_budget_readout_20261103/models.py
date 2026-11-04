"""Materialize already selected FIT predictors; no new tuning or native model."""
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from research_checks.persistence import immutable
from research_checks.gate import digest
from analysis.AB_fast_followups_20261103.A_budget import worker as old
from analysis.A_L21_learned_20261103.common import packed,array,sha,read,unwrap
from analysis.A_L21_learned_20261103.models import fit,predict,loss
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
BUDGETS=[32,64,128,256,384]
LEARNED=[f'{method}_{n}' for n in BUDGETS for method in ['C','same_U','Direct']]

def unpack_model(m):return {k:array(v) if isinstance(v,dict) and set(v)=={'dtype','shape','data'} else v for k,v in m.items()}
def forecasts(x,prior,bundle):
    U=array(bundle['U']);P=np.asarray(x@prior['B5_W']@prior['B5_U'].T,np.float32).astype(float);result=dict(P=P)
    for name,model in bundle['models'].items():
        raw=predict(unpack_model(model),x)
        result[name]=P+raw@U.T if name.startswith('C_') else raw@U.T if name.startswith('same_U_') else raw
    assert set(result)==set(LEARNED+['P'])
    if any(v.shape!=P.shape or not np.isfinite(v).all() for v in result.values()):raise ValueError('Invalid forecast')
    return result

def main():
    data=old.load();x,D,P,U,block,group,meta,_=data;joint=np.concatenate([(D-P)@U,D@U,D],1)
    identity=dict(old_inputs_sha256=sha(old.BASE/'input_manifest.json'),old_worker_sha256=sha(Path(old.__file__)),materializer_sha256=sha(Path(__file__)))
    if (BASE/'MODELS.json').exists() and unwrap(BASE/'MODELS.json')['identity']!=identity:raise ValueError('Materialized model identity changed')
    models={};evidence=[];newfits=0
    with threadpool_limits(limits=2):
        for budget in BUDGETS[:-1]:
            source=old.BASE/'run/subsets'/f'FIT0_s17_n{budget}.json';r=unwrap(source);train=np.isin(block,r['spec']['train_blocks']);valid=group=='FIT0'
            assert int(train.sum())==budget and not np.any(train&valid)
            fitted={}
            for method in ['C','same_U','direct']:
                param=r['selected'][method];key=digest(param)
                request=dict(identity=identity,budget=budget,train_rows=np.flatnonzero(train).tolist(),param=param,source_sha256=sha(source))
                path=BASE/'materialized_fits'/f'{digest(request)}.json'
                if path.exists():
                    payload=unwrap(path);assert payload['request']==request;model=payload['model']
                else:
                    m=fit(x[train],joint[train],param);model={k:packed(v) if isinstance(v,np.ndarray) else v for k,v in m.items()}
                    immutable(path,dict(request=request,model=model));newfits+=1
                fitted[key]=model
                sl=slice(0,8) if method=='C' else slice(8,16) if method=='same_U' else slice(16,None)
                name=('Direct' if method=='direct' else method)+f'_{budget}'
                models[name]={**model,'alpha':packed(array(model['alpha'])[:,sl]),'ymean':packed(array(model['ymean'])[sl])}
                raw=predict(unpack_model(models[name]),x[valid]);pred=P[valid]+raw@U.T if method=='C' else raw@U.T if method=='same_U' else raw
                measured=loss(D[valid],pred,block[valid]);expected=r['metrics'][method]['vector_L2']
                assert np.isclose(measured,expected,atol=1e-9,rtol=1e-9)
                evidence.append(dict(arm=name,param=param,train_blocks=r['spec']['train_blocks'],vector_L2=measured,saved_vector_L2=expected,source_sha256=sha(source)))
    original=ROOT/'analysis/A_L21_learned_20261103/stage1_run/MODELS.json';full=unwrap(original)
    np.testing.assert_array_equal(array(full['U']),U)
    for name,key in [('C_384','C'),('same_U_384','same_U'),('Direct_384','direct')]:models[name]=full['models'][key]
    immutable(BASE/'MODELS.json',dict(models=models,U=packed(U),historical_prior_sha256=sha(ROOT/'runs/E036/development/predictors.npz'),
        identity=identity,full384_source_sha256=sha(original),selection_rule='FIT0 outer split, seed17; first lexicographic fixed sequence for compute, chosen before newTEST; no reselection',
        lower_budget_materialization=evidence))
    immutable(BASE/'model_checks.json',dict(passed=True,distinct_materialized_joint_solves=len(list((BASE/'materialized_fits').glob('*.json'))),lower_budget_models=12,full384_models_reused=3,
        saved_validation_losses_reproduced=True,no_new_hyperparameter_selection=True,native_calls=0,models_sha256=sha(BASE/'MODELS.json')))
    print(dict(materialized=True,new_joint_solves=newfits,native_calls=0))
if __name__=='__main__':main()
