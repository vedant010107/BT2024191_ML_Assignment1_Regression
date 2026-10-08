"""Polynomial regression: model selection, training, evaluation and inference."""
import os
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[variable]='1'
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from skglm import ElasticNet
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
RIDGE_ALPHAS=[.0001,.001,.01,.1,1.,10.,100.,1000.]
SPARSE_ALPHAS=[1.,.1,.01,.001]
RATIOS=[.2,.5,.8]
TOL=1e-4
OUTER_SEED=43
INNER_SEED=42


def log_degree(rows, cached=False):
    valid=[r for r in rows if r['eligible']]
    best=min(valid,key=lambda r:r['mean_validation_mse']) if valid else None
    score=f'{best["mean_validation_mse"]:.9f}' if best else 'unavailable'
    settings=f'; {best["config"]}' if best else ''
    print(f'  degree {rows[0]["config"]["degree"]:2d}: best inner MSE={score}, '
          f'invalid candidates={len(rows)-len(valid)}/{len(rows)}'
          f'{settings}'+(' [cached]' if cached else ''),flush=True)


def log_repeated(rows, cached=False):
    print('  Repeated inner validation: 5 folds x 2 repeats on the fixed shortlist'
          +(' [cached]' if cached else ''),flush=True)
    for r in rows:
        score=f'{r["mean_validation_mse"]:.9f}' if r['eligible'] else 'excluded'
        failures=sum(v is None for v in r['validation_mse'])
        print(f'    {r["config"]}: mean MSE={score}; failed fits={failures}/10',flush=True)


def replay_search(directory,max_degree,question):
    for degree in range(1,max_degree+1):
        log_degree(json.loads((directory/f'degree_{degree:02}.json').read_text()),cached=True)
    if question==2:
        log_repeated(json.loads((directory/'repeated_selection.json').read_text()),cached=True)


def verify_outputs(question):
    folder=ROOT/f'Q{question}'
    data=pd.read_csv(ROOT/f'BT2024191/BT2024191_train_var{question}.csv')
    test=pd.read_csv(ROOT/f'BT2024191/BT2024191_test_var{question}.csv')
    values=pd.read_csv(folder/f'BT2024191_pred_var{question}.csv')
    assert list(test)==list(data.drop(columns='y'))
    assert list(values)==['y'] and len(values)==len(test) and np.isfinite(values.y).all()
    model=joblib.load(folder/'model.joblib')
    np.testing.assert_allclose(values.y,predict(model,test.to_numpy()),rtol=0,atol=1e-10)
    oof=pd.read_csv(folder/'validation_predictions.csv')
    np.testing.assert_allclose(oof.actual,data.y)
    metrics=json.loads((folder/'nested_cv.json').read_text())
    np.testing.assert_allclose(mean_squared_error(oof.actual,oof.predicted),metrics['pooled_mse'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(r2_score(oof.actual,oof.predicted),metrics['pooled_r2'],rtol=0,atol=1e-12)
    final=json.loads((folder/'final_model.json').read_text())
    assert model['config']==final['config']
    np.testing.assert_allclose(mean_squared_error(data.y,predict(model,data.drop(columns='y').to_numpy())),final['training_mse'],rtol=0,atol=1e-10)
    print(f'  Verification passed: {len(values)} finite predictions, model reload, training MSE and pooled CV metrics.',flush=True)


def print_summary(question):
    folder=ROOT/f'Q{question}'
    final=json.loads((folder/'final_model.json').read_text())
    metrics=json.loads((folder/'nested_cv.json').read_text())
    print('\nFINAL SUMMARY',flush=True)
    print(json.dumps(dict(problem=question,selected=final['config'],
        final_selection_cv_mse=final['selection_mse'],training_mse=final['training_mse'],
        pooled_outer_mse=metrics['pooled_mse'],pooled_outer_r2=metrics['pooled_r2'],
        mean_outer_mse=metrics['mean_fold_mse'],outer_mse_sd=metrics['fold_mse_sd'],
        terms=final['terms'],nonzero_coefficients=final['nonzero'],
        training_rows=len(pd.read_csv(ROOT/f'BT2024191/BT2024191_train_var{question}.csv')),
        test_rows=final['prediction_rows'],prediction_file=str(folder/f'BT2024191_pred_var{question}.csv'),
        model_file=str(folder/'model.joblib'),degree_plot=str(folder/'images/degree_mse.png'),
        note='Final-selection CV MSE is a selection score. Outer-CV scores estimate the selection procedure.'),indent=2),flush=True)


def write_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False))
    temporary.replace(path)


def splits(x,n,seed):
    groups=np.unique(x,axis=0,return_inverse=True)[1]
    return list(GroupKFold(n,shuffle=True,random_state=seed).split(x,groups=groups))


def transform_fit(x,degree):
    inputs=StandardScaler().fit(x)
    poly=PolynomialFeatures(degree,include_bias=False).fit(inputs.transform(x))
    expanded=poly.transform(inputs.transform(x))
    columns=StandardScaler().fit(expanded)
    return dict(input_scaler=inputs,polynomial=poly,scaler=columns),columns.transform(expanded)


def transform(model,x):
    return model['scaler'].transform(model['polynomial'].transform(model['input_scaler'].transform(x)))


def predict(model,x):
    return transform(model,x)@model['coef']+model['intercept']


def config(degree,family,alpha,ratio=0.):
    return dict(degree=int(degree),family=family,alpha=float(alpha),l1_ratio=float(ratio))


def gap(a,y,w,alpha,ratio):
    residual=y-a@w;l1=len(y)*alpha*ratio;l2=len(y)*alpha*(1-ratio)
    scale=min(1.,l1/max(float(np.max(np.abs(a.T@residual-l2*w))),1e-300))
    value=(.5*(1+scale**2)*(residual@residual)+l1*np.abs(w).sum()
           -scale*(residual@y)+.5*l2*(1+scale**2)*(w@w))/len(y)
    return max(float(value),0.)


def sparse_path(a,y,alphas,ratio):
    solver=ElasticNet(l1_ratio=ratio,fit_intercept=False,warm_start=True,
                      tol=TOL*.1,max_iter=20,max_epochs=100)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',ConvergenceWarning)
        _,coefs,_,_=solver.path(np.asfortranarray(a),np.ascontiguousarray(y),
                                alphas=np.asarray(alphas),return_n_iter=True)
    for alpha,w in zip(alphas,coefs.T):
        error=gap(a,y,w,alpha,ratio)
        yield w,bool(np.isfinite(w).all() and error<=TOL*np.mean(y*y)),error


def coefficients(a,y,degree):
    values,vectors=eigh(a@a.T,check_finite=False)
    projected=vectors.T@y
    for alpha in RIDGE_ALPHAS:
        w=a.T@(vectors@(projected/(np.maximum(values,0)+alpha)))
        yield config(degree,'ridge',alpha),w,bool(np.isfinite(w).all()),0.
    for ratio in [1.]+RATIOS:
        family='lasso' if ratio==1. else 'elastic_net'
        for alpha,(w,ok,error) in zip(SPARSE_ALPHAS,sparse_path(a,y,SPARSE_ALPHAS,ratio)):
            yield config(degree,family,alpha,ratio),w,ok,error


def fit(x,y,c):
    model,a=transform_fit(x,c['degree']);mean=float(y.mean());yc=y-mean
    if c['family']=='ridge':
        values,vectors=eigh(a@a.T,check_finite=False)
        w=a.T@(vectors@((vectors.T@yc)/(np.maximum(values,0)+c['alpha'])))
    else:
        alphas=[v for v in SPARSE_ALPHAS if v>=c['alpha']]
        w,ok,error=list(sparse_path(a,yc,alphas,c['l1_ratio']))[-1]
        if not ok:raise RuntimeError(f'Selected refit did not converge: dual gap {error}')
    model.update(coef=w,intercept=mean,config=c)
    return model


def degree_search(x,y,folds,degree,directory):
    file=directory/f'degree_{degree:02}.json'
    if file.exists():
        result=json.loads(file.read_text());log_degree(result,cached=True);return result
    rows={}
    with threadpool_limits(limits=1):
        for tr,va in folds:
            model,a=transform_fit(x[tr],degree);b=transform(model,x[va]);mean=y[tr].mean()
            for c,w,ok,error in coefficients(a,y[tr]-mean,degree):
                k=json.dumps(c,sort_keys=True)
                r=rows.setdefault(k,dict(config=c,train_mse=[],validation_mse=[],dual_gaps=[],failed_folds=0))
                r['train_mse'].append(float(mean_squared_error(y[tr],a@w+mean)) if ok else None)
                r['validation_mse'].append(float(mean_squared_error(y[va],b@w+mean)) if ok else None)
                r['dual_gaps'].append(error if np.isfinite(error) else None)
                r['failed_folds']+=int(not ok)
    for r in rows.values():
        r['eligible']=r['failed_folds']==0
        r['mean_validation_mse']=float(np.mean(r['validation_mse'])) if r['eligible'] else None
        r['mean_train_mse']=float(np.mean(r['train_mse'])) if r['eligible'] else None
    result=list(rows.values());write_json(file,result)
    log_degree(result)
    return result


def winner(rows):
    valid=[r for r in rows if r['eligible']]
    if not valid:raise RuntimeError('No candidate converged in all inner folds.')
    return min(valid,key=lambda r:(r['mean_validation_mse'],r['config']['degree'],json.dumps(r['config'],sort_keys=True)))


def select_model(x,y,max_degree,directory,jobs,question):
    directory.mkdir(parents=True,exist_ok=True)
    folds=splits(x,3,INNER_SEED)
    write_json(directory/'inner_splits.json',[dict(train=a.tolist(),validation=b.tolist()) for a,b in folds])
    batches=joblib.Parallel(n_jobs=jobs)(joblib.delayed(degree_search)(x,y,folds,d,directory) for d in range(1,max_degree+1))
    rows=[r for batch in batches for r in batch]
    pd.DataFrame([dict(**r['config'],mean_train_mse=r['mean_train_mse'],mean_validation_mse=r['mean_validation_mse'],
                       eligible=r['eligible'],failed_folds=r['failed_folds']) for r in rows]).to_csv(directory/'search_scores.csv',index=False)
    if question==2:
        file=directory/'repeated_selection.json'
        if file.exists():
            repeated=json.loads(file.read_text());log_repeated(repeated,cached=True)
            return winner(repeated)
        shortlist=[]
        for family in ['ridge','lasso','elastic_net']:
            seen=set()
            for r in sorted([r for r in rows if r['eligible'] and r['config']['family']==family],key=lambda r:r['mean_validation_mse']):
                if r['config']['degree'] in seen:continue
                seen.add(r['config']['degree']);shortlist.append(r['config'])
                if len(seen)==2:break
        repeated=[]
        repeated_folds=[s for repeat in range(2) for s in splits(x,5,INNER_SEED+100+repeat)]
        write_json(directory/'repeated_splits.json',[dict(train=a.tolist(),validation=b.tolist()) for a,b in repeated_folds])
        for c in shortlist:
            scores=[]
            for tr,va in repeated_folds:
                try:scores.append(float(mean_squared_error(y[va],predict(fit(x[tr],y[tr],c),x[va]))))
                except RuntimeError:scores.append(None)
            valid=all(v is not None for v in scores)
            repeated.append(dict(config=c,validation_mse=scores,eligible=valid,
                                 mean_validation_mse=float(np.mean(scores)) if valid else None))
        write_json(file,repeated);log_repeated(repeated);return winner(repeated)
    return winner(rows)


def run(question,jobs):
    folder=ROOT/f'Q{question}';folder.mkdir(exist_ok=True)
    train=ROOT/f'BT2024191/BT2024191_train_var{question}.csv'
    test=ROOT/f'BT2024191/BT2024191_test_var{question}.csv'
    data=pd.read_csv(train);x=data.drop(columns='y').to_numpy();y=data.y.to_numpy()
    xt=pd.read_csv(test).to_numpy()
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(xt).all()
    max_degree=10 if question==1 else 20
    print('\n'+'='*78+f'\nQUESTION {question}: POLYNOMIAL REGRESSION\n'+'='*78,flush=True)
    print(f'Training rows={len(y)}; test rows={len(xt)}; inputs={x.shape[1]}; degrees=1..{max_degree}',flush=True)
    print('Models: Ridge, Lasso, Elastic Net | 24 configurations per degree',flush=True)
    print('Outer CV: 5 grouped folds | Inner screening: 3 grouped folds',flush=True)
    if question==2:print('Q2 selection: fixed shortlist, then 5 inner folds repeated twice',flush=True)
    manifest=dict(question=question,max_degree=max_degree,outer_folds=5,inner_folds=3,
                  outer_seed=OUTER_SEED,inner_seed=INNER_SEED,ridge_alphas=RIDGE_ALPHAS,
                  sparse_alphas=SPARSE_ALPHAS,l1_ratios=RATIOS,tolerance=TOL,
                  train_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),
                  test_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  versions={p:__import__(p).__version__ for p in ['numpy','scipy','sklearn','skglm','pandas','joblib']})
    if (folder/'manifest.json').exists():
        saved_manifest=json.loads((folder/'manifest.json').read_text())
        if saved_manifest!=manifest:
            # Validate compatibility with the recorded training implementation.
            compatibility=folder/'logging_compatibility.json'
            allowed=json.loads(compatibility.read_text()) if compatibility.exists() else {}
            same_settings=all(saved_manifest.get(k)==v for k,v in manifest.items() if k!='code_sha256')
            if not (same_settings and allowed.get('training_source_sha256')==saved_manifest.get('code_sha256')
                    and allowed.get('logging_source_sha256')==manifest['code_sha256']):
                raise RuntimeError('Code/data/settings changed; use a separate output folder.')
    else:write_json(folder/'manifest.json',manifest)
    outer=splits(x,5,OUTER_SEED)
    write_json(folder/'outer_splits.json',[dict(train=a.tolist(),validation=b.tolist()) for a,b in outer])
    evaluations=[];out_of_fold=np.empty(len(y))
    with threadpool_limits(limits=1):
        for number,(tr,va) in enumerate(outer,1):
            directory=folder/'search'/f'outer_{number}'
            saved=directory/'evaluation.json'
            print(f'\nOuter fold {number}/5 | training={len(tr)}, held out={len(va)}',flush=True)
            if saved.exists():
                replay_search(directory,max_degree,question)
                result=json.loads(saved.read_text());prediction=pd.read_csv(directory/'predictions.csv').predicted.to_numpy()
            else:
                choice=select_model(x[tr],y[tr],max_degree,directory,jobs,question)
                model=fit(x[tr],y[tr],choice['config']);prediction=predict(model,x[va])
                result=dict(fold=number,config=choice['config'],inner_mse=choice['mean_validation_mse'],
                            training_mse=float(mean_squared_error(y[tr],predict(model,x[tr]))),
                            mse=float(mean_squared_error(y[va],prediction)),r2=float(r2_score(y[va],prediction)),
                            train_rows=len(tr),validation_rows=len(va))
                pd.DataFrame(dict(row=va,actual=y[va],predicted=prediction)).to_csv(directory/'predictions.csv',index=False)
                write_json(saved,result)
            out_of_fold[va]=prediction;evaluations.append(result)
            print(f'  outer MSE={result["mse"]:.9f}, R2={result["r2"]:.9f}; {result["config"]}',flush=True)
        if (folder/'validation_predictions.csv').exists():
            cached_oof=pd.read_csv(folder/'validation_predictions.csv')
            np.testing.assert_allclose(cached_oof.actual,y,rtol=0,atol=1e-10)
            np.testing.assert_allclose(cached_oof.predicted,out_of_fold,rtol=0,atol=1e-10)
        else:
            pd.DataFrame(dict(actual=y,predicted=out_of_fold)).to_csv(folder/'validation_predictions.csv',index=False)
        metrics=dict(pooled_mse=float(mean_squared_error(y,out_of_fold)),pooled_r2=float(r2_score(y,out_of_fold)),
                     mean_fold_mse=float(np.mean([r['mse'] for r in evaluations])),
                     fold_mse_sd=float(np.std([r['mse'] for r in evaluations],ddof=1)),folds=evaluations)
        if (folder/'nested_cv.json').exists():
            cached_metrics=json.loads((folder/'nested_cv.json').read_text())
            np.testing.assert_allclose(cached_metrics['pooled_mse'],metrics['pooled_mse'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(cached_metrics['pooled_r2'],metrics['pooled_r2'],rtol=0,atol=1e-12)
        else:write_json(folder/'nested_cv.json',metrics)
        directory=folder/'search'/'full'
        print('\nFinal selection on all training rows (outer scores are not used to select).',flush=True)
        complete=all((folder/name).exists() for name in ['final_model.json','model.joblib',f'BT2024191_pred_var{question}.csv'])
        if complete:
            replay_search(directory,max_degree,question)
            verify_outputs(question)
            if not all((folder/'images'/name).exists() for name in ['degree_mse.png','validation_predictions.png']):plot(question)
            print_summary(question)
            return
        choice=select_model(x,y,max_degree,directory,jobs,question)
        model=fit(x,y,choice['config']);joblib.dump(model,folder/'model.joblib')
        values=predict(model,xt)
        pd.DataFrame({'y':values}).to_csv(folder/f'BT2024191_pred_var{question}.csv',index=False)
        np.testing.assert_allclose(values,predict(joblib.load(folder/'model.joblib'),xt),atol=1e-10)
        write_json(folder/'final_model.json',dict(config=choice['config'],selection_mse=choice['mean_validation_mse'],
                   training_mse=float(mean_squared_error(y,predict(model,x))),terms=len(model['coef']),
                   nonzero=int(np.count_nonzero(model['coef'])),prediction_rows=len(values)))
    assert hashlib.sha256(train.read_bytes()).hexdigest()==manifest['train_sha256']
    assert hashlib.sha256(test.read_bytes()).hexdigest()==manifest['test_sha256']
    plot(question)
    verify_outputs(question)
    print_summary(question)


def plot(question):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ROOT/f'Q{question}';images=folder/'images';images.mkdir(exist_ok=True)
    table=pd.read_csv(folder/'search/full/search_scores.csv')
    eligible=table[table.eligible].copy()
    best=eligible.loc[eligible.groupby(['degree','family']).mean_validation_mse.idxmin()]
    best.to_csv(folder/'degree_mse.csv',index=False)
    plt.rcParams.update({'font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for family,style in [('ridge','-'),('lasso','--'),('elastic_net',':')]:
        rows=best[best.family==family].sort_values('degree')
        axes[0].plot(rows.degree,rows.mean_validation_mse,style,marker='o',markersize=3,label=family,color='black')
    axes[0].set(xlabel='Polynomial degree',ylabel='Mean inner-validation MSE',yscale='log')
    axes[0].legend();axes[0].grid(alpha=.2)
    by_degree=eligible.loc[eligible.groupby('degree').mean_validation_mse.idxmin()].sort_values('degree')
    axes[1].plot(by_degree.degree,by_degree.mean_train_mse,'k--',label='Training MSE')
    axes[1].plot(by_degree.degree,by_degree.mean_validation_mse,'k-',label='Validation MSE')
    axes[1].set(xlabel='Polynomial degree',ylabel='MSE',yscale='log');axes[1].legend();axes[1].grid(alpha=.2)
    chosen=json.loads((folder/'final_model.json').read_text())['config']['degree']
    for axis in axes:axis.axvline(chosen,color='.5',linestyle=':')
    fig.savefig(images/'degree_mse.png',dpi=200);plt.close(fig)
    oof=pd.read_csv(folder/'validation_predictions.csv')
    fig,axis=plt.subplots(figsize=(5,4.5),layout='constrained')
    axis.scatter(oof.actual,oof.predicted,s=8,color='.3',alpha=.4)
    limits=[min(oof.actual.min(),oof.predicted.min()),max(oof.actual.max(),oof.predicted.max())]
    axis.plot(limits,limits,'k--');axis.set(xlabel='Actual y',ylabel='Outer-fold predicted y')
    fig.savefig(images/'validation_predictions.png',dpi=200);plt.close(fig)


def check():
    from sklearn.linear_model import Ridge,ElasticNet as ReferenceElasticNet
    rng=np.random.default_rng(99);x=rng.normal(size=(60,3));y=x[:,0]**2+x[:,1]+rng.normal(0,.1,60)
    duplicate=np.repeat(x,2,axis=0)
    for tr,va in splits(duplicate,5,43):assert not set(map(tuple,duplicate[tr])) & set(map(tuple,duplicate[va]))
    for family,alpha,ratio in [('ridge',.1,0.),('lasso',.01,1.),('elastic_net',.01,.5)]:
        c=config(2,family,alpha,ratio);m=fit(x,y,c);z=transform(m,x)
        reference=(Ridge(alpha=alpha,solver='svd') if family=='ridge' else
                   ReferenceElasticNet(alpha=alpha,l1_ratio=ratio,tol=1e-10,max_iter=100000)).fit(z,y)
        np.testing.assert_allclose(predict(m,x),reference.predict(z),atol=.002)
        np.testing.assert_allclose(m['input_scaler'].mean_,x.mean(axis=0))
    print('Checks passed: duplicate isolation, train-only scaling, Ridge/Lasso/Elastic Net solver agreement.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--question',choices=['1','2','both'],default='both')
    parser.add_argument('--action',choices=['run','predict','plot','check'],default='run')
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args()
    if args.jobs<1:parser.error('--jobs must be positive')
    with threadpool_limits(limits=1):
        if args.action=='check':check()
        else:
            for question in ([1,2] if args.question=='both' else [int(args.question)]):
                if args.action=='run':run(question,args.jobs)
                elif args.action=='plot':
                    plot(question);print(f'Q{question} plots saved in {ROOT/f"Q{question}/images"}',flush=True)
                else:
                    folder=ROOT/f'Q{question}'
                    x=pd.read_csv(ROOT/f'BT2024191/BT2024191_test_var{question}.csv').to_numpy()
                    y=predict(joblib.load(folder/'model.joblib'),x)
                    pd.DataFrame({'y':y}).to_csv(folder/f'BT2024191_pred_var{question}.csv',index=False)
                    print(f'Q{question}: saved {len(y)} predictions to {folder/f"BT2024191_pred_var{question}.csv"}',flush=True)

if __name__=='__main__':main()
