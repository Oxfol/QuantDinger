#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
import v4_temporal_optimizer as v

MODELS=[('extra',3,10),('extra',5,15),('extra',None,20),('hist',None,10),('hist',None,20)]
FAMILIES=['all','launch','momentum','pullback','oversold']
QUANTILES=[.70,.80,.85,.90,.93,.95,.97]

def make_model(spec, trees=180):
    name,depth,leaf=spec
    if name=='extra': return ExtraTreesRegressor(n_estimators=trees,max_depth=depth,min_samples_leaf=leaf,max_features=.8,random_state=v.SEED,n_jobs=-1)
    return HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=15,min_samples_leaf=leaf,l2_regularization=1.0,random_state=v.SEED)

def metric(r):
    if r['days']<5 or r['trades']<15 or r['max_dd']>.30: return -1e9
    robust=r['net_without_top3']/max(1,r['days'])
    return min(r['avg_daily'],robust)-2*max(0,r['max_dd']-.20)

def main(corpus:Path,out:Path):
    eps=v.load(corpus); base=pd.DataFrame([v.feature_row(e) for e in eps])
    base['day']=pd.to_datetime(base.signal_t,unit='s',utc=True).dt.date.astype(str)
    days=sorted(base.day.unique()); i=max(1,int(len(days)*.5)); j=max(i+1,int(len(days)*.75))
    dsets={'train':set(days[:i]),'val':set(days[i:j]),'test':set(days[j:])}
    idx={k:np.flatnonzero(base.day.isin(x).to_numpy()) for k,x in dsets.items()}
    best=None; leaderboard=[]
    for pname,profile in v.PROFILES.items():
        d=base.copy(); o=[v.outcome(e,profile,v.COSTS['base'],'last') for e in eps]
        d['target']=[x[0] for x in o]; d['exit_t']=[x[1] for x in o]; d['exit_reason']=[x[2] for x in o]; d['censored']=[x[3] for x in o]
        for family in FAMILIES:
            mask=v.subset_mask(d,family); tr=idx['train'][mask[idx['train']]]; va=idx['val'][mask[idx['val']]]; te=idx['test'][mask[idx['test']]]
            if min(len(tr),len(va),len(te))<40 or len(tr)<100: continue
            Xtr=v.encode(d.loc[tr]); cols=list(Xtr.columns); Xv=v.encode(d.loc[va],cols); y=d.loc[tr,'target'].clip(-1,3)
            for spec in MODELS:
                model=make_model(spec); model.fit(Xtr,y); pred=model.predict(Xv)
                for q in QUANTILES:
                    th=float(np.quantile(pred,q))
                    for pf in [.05,.10,.15,.20]:
                        for mex in [.30,.50]:
                            r=v.portfolio(d.loc[va],pred,th,pf,mex,5,5,.10); s=metric(r)
                            rec={'score':s,'profile':pname,'family':family,'model':spec[0],'depth':spec[1],'leaf':spec[2],'quantile':q,'position_fraction':pf,'max_exposure':mex,'validation':{k:x for k,x in r.items() if k!='rows'}}
                            leaderboard.append(rec)
                            if best is None or s>best['score']: best=rec
    if best is None: raise RuntimeError('no validation candidate')
    profile=v.PROFILES[best['profile']]; d=base.copy(); o=[v.outcome(e,profile,v.COSTS['base'],'last') for e in eps]
    d['target']=[x[0] for x in o]; d['exit_t']=[x[1] for x in o]; d['exit_reason']=[x[2] for x in o]; d['censored']=[x[3] for x in o]
    mask=v.subset_mask(d,best['family']); fit=np.concatenate([idx['train'][mask[idx['train']]],idx['val'][mask[idx['val']]]]); te=idx['test'][mask[idx['test']]]
    X=v.encode(d.loc[fit]); cols=list(X.columns); Xt=v.encode(d.loc[te],cols); y=d.loc[fit,'target'].clip(-1,3)
    spec=(best['model'],best['depth'],best['leaf']); model=make_model(spec,500); model.fit(X,y); pfit=model.predict(X); pt=model.predict(Xt); threshold=float(np.quantile(pfit,best['quantile']))
    result=v.portfolio(d.loc[te],pt,threshold,best['position_fraction'],best['max_exposure'],5,5,.10)
    stress={}
    for censor in ['last','zero']:
        for cost_name,cost in v.COSTS.items():
            s=base.copy(); z=[v.outcome(e,profile,cost,censor) for e in eps]
            s['target']=[x[0] for x in z]; s['exit_t']=[x[1] for x in z]; s['exit_reason']=[x[2] for x in z]; s['censored']=[x[3] for x in z]
            rr=v.portfolio(s.loc[te],pt,threshold,best['position_fraction'],best['max_exposure'],5,5,.10)
            stress[f'{censor}_{cost_name}']={k:x for k,x in rr.items() if k!='rows'}
    gate=result['days']>=10 and result['trades']>=50 and (result['avg_daily'] or -999)>=2 and result['max_dd']<=.30 and (stress['zero_stress']['avg_daily'] or -999)>=0
    importance=[]
    if hasattr(model,'feature_importances_'):
        importance=sorted([{'feature':c,'importance':float(x)} for c,x in zip(cols,model.feature_importances_)],key=lambda x:x['importance'],reverse=True)[:25]
    report={'status':'passed' if gate else 'failed','source':{'md5':hashlib.md5(corpus.read_bytes()).hexdigest(),'episodes':len(eps),'first':datetime.fromtimestamp(eps[0].signal_t,tz=timezone.utc).isoformat(),'last':datetime.fromtimestamp(eps[-1].signal_t,tz=timezone.utc).isoformat()},'split':{'train_days':len(dsets['train']),'validation_days':len(dsets['val']),'test_days':len(dsets['test']),'train':len(idx['train']),'validation':len(idx['val']),'test':len(idx['test'])},'selected':best,'threshold_after_refit':threshold,'test':{k:x for k,x in result.items() if k!='rows'},'stress':stress,'feature_importance':importance,'gate':{'passed':gate,'target_daily':2,'minimum_days':10,'minimum_trades':50,'maximum_drawdown':.30},'method':['Complete-day chronological split','Validation-only family/model/risk selection','Test opened once','6% base friction; 3% and 10% stress matrix','Carry-forward and zero-recovery censoring bounds']}
    out.mkdir(parents=True,exist_ok=True); (out/'v4-fast-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)); pd.DataFrame(result['rows']).to_csv(out/'v4-fast-test-trades.csv',index=False); pd.DataFrame(sorted(leaderboard,key=lambda x:x['score'],reverse=True)[:100]).to_json(out/'v4-fast-leaderboard.json',orient='records',indent=2); (out/'v4-fast-summary.md').write_text(f"# V4 fast temporal optimizer\n\nStatus: **{report['status']}**\n\nTest {result['days']} days, {result['trades']} trades, net ${result['net']:.4f}, average/day ${result['avg_daily']:.4f}, max drawdown {result['max_dd']:.2%}.\n")
    print(json.dumps({'status':report['status'],'selected':best,'test':report['test'],'stress':stress},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--corpus',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    try: main(a.corpus,a.output)
    except Exception:
        import traceback; a.output.mkdir(parents=True,exist_ok=True); (a.output/'v4-fast-error.txt').write_text(traceback.format_exc()); raise
