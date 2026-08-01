#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib, statistics
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
import v4_temporal_optimizer as v

SPECS=[('extra',5,10),('extra',None,20),('hist',None,20)]
FAMILIES=['all','launch','momentum','pullback','oversold']
Q=[.50,.60,.70,.80,.90]
TOPK=[1,2,3]
POS=[.05,.08,.10,.15]
EXPOSURE=[.30,.50]
CAP_RETURN=4.0

def encode(df, cols=None): return v.encode(df.drop(columns=['day'],errors='ignore'),cols)

def model_for(spec,seed,large=False):
    name,depth,leaf=spec
    if name=='extra': return ExtraTreesRegressor(n_estimators=500 if large else 180,max_depth=depth,min_samples_leaf=leaf,max_features=.8,random_state=seed,n_jobs=-1)
    return HistGradientBoostingRegressor(max_iter=260 if large else 140,max_leaf_nodes=15,min_samples_leaf=leaf,l2_regularization=1.5,random_state=seed)

def daily_top_predictions(df,pred,threshold,topk):
    out=np.full(len(pred),-1e18,dtype=float)
    work=pd.DataFrame({'day':df.day.to_numpy(),'p':pred,'i':np.arange(len(pred))})
    for _,g in work.groupby('day',sort=False):
        g=g[g.p>=threshold].nlargest(topk,'p')
        out[g.i.to_numpy()]=g.p.to_numpy()
    return out

def robust(r):
    if r['days']<4 or r['trades']<5 or r['max_dd']>.30: return -1e9
    core=r['net_without_top3']/max(1,r['days'])
    return .55*r['avg_daily']+.45*core-2*max(0,r['max_dd']-.20)

def build(base,eps,profile,cost=.06,censor='last'):
    d=base.copy(); o=[v.outcome(e,profile,cost,censor) for e in eps]
    d['target']=[max(-1,min(CAP_RETURN,x[0])) for x in o]; d['exit_t']=[x[1] for x in o]; d['exit_reason']=[x[2] for x in o]; d['censored']=[x[3] for x in o]
    return d

def folds(days):
    n=len(days); test_start=max(1,int(n*.75)); pre=days[:test_start]
    cuts=[int(len(pre)*.40),int(len(pre)*.60),int(len(pre)*.80),len(pre)]
    out=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        tr=set(pre[:a]); va=set(pre[a:b])
        if len(tr)>=8 and len(va)>=4: out.append((tr,va))
    return out,set(days[test_start:]),set(pre)

def evaluate(corpus,out):
    eps=v.load(corpus); base=pd.DataFrame([v.feature_row(e) for e in eps]); base['day']=pd.to_datetime(base.signal_t,unit='s',utc=True).dt.date.astype(str)
    days=sorted(base.day.unique()); wf,test_days,fit_days=folds(days)
    candidates=[]
    for pname,profile in v.PROFILES.items():
        d=build(base,eps,profile)
        for family in FAMILIES:
            family_mask=v.subset_mask(d,family)
            for spec in SPECS:
                fold_cache=[]; valid=True
                for fi,(trdays,vadays) in enumerate(wf):
                    tr=np.flatnonzero(d.day.isin(trdays).to_numpy()&family_mask); va=np.flatnonzero(d.day.isin(vadays).to_numpy()&family_mask)
                    if len(tr)<100 or len(va)<30: valid=False; break
                    X=encode(d.loc[tr]); cols=list(X.columns); Xv=encode(d.loc[va],cols); y=d.loc[tr,'target'].clip(-1,3)
                    m=model_for(spec,v.SEED+fi); m.fit(X,y); ptr=m.predict(X); pv=m.predict(Xv)
                    fold_cache.append((va,pv,ptr))
                if not valid: continue
                for q in Q:
                    for topk in TOPK:
                        for pos in POS:
                            for exp in EXPOSURE:
                                fr=[]
                                for va,pv,ptr in fold_cache:
                                    th=float(np.quantile(ptr,q)); ranked=daily_top_predictions(d.loc[va],pv,th,topk); r=v.portfolio(d.loc[va],ranked,th,pos,exp,5,topk,.10); fr.append(r)
                                scores=[robust(r) for r in fr]; positive=sum((r['avg_daily'] or -99)>0 for r in fr); trades=sum(r['trades'] for r in fr)
                                score=statistics.median(scores)+.35*min(scores) if trades>=30 and positive>=2 else -1e9
                                candidates.append({'score':score,'profile':pname,'family':family,'model':spec[0],'depth':spec[1],'leaf':spec[2],'quantile':q,'topk':topk,'position_fraction':pos,'max_exposure':exp,'folds':[{k:x for k,x in r.items() if k!='rows'} for r in fr]})
    candidates.sort(key=lambda x:x['score'],reverse=True); best=candidates[0]
    profile=v.PROFILES[best['profile']]; d=build(base,eps,profile); fm=v.subset_mask(d,best['family']); fit=np.flatnonzero(d.day.isin(fit_days).to_numpy()&fm); te=np.flatnonzero(d.day.isin(test_days).to_numpy()&fm)
    X=encode(d.loc[fit]); cols=list(X.columns); Xt=encode(d.loc[te],cols); y=d.loc[fit,'target'].clip(-1,3); spec=(best['model'],best['depth'],best['leaf']); m=model_for(spec,v.SEED,True); m.fit(X,y); pfit=m.predict(X); pt=m.predict(Xt); th=float(np.quantile(pfit,best['quantile'])); rank=daily_top_predictions(d.loc[te],pt,th,best['topk']); result=v.portfolio(d.loc[te],rank,th,best['position_fraction'],best['max_exposure'],5,best['topk'],.10)
    stress={}
    for censor in ['last','zero']:
      for cname,cost in v.COSTS.items():
        sd=build(base,eps,profile,cost,censor); rr=v.portfolio(sd.loc[te],rank,th,best['position_fraction'],best['max_exposure'],5,best['topk'],.10); stress[f'{censor}_{cname}']={k:x for k,x in rr.items() if k!='rows'}
    gate=result['days']>=10 and result['trades']>=50 and (result['avg_daily'] or -999)>=2 and result['max_dd']<=.30 and (stress['zero_stress']['avg_daily'] or -999)>=0
    imp=[]
    if hasattr(m,'feature_importances_'): imp=sorted([{'feature':c,'importance':float(x)} for c,x in zip(cols,m.feature_importances_)],key=lambda x:x['importance'],reverse=True)[:25]
    rep={'status':'passed' if gate else 'failed','source':{'md5':hashlib.md5(corpus.read_bytes()).hexdigest(),'episodes':len(eps),'first':datetime.fromtimestamp(eps[0].signal_t,tz=timezone.utc).isoformat(),'last':datetime.fromtimestamp(eps[-1].signal_t,tz=timezone.utc).isoformat()},'days':{'all':len(days),'walk_forward_folds':len(wf),'fit':len(fit_days),'test':len(test_days)},'selected':best,'threshold':th,'test':{k:x for k,x in result.items() if k!='rows'},'stress':stress,'feature_importance':imp,'gate':{'passed':gate,'target_daily':2,'min_days':10,'min_trades':50,'max_dd':.30},'method':['Three expanding pre-test walk-forward folds','Daily top-K ranking','Return capped at 400% for robustness','6% base friction and 3%/10% stress','Test opened once']}
    out.mkdir(parents=True,exist_ok=True); (out/'v5-report.json').write_text(json.dumps(rep,indent=2,ensure_ascii=False)); pd.DataFrame(result['rows']).to_csv(out/'v5-test-trades.csv',index=False); pd.DataFrame(candidates[:100]).to_json(out/'v5-leaderboard.json',orient='records',indent=2); (out/'v5-summary.md').write_text(f"# V5 walk-forward optimizer\n\nStatus **{rep['status']}**. Test {result['days']} calendar days, {result['trades']} trades, net ${result['net']:.4f}, avg/day ${result['avg_daily']:.4f}, max DD {result['max_dd']:.2%}.\n")
    print(json.dumps({'status':rep['status'],'selected':best,'test':rep['test'],'stress':stress},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--corpus',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); evaluate(a.corpus,a.output)
