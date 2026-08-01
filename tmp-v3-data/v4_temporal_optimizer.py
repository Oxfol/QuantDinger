#!/usr/bin/env python3
from __future__ import annotations
import argparse, gzip, json, math, hashlib, traceback
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor

INITIAL=50.0
TARGET_DAILY=2.0
SEED=20260801


def num(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except Exception: return None

def ts(x):
    if isinstance(x,(int,float)):
        v=float(x); return v/1000 if v>1e10 else v
    if not x: return None
    try: return datetime.fromisoformat(str(x).replace('Z','+00:00')).timestamp()
    except Exception: return None

def clip(v, lo, hi, default=0.0):
    if v is None or not math.isfinite(v): return default
    return max(lo,min(hi,v))

@dataclass
class Point:
    t: float
    p: float

@dataclass
class Episode:
    mint: str; symbol: str; reason: str; dex: str
    reject_t: float; signal_t: float; delay_m: float
    age: float|None; liq: float|None; vol: float|None
    pc5: float|None; pc1: float|None; pc24: float|None
    regime: dict[str,float]
    path: list[Point]

PROFILES={
 'fast': dict(h=1, sl=.12, tp1=.35, tp2=1.0, trail_act=.20, trail=.12),
 'swing': dict(h=4, sl=.16, tp1=.50, tp2=2.0, trail_act=.30, trail=.18),
 'runner': dict(h=24, sl=.20, tp1=.75, tp2=4.0, trail_act=.50, trail=.25),
}
COSTS={'optimistic':.03,'base':.06,'stress':.10}


def load(path:Path)->list[Episode]:
    tokens={}
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:r=json.loads(line)
            except Exception: continue
            mint=str(r.get('mint') or '')
            rt=ts(r.get('rejectTs') or r.get('rejectionTimestamp'))
            st=ts(r.get('sampleTs') or r.get('queriedAt'))
            p=num(r.get('priceUsd'))
            reason=str(r.get('rejectReason') or r.get('reason') or '')
            if not mint or rt is None or st is None or p is None or p<=0 or not reason or st<rt: continue
            x=tokens.setdefault(mint,dict(mint=mint,symbol=str(r.get('symbol') or 'UNKNOWN'),reason=reason,reject_t=rt,rows=[]))
            reg=r.get('regime') if isinstance(r.get('regime'),dict) else {}
            flat={}
            for k,v in reg.items():
                nv=num(v)
                if nv is not None: flat[f'reg_{k}']=nv
            x['rows'].append(dict(t=st,p=p,age=num(r.get('ageMin')),liq=num(r.get('liquidity') if r.get('liquidity') is not None else r.get('liquidityUsd')),vol=num(r.get('volume24h') if r.get('volume24h') is not None else r.get('volumeH24')),pc5=num(r.get('priceChange5m')),pc1=num(r.get('priceChange1h')),pc24=num(r.get('priceChange24h')),dex=str(r.get('dexId') or 'unknown'),reg=flat))
    out=[]
    for x in tokens.values():
        rows=sorted(x['rows'],key=lambda z:z['t'])
        e=rows[0]; delay=(e['t']-x['reject_t'])/60
        if delay<0 or delay>90: continue
        pathpts=[Point(z['t'],z['p']) for z in rows if z['t']>=e['t'] and z['t']<=e['t']+25*3600]
        if len(pathpts)<2: continue
        out.append(Episode(x['mint'],x['symbol'],x['reason'],e['dex'],x['reject_t'],e['t'],delay,e['age'],e['liq'],e['vol'],e['pc5'],e['pc1'],e['pc24'],e['reg'],pathpts))
    return sorted(out,key=lambda e:e.signal_t)


def outcome(e:Episode, profile:dict, cost:float, censor:str):
    p0=e.path[0].p
    end=e.signal_t+profile['h']*3600
    pts=[x for x in e.path if x.t<=end]
    observed=bool(pts and pts[-1].t>=e.signal_t+profile['h']*3600*.75)
    remaining=1.0; proceeds=0.0; peak=p0; tp1=False; tp2=False; exit_t=pts[-1].t; reason='horizon'
    for x in pts[1:]:
        r=x.p/p0-1
        peak=max(peak,x.p)
        if r<=-profile['sl']:
            proceeds += remaining*(1+r); remaining=0; reason='stop'; exit_t=x.t; break
        if not tp1 and r>=profile['tp1']:
            proceeds += .35*(1+r); remaining-=.35; tp1=True; reason='tp1'; exit_t=x.t
        if not tp2 and r>=profile['tp2']:
            q=min(.35,remaining); proceeds += q*(1+r); remaining-=q; tp2=True; reason='tp2'; exit_t=x.t
        if peak/p0-1>=profile['trail_act'] and x.p<=peak*(1-profile['trail']):
            proceeds += remaining*(1+r); remaining=0; reason='trail'; exit_t=x.t; break
    censored=not observed
    if remaining>1e-12:
        if censored and censor=='zero':
            remaining=0; reason='censored_zero'
        else:
            last=pts[-1].p if pts else p0
            r=last/p0-1
            proceeds += remaining*(1+r); remaining=0
            reason='censored_last' if censored else reason
            exit_t=pts[-1].t if pts else end
    net=max(-1.0,proceeds-1.0-cost)
    return net,exit_t,reason,censored


def feature_row(e:Episode)->dict[str,Any]:
    liq=e.liq or 0; vol=e.vol or 0
    row={'signal_t':e.signal_t,'mint':e.mint,'symbol':e.symbol,'reason':e.reason,'dex':e.dex,'delay_m':clip(e.delay_m,0,90),'age':clip(e.age,0,3000),'log_age':math.log1p(max(0,e.age or 0)),'log_liq':math.log1p(max(0,liq)),'log_vol':math.log1p(max(0,vol)),'log_vl':math.log1p(max(0,vol/max(1,liq))),'pc5':clip(e.pc5,-1000,1000),'pc1':clip(e.pc1,-1000,2000),'pc24':clip(e.pc24,-1000,10000),'hour_sin':math.sin(2*math.pi*datetime.fromtimestamp(e.signal_t,tz=timezone.utc).hour/24),'hour_cos':math.cos(2*math.pi*datetime.fromtimestamp(e.signal_t,tz=timezone.utc).hour/24)}
    row.update({k:clip(v,-1e6,1e6) for k,v in e.regime.items()})
    return row


def encode(df, columns=None):
    X=df.drop(columns=[c for c in ['signal_t','mint','symbol','target','exit_t','exit_reason','censored'] if c in df],errors='ignore').copy()
    X=pd.get_dummies(X,columns=[c for c in ['reason','dex'] if c in X],dummy_na=True)
    X=X.replace([np.inf,-np.inf],np.nan).fillna(0.0)
    if columns is not None: X=X.reindex(columns=columns,fill_value=0.0)
    return X


def subset_mask(df,name):
    if name=='all': return np.ones(len(df),dtype=bool)
    if name=='launch': return (df.age<=120).to_numpy()
    if name=='momentum': return ((df.pc1>=10)&(df.pc5>=-5)).to_numpy()
    if name=='pullback': return ((df.pc1>=20)&(df.pc5<=-2)&(df.pc5>=-35)).to_numpy()
    if name=='oversold': return ((df.pc1<=-15)&(df.pc5<=-5)&(df.pc24>=-70)).to_numpy()
    raise ValueError(name)


def portfolio(df,pred,threshold,pos_frac,max_exposure,max_open,max_daily_trades,daily_loss_cap):
    selected=df.loc[pred>=threshold].copy(); selected['pred']=pred[pred>=threshold]
    selected=selected.sort_values(['signal_t','pred'],ascending=[True,False])
    cash=INITIAL; active=[]; peak=INITIAL; mdd=0; trades=[]; skipped=0; day_counts=Counter(); day_start={}; daily_real=defaultdict(float)
    def equity(): return cash+sum(x['size'] for x in active)
    def release(t):
        nonlocal cash,peak,mdd
        due=sorted([x for x in active if x['exit_t']<=t],key=lambda x:x['exit_t'])
        for x in due:
            cash += x['size']*(1+x['target']); active.remove(x); daily_real[x['day']]+=x['size']*x['target']
            eq=equity(); peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak if peak else 0); x['equity_after']=eq; trades.append(x)
    for _,r in selected.iterrows():
        release(float(r.signal_t)); day=datetime.fromtimestamp(float(r.signal_t),tz=timezone.utc).date().isoformat()
        if day not in day_start: day_start[day]=equity()
        if day_counts[day]>=max_daily_trades or daily_real[day]<=-daily_loss_cap*day_start[day] or len(active)>=max_open:
            skipped+=1; continue
        eq=equity(); exposure=sum(x['size'] for x in active); room=max(0,eq*max_exposure-exposure); size=min(cash,room,eq*pos_frac)
        if size<.25: skipped+=1; continue
        cash-=size; day_counts[day]+=1
        active.append(dict(signal_t=float(r.signal_t),exit_t=float(r.exit_t),mint=r.mint,symbol=r.symbol,size=size,target=float(r.target),pred=float(r['pred']),day=day,exit_reason=r.exit_reason,censored=bool(r.censored)))
    release(float('inf'))
    if len(df):
        d0=datetime.fromtimestamp(float(df.signal_t.min()),tz=timezone.utc).date(); d1=datetime.fromtimestamp(float(df.signal_t.max()),tz=timezone.utc).date(); days=(d1-d0).days+1
    else: days=0
    net=cash-INITIAL; wins=sum(x['size']*x['target']>0 for x in trades); top3=sum(sorted([x['size']*x['target'] for x in trades if x['target']>0],reverse=True)[:3])
    return dict(final=cash,net=net,days=days,avg_daily=net/days if days else None,max_dd=mdd,trades=len(trades),win_rate=wins/len(trades) if trades else None,skipped=skipped,net_without_top3=net-top3,rows=trades)


def objective(r):
    if r['trades']<20 or r['days']<5 or r['max_dd']>.30: return -1e9
    return min(r['avg_daily'],r['net_without_top3']/max(1,r['days'])) - 2*max(0,r['max_dd']-.20)


def run(corpus:Path,out:Path):
    eps=load(corpus); base=pd.DataFrame([feature_row(e) for e in eps])
    n=len(base); a=int(n*.50); b=int(n*.75); splits={'train':np.arange(0,a),'val':np.arange(a,b),'test':np.arange(b,n)}
    model_specs=[]
    for name in ['extra','rf','hist']:
      for depth in ([3,5,8,None] if name!='hist' else [None]):
       for leaf in [5,10,20,40]: model_specs.append((name,depth,leaf))
    all_results=[]; chosen=None
    for censor in ['last','zero']:
      for prof_name,prof in PROFILES.items():
       for cost_name,cost in COSTS.items():
        df=base.copy(); outs=[outcome(e,prof,cost,censor) for e in eps]
        df['target']=[x[0] for x in outs]; df['exit_t']=[x[1] for x in outs]; df['exit_reason']=[x[2] for x in outs]; df['censored']=[x[3] for x in outs]
        for family in ['all','launch','momentum','pullback','oversold']:
         mask=subset_mask(df,family); train_idx=splits['train'][mask[splits['train']]]; val_idx=splits['val'][mask[splits['val']]]; test_idx=splits['test'][mask[splits['test']]]
         if len(train_idx)<100 or len(val_idx)<40 or len(test_idx)<40: continue
         Xtr=encode(df.loc[train_idx]); cols=list(Xtr.columns); Xv=encode(df.loc[val_idx],cols); ytr=df.loc[train_idx,'target'].clip(-1,3)
         for mn,depth,leaf in model_specs:
          if mn=='extra': model=ExtraTreesRegressor(n_estimators=200,max_depth=depth,min_samples_leaf=leaf,max_features=.8,random_state=SEED,n_jobs=-1)
          elif mn=='rf': model=RandomForestRegressor(n_estimators=200,max_depth=depth,min_samples_leaf=leaf,max_features=.8,random_state=SEED,n_jobs=-1)
          else: model=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,min_samples_leaf=leaf,l2_regularization=1.0,random_state=SEED)
          model.fit(Xtr,ytr); pv=model.predict(Xv)
          for q in [.70,.80,.85,.90,.93,.95,.97]:
           th=float(np.quantile(pv,q))
           for pf in [.05,.08,.10,.15,.20]:
            for mex in [.20,.30,.50]:
             r=portfolio(df.loc[val_idx],pv,th,pf,mex,5,5,.10); s=objective(r)
             rec=dict(score=s,censor=censor,profile=prof_name,cost=cost_name,family=family,model=mn,depth=depth,leaf=leaf,q=q,threshold=th,pos_frac=pf,max_exposure=mex,val={k:v for k,v in r.items() if k!='rows'},columns=cols)
             all_results.append(rec)
             if chosen is None or s>chosen['score']: chosen=rec|{'model_obj':model}
    if chosen is None: raise RuntimeError('no candidate')
    prof=PROFILES[chosen['profile']]; cost=COSTS[chosen['cost']]; censor=chosen['censor']; df=base.copy(); outs=[outcome(e,prof,cost,censor) for e in eps]
    df['target']=[x[0] for x in outs]; df['exit_t']=[x[1] for x in outs]; df['exit_reason']=[x[2] for x in outs]; df['censored']=[x[3] for x in outs]
    mask=subset_mask(df,chosen['family']); fit_idx=np.concatenate([splits['train'][mask[splits['train']]],splits['val'][mask[splits['val']]]]); test_idx=splits['test'][mask[splits['test']]]
    Xfit=encode(df.loc[fit_idx]); cols=list(Xfit.columns); Xt=encode(df.loc[test_idx],cols); yfit=df.loc[fit_idx,'target'].clip(-1,3)
    mn=chosen['model']; depth=chosen['depth']; leaf=chosen['leaf']
    if mn=='extra': model=ExtraTreesRegressor(n_estimators=500,max_depth=depth,min_samples_leaf=leaf,max_features=.8,random_state=SEED,n_jobs=-1)
    elif mn=='rf': model=RandomForestRegressor(n_estimators=500,max_depth=depth,min_samples_leaf=leaf,max_features=.8,random_state=SEED,n_jobs=-1)
    else: model=HistGradientBoostingRegressor(max_iter=250,max_leaf_nodes=15,min_samples_leaf=leaf,l2_regularization=1.0,random_state=SEED)
    model.fit(Xfit,yfit); pfit=model.predict(Xfit); pt=model.predict(Xt); th=float(np.quantile(pfit,chosen['q']))
    test_res=portfolio(df.loc[test_idx],pt,th,chosen['pos_frac'],chosen['max_exposure'],5,5,.10)
    stress={}
    for cz in ['last','zero']:
      for cn,cst in COSTS.items():
        d2=base.copy(); oo=[outcome(e,prof,cst,cz) for e in eps]; d2['target']=[x[0] for x in oo]; d2['exit_t']=[x[1] for x in oo]; d2['exit_reason']=[x[2] for x in oo]; d2['censored']=[x[3] for x in oo]
        rr=portfolio(d2.loc[test_idx],pt,th,chosen['pos_frac'],chosen['max_exposure'],5,5,.10); stress[f'{cz}_{cn}']={k:v for k,v in rr.items() if k!='rows'}
    gate=test_res['days']>=10 and test_res['trades']>=50 and (test_res['avg_daily'] or -999)>=TARGET_DAILY and test_res['max_dd']<=.30 and stress['zero_stress']['avg_daily']>=0
    feature_importance=[]
    if hasattr(model,'feature_importances_'):
      feature_importance=sorted([{'feature':c,'importance':float(v)} for c,v in zip(cols,model.feature_importances_)],key=lambda x:x['importance'],reverse=True)[:25]
    report=dict(status='passed' if gate else 'failed',source=dict(md5=hashlib.md5(corpus.read_bytes()).hexdigest(),episodes=len(eps),first=datetime.fromtimestamp(eps[0].signal_t,tz=timezone.utc).isoformat(),last=datetime.fromtimestamp(eps[-1].signal_t,tz=timezone.utc).isoformat()),split=dict(train=a,val=b-a,test=n-b),selected={k:v for k,v in chosen.items() if k not in ['model_obj','columns']},test={k:v for k,v in test_res.items() if k!='rows'},stress=stress,gate=dict(passed=gate,target_daily=TARGET_DAILY,min_days=10,min_trades=50,max_dd=.30),feature_importance=feature_importance,notes=['Test interval opened once after validation selection.','External corpus is rejected Solana candidates, not DBotX accepted universe.','Stress case requires zero-recovery censoring and 10% all-in round-trip friction to remain non-negative.'])
    out.mkdir(parents=True,exist_ok=True); (out/'v4-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    pd.DataFrame(test_res['rows']).to_csv(out/'v4-test-trades.csv',index=False)
    pd.DataFrame(sorted(all_results,key=lambda x:x['score'],reverse=True)[:100]).drop(columns=['columns'],errors='ignore').to_json(out/'v4-top-validation.json',orient='records',indent=2)
    (out/'v4-summary.md').write_text(f"# V4 temporal optimizer\n\nStatus: **{report['status']}**\n\nTest: {test_res['days']} days, {test_res['trades']} trades, net ${test_res['net']:.4f}, average/day ${test_res['avg_daily']:.4f}, max drawdown {test_res['max_dd']:.2%}.\n\nSelected: `{json.dumps(report['selected'],ensure_ascii=False)}`\n")
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--corpus',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    try:
      rep=run(args.corpus,args.output); print(json.dumps({'status':rep['status'],'test':rep['test'],'selected':rep['selected']},indent=2))
    except Exception:
      args.output.mkdir(parents=True,exist_ok=True); (args.output/'error.txt').write_text(traceback.format_exc()); raise
