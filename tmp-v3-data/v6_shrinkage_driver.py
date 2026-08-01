#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib, math, statistics
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import v4_temporal_optimizer as v

TOPK=[2,3,4,5]
POS=[.05,.08,.10,.12,.15]
EXPOSURE=[.30,.50]
PRIORS=[5,10,20,40,80]
CAP=4.0

def bin_label(x, cuts):
    try: z=float(x)
    except Exception: z=0.0
    for i,c in enumerate(cuts):
        if z<c: return str(i)
    return str(len(cuts))

def enrich(df):
    x=df.copy()
    x['age_bin']=[bin_label(z,[10,30,60,120,240,720]) for z in x.age]
    x['pc5_bin']=[bin_label(z,[-20,-5,0,5,20,50]) for z in x.pc5]
    x['pc1_bin']=[bin_label(z,[-50,-20,0,20,50,100,250]) for z in x.pc1]
    x['pc24_bin']=[bin_label(z,[-70,-30,0,50,150,500]) for z in x.pc24]
    x['liq_bin']=[bin_label(math.expm1(max(0,z)),[5000,10000,20000,50000,100000,250000]) for z in x.log_liq]
    x['vl_bin']=[bin_label(math.expm1(max(0,z)),[2,5,10,20,50,100]) for z in x.log_vl]
    x['hour_bin']=pd.to_datetime(x.signal_t,unit='s',utc=True).dt.hour.astype(str)
    x['reason_pc1']=x.reason.astype(str)+'|'+x.pc1_bin
    x['reason_age']=x.reason.astype(str)+'|'+x.age_bin
    x['pc1_pc5']=x.pc1_bin+'|'+x.pc5_bin
    return x

FACTORS=['reason','dex','age_bin','pc5_bin','pc1_bin','pc24_bin','liq_bin','vl_bin','hour_bin','reason_pc1','reason_age','pc1_pc5']

def fit_scores(train, apply, prior):
    global_mean=float(train.target.mean())
    score=np.full(len(apply),global_mean)
    weight=np.ones(len(apply))
    for col in FACTORS:
        stats=train.groupby(col).target.agg(['sum','count'])
        mapping=((stats['sum']+prior*global_mean)/(stats['count']+prior)).to_dict()
        vals=apply[col].map(mapping).fillna(global_mean).to_numpy(float)
        w=.5 if col in {'reason_pc1','reason_age','pc1_pc5'} else 1.0
        score += w*vals; weight += w
    return score/weight

def daily_rank(df, scores, topk, q):
    threshold=float(np.quantile(scores,q))
    out=np.full(len(scores),-1e18)
    work=pd.DataFrame({'day':df.day.to_numpy(),'s':scores,'i':np.arange(len(scores))})
    for _,g in work.groupby('day',sort=False):
        g=g[g.s>=threshold].nlargest(topk,'s')
        out[g.i.to_numpy()]=g.s.to_numpy()
    return out,threshold

def robust(r):
    if r['days']<4 or r['trades']<6 or r['max_dd']>.35: return -1e9
    return .45*r['avg_daily']+.55*(r['net_without_top3']/max(1,r['days']))-2*max(0,r['max_dd']-.20)

def build(base, eps, profile, cost=.06, censor='last'):
    d=base.copy(); oo=[v.outcome(e,profile,cost,censor) for e in eps]
    d['target']=[max(-1,min(CAP,z[0])) for z in oo]; d['exit_t']=[z[1] for z in oo]; d['exit_reason']=[z[2] for z in oo]; d['censored']=[z[3] for z in oo]
    return enrich(d)

def folds(days):
    test_start=max(1,int(len(days)*.75)); pre=days[:test_start]
    cuts=[int(len(pre)*.35),int(len(pre)*.55),int(len(pre)*.75),len(pre)]
    fs=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        tr=set(pre[:a]); va=set(pre[a:b])
        if len(tr)>=8 and len(va)>=4: fs.append((tr,va))
    return fs,set(pre),set(days[test_start:])

def run(corpus,out):
    eps=v.load(corpus); base=pd.DataFrame([v.feature_row(e) for e in eps]); base['day']=pd.to_datetime(base.signal_t,unit='s',utc=True).dt.date.astype(str)
    days=sorted(base.day.unique()); fs,fit_days,test_days=folds(days)
    board=[]
    for pname,profile in v.PROFILES.items():
        d=build(base,eps,profile)
        for prior in PRIORS:
            caches=[]
            for trd,vad in fs:
                tr=d[d.day.isin(trd)]; va=d[d.day.isin(vad)]
                if len(tr)<100 or len(va)<30: caches=[]; break
                caches.append((va,fit_scores(tr,va,prior)))
            if not caches: continue
            for q in [.50,.60,.70,.80,.90]:
                for topk in TOPK:
                    for pos in POS:
                        for exposure in EXPOSURE:
                            rs=[]
                            for va,s in caches:
                                rank,th=daily_rank(va,s,topk,q)
                                rs.append(v.portfolio(va,rank,th,pos,exposure,5,topk,.10))
                            positive=sum((r['avg_daily'] or -99)>0 for r in rs); trades=sum(r['trades'] for r in rs)
                            score=statistics.median([robust(r) for r in rs])+.35*min(robust(r) for r in rs) if trades>=35 and positive>=2 else -1e9
                            board.append({'score':score,'profile':pname,'prior':prior,'quantile':q,'topk':topk,'position_fraction':pos,'max_exposure':exposure,'folds':[{k:z for k,z in r.items() if k!='rows'} for r in rs]})
    board.sort(key=lambda x:x['score'],reverse=True); best=board[0]
    profile=v.PROFILES[best['profile']]; d=build(base,eps,profile); fit=d[d.day.isin(fit_days)]; test=d[d.day.isin(test_days)]; s=fit_scores(fit,test,best['prior']); rank,th=daily_rank(test,s,best['topk'],best['quantile']); result=v.portfolio(test,rank,th,best['position_fraction'],best['max_exposure'],5,best['topk'],.10)
    stress={}
    for censor in ['last','zero']:
        for cname,cost in v.COSTS.items():
            sd=build(base,eps,profile,cost,censor); st=sd[sd.day.isin(test_days)]; rr=v.portfolio(st,rank,th,best['position_fraction'],best['max_exposure'],5,best['topk'],.10); stress[f'{censor}_{cname}']={k:z for k,z in rr.items() if k!='rows'}
    gate=result['days']>=10 and result['trades']>=50 and (result['avg_daily'] or -999)>=2 and result['max_dd']<=.30 and (stress['zero_stress']['avg_daily'] or -999)>=0
    rep={'status':'passed' if gate else 'failed','source':{'md5':hashlib.md5(corpus.read_bytes()).hexdigest(),'episodes':len(eps),'first':datetime.fromtimestamp(eps[0].signal_t,tz=timezone.utc).isoformat(),'last':datetime.fromtimestamp(eps[-1].signal_t,tz=timezone.utc).isoformat()},'days':{'all':len(days),'folds':len(fs),'fit':len(fit_days),'test':len(test_days)},'selected':best,'threshold':th,'test':{k:z for k,z in result.items() if k!='rows'},'stress':stress,'gate':{'passed':gate,'target_daily':2,'min_days':10,'min_trades':50,'max_dd':.30},'method':['Hierarchical shrinkage factor model','Three expanding chronological validation folds','Return capped at 400%','Daily top-K selection','Test opened once']}
    out.mkdir(parents=True,exist_ok=True); (out/'v6-report.json').write_text(json.dumps(rep,indent=2,ensure_ascii=False)); pd.DataFrame(result['rows']).to_csv(out/'v6-test-trades.csv',index=False); pd.DataFrame(board[:100]).to_json(out/'v6-leaderboard.json',orient='records',indent=2); (out/'v6-summary.md').write_text(f"# V6 shrinkage strategy\n\nStatus **{rep['status']}**. Test {result['days']} days, {result['trades']} trades, net ${result['net']:.4f}, avg/day ${result['avg_daily']:.4f}, max DD {result['max_dd']:.2%}.\n")
    print(json.dumps({'status':rep['status'],'selected':best,'test':rep['test'],'stress':stress},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--corpus',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); run(a.corpus,a.output)
