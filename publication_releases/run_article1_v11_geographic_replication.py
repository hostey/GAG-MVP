#!/usr/bin/env python3
"""Article 1 v11: geographic Monte Carlo replication benchmark.

Design:
- Train the same UCI Cleveland models for repeated model seeds.
- For each referral-eligible clinical test case (Y=1), draw K primary-facility
  origins uniformly with replacement from all functional GRID3 primary facilities.
- Reuse exactly the same geographic draws across model variants within a seed.
- Compute optimistic geodesic accessibility proxy for tau in {30,45,60,90}.
- Summarize metrics at seed level; bootstrap seed-level estimates, not replicated rows.
- Run S0-S3 counterfactuals at 30 and 60 minutes.

The K geographic realizations are Monte Carlo location replicates, not independent
patients and not population-weighted Nigerian observations.
"""
from __future__ import annotations
import argparse, ast, hashlib, json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODULE = Path(__file__).resolve().parent / '02_Healthcare_Equity_Article1_v10.py'

class DummyStreamlit:
    @staticmethod
    def cache_data(*args, **kwargs):
        def deco(fn): return fn
        return deco


def load_core():
    tree=ast.parse(MODULE.read_text(encoding='utf-8'))
    wanted={'ModelArtifact','Article1ReferralConfig','_normalize_grid3_for_article1','_haversine_vector_km','_article1_origin_pool','_precompute_origin_access','_train_and_score'}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted]
    mod=ast.Module(body=nodes,type_ignores=[]); ast.fix_missing_locations(mod)
    ns=dict(np=np,pd=pd,dataclass=dataclass,RandomForestClassifier=RandomForestClassifier,
            accuracy_score=accuracy_score,recall_score=recall_score,precision_score=precision_score,
            f1_score=f1_score,confusion_matrix=confusion_matrix,roc_auc_score=roc_auc_score,
            train_test_split=train_test_split,StandardScaler=StandardScaler,st=DummyStreamlit())
    exec(compile(mod,str(MODULE),'exec'),ns)
    return ns


def load_uci(path: Path):
    cols=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
    df=pd.read_csv(path,names=cols,na_values='?').dropna().copy()
    df['target']=(df['target']>0).astype(int)
    df['demographic_group']=df['sex'].astype(int)
    features=[c for c in cols if c!='target']
    return df,features,df[features].to_numpy(float),df['target'].to_numpy(int),df['demographic_group'].to_numpy(int)


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def seed_bootstrap(df, value_cols, group_cols, n_boot=1000, rng_seed=20260913):
    """Bootstrap whole seed estimates. Replicated geography never enters as iid rows."""
    rng=np.random.default_rng(rng_seed)
    out=[]
    for keys,g in df.groupby(group_cols,dropna=False):
        if not isinstance(keys,tuple): keys=(keys,)
        seeds=np.array(sorted(g.seed.unique()))
        for val in value_cols:
            vals=[]
            for _ in range(n_boot):
                pick=rng.choice(seeds,size=len(seeds),replace=True)
                # preserve duplicated sampled seeds as bootstrap multiplicities
                by_seed=g.set_index('seed')[val]
                vals.append(float(np.nanmean([by_seed.loc[s] for s in pick])))
            row=dict(zip(group_cols,keys)); row.update(metric=val,mean=float(g[val].mean()),ci_low=float(np.nanpercentile(vals,2.5)),ci_high=float(np.nanpercentile(vals,97.5)),n_seeds=int(len(seeds)))
            out.append(row)
    return pd.DataFrame(out)


def make_geo_draws(access: pd.DataFrame, n_cases: int, k: int, seed: int):
    # Uniform over facilities, with replacement; deterministic and common across variants.
    rng=np.random.default_rng(seed+88001)
    idx=rng.integers(0,len(access),size=(n_cases,k))
    flat=access.iloc[idx.ravel()].reset_index(drop=True).copy()
    flat['case_index']=np.repeat(np.arange(n_cases),k)
    flat['geo_rep']=np.tile(np.arange(k),n_cases)
    return flat


def replicated_records(artifact, geo, speed, variant, seed):
    y=np.asarray(artifact.y_test,int); pred=np.asarray(artifact.y_prediction,int); prob=np.asarray(artifact.y_probability,float); grp=np.asarray(artifact.demo_test,int)
    pos=np.flatnonzero(y==1)
    if len(pos)==0: return pd.DataFrame()
    if geo['case_index'].max()+1 != len(pos): raise ValueError('geo case count mismatch')
    clinical=pd.DataFrame({'case_index':np.arange(len(pos)),'test_index':pos,'clinical_group':grp[pos],
                           'clinical_target_y':y[pos],'predicted_positive':pred[pos],'prediction_probability':prob[pos],
                           'missed_referral_m':(pred[pos]==0).astype(int)})
    rec=geo.merge(clinical,on='case_index',how='left',validate='many_to_one')
    rec['seed']=seed; rec['model_variant']=variant
    rec['travel_time_geo_min']=rec['straight_line_km']/speed*60.0
    return rec


def summarize_thresholds(rec, thresholds):
    rows=[]
    for tau in thresholds:
        d=rec.copy(); d['C']=(d.travel_time_geo_min>tau).astype(int); d['J']=d.missed_referral_m*d.C
        for gval in list(sorted(d.clinical_group.unique()))+['ALL']:
            g=d if gval=='ALL' else d[d.clinical_group==gval]
            # clinical FNR must not be weighted by number of geo replicas; dedupe case.
            cases=g.drop_duplicates('case_index')
            fnr=float(cases.missed_referral_m.mean())
            c=float(g.C.mean()); j=float(g.J.mean()); jind=fnr*c; e=j-jind
            c1=g[g.C==1]; c0=g[g.C==0]
            dep=float(c1.missed_referral_m.mean()-c0.missed_referral_m.mean()) if len(c1) and len(c0) else np.nan
            rows.append({'seed':int(d.seed.iloc[0]),'model_variant':d.model_variant.iloc[0],'threshold_min':float(tau),'group':str(gval),
                         'n_clinical_R1':int(cases.case_index.nunique()),'n_geo_realizations':int(len(g)),
                         'n_unique_origins':int(g.origin_facility_id.nunique()),'fnr':fnr,'C':c,'J':j,'J_ind':jind,'E':e,'D':dep,
                         'joint_event_count':int(g.J.sum()),'constrained_event_count':int(g.C.sum())})
    return rows


def s0_s3(rec, thresholds=(30.0,60.0), travel_reduction=.25):
    """Baseline-model technical counterfactuals. S1 closes group FNR gap at clinical-case level."""
    out=[]
    base_cases=rec.drop_duplicates('case_index')[['case_index','clinical_group','missed_referral_m','prediction_probability']].copy()
    groups=sorted(base_cases.clinical_group.unique())
    repaired=set()
    if len(groups)>=2:
        stats={g:base_cases.loc[base_cases.clinical_group==g,'missed_referral_m'].mean() for g in groups}
        hi=max(stats,key=stats.get); lo=min(stats,key=stats.get)
        hi_cases=base_cases[base_cases.clinical_group==hi]; n_hi=len(hi_cases)
        target=stats[lo]
        current_fn=int(hi_cases.missed_referral_m.sum())
        target_fn=int(round(target*n_hi)); n_repair=max(0,current_fn-target_fn)
        cand=hi_cases[hi_cases.missed_referral_m==1].sort_values('prediction_probability',ascending=False)
        repaired=set(cand.head(n_repair).case_index.astype(int).tolist())
    for tau in thresholds:
        for scenario in ['S0','S1','S2','S3']:
            d=rec.copy()
            m=d.missed_referral_m.copy()
            t=d.travel_time_geo_min.copy()
            if scenario in {'S1','S3'} and repaired:
                m=np.where(d.case_index.isin(repaired),0,m)
            if scenario in {'S2','S3'}: t=t*(1.0-travel_reduction)
            c=(t>tau).astype(int); j=m*c
            # FNR from unique clinical cases after repair
            tmp=d[['case_index','clinical_group']].copy(); tmp['m']=m
            fnr=float(tmp.drop_duplicates('case_index').m.mean())
            out.append({'seed':int(d.seed.iloc[0]),'threshold_min':tau,'scenario':scenario,'fnr':fnr,'C':float(c.mean()),'J':float(j.mean()),
                        'joint_event_count':int(j.sum()),'n_geo_realizations':int(len(d)),'n_repaired_cases':len(repaired)})
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--uci',required=True); ap.add_argument('--grid3',required=True)
    ap.add_argument('--runs',type=int,default=30); ap.add_argument('--geo-reps',type=int,default=100)
    ap.add_argument('--bootstrap',type=int,default=1000); ap.add_argument('--seed-base',type=int,default=42)
    ap.add_argument('--speed',type=float,default=40.0); ap.add_argument('--outdir',default='/mnt/data/article1_v11_geographic_replication')
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    ns=load_core(); uci=Path(args.uci); grid=Path(args.grid3)
    df,features,X,y,demo=load_uci(uci)
    facilities=pd.read_excel(grid)
    cfg=ns['Article1ReferralConfig'](enabled=True,routing_method='geodesic_proxy',assumed_speed_kmh=args.speed)
    norm=ns['_normalize_grid3_for_article1'](facilities); origins,eligible=ns['_article1_origin_pool'](norm,cfg)
    access=ns['_precompute_origin_access'](origins,eligible)
    access.to_pickle(out/'origin_to_nearest_higher_level.pkl')

    variants=['none','reweighing','post_processing']; thresholds=[30.0,45.0,60.0,90.0]
    all_seed=[]; all_s=[]; origin_audit=[]
    for r in range(args.runs):
        seed=args.seed_base+r
        arts={}; mets={}
        for variant in variants:
            mets[variant],arts[variant]=ns['_train_and_score'](X,y,demo,feature_names=features,random_state=seed,mitigation_strategy=variant,threshold=.5)
        npos=int(np.sum(np.asarray(arts['none'].y_test)==1))
        geo=make_geo_draws(access,npos,args.geo_reps,seed)
        origin_audit.append({'seed':seed,'n_R1':npos,'geo_reps_per_case':args.geo_reps,'n_case_location_realizations':len(geo),
                             'n_unique_origins':geo.origin_facility_id.nunique(),'uniform_facility_sampling':True})
        for variant in variants:
            rec=replicated_records(arts[variant],geo,args.speed,variant,seed)
            all_seed.extend(summarize_thresholds(rec,thresholds))
            if variant=='none': all_s.extend(s0_s3(rec,(30.0,60.0),.25))
        print(f'completed seed {seed} ({r+1}/{args.runs}); R=1={npos}; realizations={len(geo)}')

    seed_df=pd.DataFrame(all_seed); s_df=pd.DataFrame(all_s); audit=pd.DataFrame(origin_audit)
    seed_df.to_csv(out/'v11_seed_level_threshold_metrics.csv',index=False); s_df.to_csv(out/'v11_seed_level_s0_s3.csv',index=False); audit.to_csv(out/'v11_origin_sampling_audit.csv',index=False)
    ci=seed_bootstrap(seed_df,['fnr','C','J','J_ind','E'],['model_variant','threshold_min','group'],args.bootstrap)
    ci.to_csv(out/'v11_threshold_bootstrap_ci.csv',index=False)
    sci=seed_bootstrap(s_df,['fnr','C','J'],['threshold_min','scenario'],args.bootstrap)
    sci.to_csv(out/'v11_s0_s3_bootstrap_ci.csv',index=False)

    # Delta J / FNR at seed-level between clinical groups 0-1.
    sub=seed_df[seed_df.group.isin(['0','1'])].copy()
    wide=sub.pivot_table(index=['seed','model_variant','threshold_min'],columns='group',values=['fnr','C','J','E']).reset_index()
    wide.columns=['_'.join([str(x) for x in c if str(x)!='']) if isinstance(c,tuple) else c for c in wide.columns]
    delta=pd.DataFrame({'seed':wide.seed,'model_variant':wide.model_variant,'threshold_min':wide.threshold_min,
                        'delta_fnr_0_minus_1':wide['fnr_0']-wide['fnr_1'],'delta_C_0_minus_1':wide['C_0']-wide['C_1'],
                        'delta_J_0_minus_1':wide['J_0']-wide['J_1'],'delta_E_0_minus_1':wide['E_0']-wide['E_1']})
    delta.to_csv(out/'v11_seed_level_disparities.csv',index=False)
    dci=seed_bootstrap(delta,['delta_fnr_0_minus_1','delta_C_0_minus_1','delta_J_0_minus_1','delta_E_0_minus_1'],['model_variant','threshold_min'],args.bootstrap)
    dci.to_csv(out/'v11_disparity_bootstrap_ci.csv',index=False)

    meta={'design_version':'v11-geographic-replication','uci_file':str(uci),'uci_sha256':sha256(uci),'grid3_file':str(grid),'grid3_sha256':sha256(grid),
          'uci_complete_rows':len(df),'functional_primary_origins':len(origins),'eligible_higher_level':len(eligible),'model_seeds':args.runs,
          'seed_base':args.seed_base,'geo_reps_per_R1_case':args.geo_reps,'origin_sampling':'uniform with replacement over all functional primary facilities; not population-weighted',
          'geodesic_proxy_speed_kmh':args.speed,'thresholds_min':thresholds,'bootstrap':'seed-level bootstrap; geographic replicas are not iid clinical observations',
          'network_layer_status':'not supplied; paired road/friction sensitivity pending exact GRID3 layer','clinical_features':features}
    (out/'v11_metadata.json').write_text(json.dumps(meta,indent=2))
    print('\nPrimary threshold summary (baseline, ALL):')
    print(ci[(ci.model_variant=='none')&(ci.group=='ALL')&ci.metric.isin(['C','J'])].to_string(index=False))
    print('\nS0-S3:')
    print(sci[sci.metric=='J'].to_string(index=False))
    print('\nDisparity:')
    print(dci[(dci.model_variant=='none')&(dci.metric=='delta_J_0_minus_1')].to_string(index=False))
    print(f'\nWrote {out}')

if __name__=='__main__': main()
