"""
coper_vr.data.gen_param
=======================
Parametrized re-generation for the A3 generator-dependence audit. Defaults
reproduce the original XR-PERIL-Multi generator EXACTLY (verified in the
audit runner). Exposes: peer-contribution coefficients, severity thresholds,
noise, attacker temporal profile, attack-intensity scale, and a
channel--label DECOUPLE control. Corpora are written with the same schema as
data.xrperil_multi so the unchanged feature pipeline (multi_sequences) builds
features -- no new feature code, hence no new leakage surface.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np, pandas as pd

from coper_vr.config import PEER_CHANNELS
from coper_vr.data.xrperil_multi import (
    _sim_user_track, _sim_peer_state, _sample_avatar_pose, _fov_overlap,
    _severity_from_score, _USER_CHANNELS)

_PEER_NAMES = tuple(c.name for c in PEER_CHANNELS)

@dataclass(frozen=True)
class GenParams:
    coeffs: Tuple[float,float,float,float,float] = (0.35,0.20,0.15,0.20,0.10)  # ang,tele,part,flick,audio
    thresholds: Tuple[float,float,float] = (0.15,0.30,0.55)
    noise_sigma: float = 0.05
    temporal_profile: str = "ramp"   # ramp|pulse|burst|intermittent|constant
    intensity_scale: float = 1.0     # multiplies the attacker channel target
    decouple: bool = False           # shuffle peer channels vs. label (control)
    seed: int = 0
    # v2 fix (Oct 2026): the original generator scored an OUT-of-FOV peer's audio with
    # weight 1.0 (audio/100/d) but IN-FOV audio with coeffs[4]=0.10 x FOV. Setting
    # audio_oof_weight=None uses coeffs[4] in both cases. peer_gain scales every peer
    # contribution (calibrated label-free so unsafe prevalence stays ~1.4%).
    audio_oof_weight: float | None = 1.0   # 1.0 = original (v1) behaviour
    peer_gain: float = 1.0

def _attacker_profile(rng, T, ch, params: GenParams):
    """Attacker channel time series for the chosen temporal profile."""
    c = next(x for x in PEER_CHANNELS if x.name==ch)
    hi = c.hi*params.intensity_scale; lo=c.benign_hi
    prof = params.temporal_profile
    if prof=="ramp":
        y=np.linspace(lo,hi,T)*rng.uniform(0.85,1.0,T)
    elif prof=="constant":
        y=np.full(T,0.9*hi)
    elif prof=="pulse":
        y=np.full(T,lo); s=T//2; y[s:s+max(2,T//8)]=hi
    elif prof=="burst":
        y=np.full(T,lo)
        for st in range(0,T,max(4,T//5)): y[st:st+2]=hi
    elif prof=="intermittent":
        y=np.where(rng.random(T)<0.3,hi,lo)
    else:
        y=np.linspace(lo,hi,T)
    return np.clip(y,c.lo,c.hi).astype(np.float32)

def _peer_state_param(rng,T,attacker,ch,params):
    cols={}
    for c in PEER_CHANNELS:
        cols[c.name]=rng.uniform(c.benign_lo,c.benign_hi,T)
    if attacker and ch:
        cols[ch]=_attacker_profile(rng,T,ch,params)
    return pd.DataFrame(cols)

def _peer_contrib_param(peer_df,pos_i,ori_i,fov_i,pos_j,coeffs,audio_oof_weight=1.0,gain=1.0):
    T=len(peer_df); d=float(np.linalg.norm(np.array(pos_j)-np.array(pos_i)))+1e-3
    ov=_fov_overlap(pos_i,ori_i,pos_j,fov_i); dec=gain/d
    if ov==0.0:
        w=coeffs[4] if audio_oof_weight is None else audio_oof_weight
        return (w*peer_df["spatial_audio_jit"].values/100.0)*dec
    ang=peer_df["avatar_ang_vel"].values/720.0; tele=peer_df["teleport_rate"].values/5.0
    part=peer_df["particle_intensity"].values/1.0; flick=peer_df["shader_flicker_hz"].values/30.0
    audio=peer_df["spatial_audio_jit"].values/100.0
    a,b,c,dd,e=coeffs
    return ((a*ang+b*tele+c*part+dd*flick+e*audio)*ov*dec).astype(np.float32)

def generate_param(n_cohorts=40,cohort_size=4,tracks=8,T=48,attack_prob=0.5,params=GenParams()):
    rows_u=[];rows_p=[];meta=[]
    for cid in range(n_cohorts):
        rng=np.random.default_rng(params.seed+cid)
        hmds=rng.integers(0,4,cohort_size)
        for tk in range(tracks):
            poses=[_sample_avatar_pose(rng,8.0) for _ in range(cohort_size)]
            udf=[_sim_user_track(rng,T,int(hmds[u])) for u in range(cohort_size)]
            is_atk=rng.random()<attack_prob
            aid=int(rng.integers(0,cohort_size)) if is_atk else -1
            ach=rng.choice(_PEER_NAMES) if is_atk else ""
            pdf=[_peer_state_param(rng,T,u==aid,ach if u==aid else "",params) for u in range(cohort_size)]
            sev=np.zeros((cohort_size,T),dtype=np.int64); attrib=np.full((cohort_size,T),-1,dtype=np.int64)
            for i in range(cohort_size):
                pos_i,ori_i,fov_i=poses[i]
                own=(udf[i]["optic_flow"].values/50.0+np.abs(udf[i]["yaw_rate"].values)/300.0+udf[i]["latency_ms"].values/200.0)/3.0
                pp=np.zeros((cohort_size,T),dtype=np.float32)
                for j in range(cohort_size):
                    if j==i: continue
                    pp[j]=_peer_contrib_param(pdf[j],pos_i,ori_i,fov_i,poses[j][0],params.coeffs,
                                              params.audio_oof_weight,params.peer_gain)
                score=own+pp.sum(0)
                t1,t2,t3=params.thresholds
                s=score+rng.normal(0,params.noise_sigma,T)
                yy=np.zeros(T,dtype=np.int64); yy[s>t1]=1; yy[s>t2]=2; yy[s>t3]=3
                sev[i]=yy
                if is_atk and i!=aid and pp[aid].mean()>0.02: attrib[i,:]=aid
            # DECOUPLE control: shuffle each peer's channel rows in time so channels no longer align with the label
            if params.decouple:
                for u in range(cohort_size):
                    perm=rng.permutation(T); pdf[u]=pdf[u].iloc[perm].reset_index(drop=True)
            for u in range(cohort_size):
                du=udf[u].copy(); du["cohort_id"]=cid; du["track_id"]=tk; du["user_id"]=u; du["t"]=np.arange(T)
                du["severity"]=sev[u]; du["attributed_peer"]=attrib[u]; rows_u.append(du)
                dp=pdf[u].copy(); dp["cohort_id"]=cid; dp["track_id"]=tk; dp["user_id"]=u; dp["t"]=np.arange(T); rows_p.append(dp)
            meta.append({"cohort_id":cid,"track_id":tk,"attacker_id":aid,"attack_channel":ach,
                         "poses":[(p[0].tolist(),p[1].tolist(),p[2]) for p in poses]})
    return rows_u,rows_p,meta

def save_param(rows_u,rows_p,meta,out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    pd.concat(rows_u,ignore_index=True).to_csv(out/"user_telemetry.csv",index=False)
    pd.concat(rows_p,ignore_index=True).to_csv(out/"peer_state.csv",index=False)
    with open(out/"tracks_meta.jsonl","w") as f:
        for m in meta: f.write(json.dumps(m)+"\n")
    schema={"user_channels":list(_USER_CHANNELS),"peer_channels":list(_PEER_NAMES),
            "severity_classes":[0,1,2,3],"attributed_peer":"int, -1 means no attacker"}
    json.dump(schema,open(out/"schema.json","w"),indent=2)

def build(out_dir,params,n_cohorts=40):
    ru,rp,mt=generate_param(n_cohorts=n_cohorts,params=params); save_param(ru,rp,mt,out_dir); return out_dir
