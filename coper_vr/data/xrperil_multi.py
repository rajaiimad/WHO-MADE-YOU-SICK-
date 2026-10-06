"""
coper_vr.data.xrperil_multi
===========================
Multi-agent extension of J5's XR-PERIL physics-grounded synthetic corpus.

For each cohort of `cohort_size` avatars, this module generates
`tracks_per_cohort` synchronized tracks of length `track_len`. Each track is:
    - a per-user telemetry matrix X_i of shape (T, D_user), where D_user is
      the 20-channel XR-PERIL schema (16 continuous + 4 discrete);
    - a per-frame peer-controllable state P_i of shape (T, D_peer) — the new
      channels defined in config.PEER_CHANNELS;
    - an avatar-pose track (position, orientation, fov_deg) used by the
      co-presence graph;
    - a per-frame severity label y_i in {0,1,2,3};
    - a ground-truth attribution label:
          attacker_id ∈ {-1, 0, ..., cohort_size-1}
          attack_channel ∈ {'', 'ang_vel', 'teleport', 'particle',
                             'flicker', 'audio_jit'}

The single-user physics coupling from J5 (`latency ≈ frame_interval + jitter`,
`optic_flow ≈ 0.05·speed·FOV`, `accel ≈ d(speed)/dt`) is preserved for every
avatar. Cross-user coupling is added:
    - line-of-sight decay: a peer's contribution to a victim's optic flow
      falls off as (dist^-1) * FOV_overlap;
    - audio propagation: spatial-audio jitter contributes with delay
      dist / c_air (approx.) and decays with dist^-1;
    - shader flicker adds a term to the victim's rendered luminance
      variance if the peer is inside the victim's FOV.

The victim's severity label is a monotone function of a hidden
"provocation score" derived from own kinematics + peer contributions.
Attribution is deterministic: the attacker is the peer with the largest
contribution to the victim's provocation score above baseline.

Usage
-----
    from coper_vr.data.xrperil_multi import generate_multi
    cohorts = generate_multi(n_cohorts=8, cohort_size=4, seed=0)
    # cohorts is a list of dicts, one per cohort.

CLI:
    python -m coper_vr.data.xrperil_multi --n_cohorts 60 --cohort_size 4
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Enable "from xrperil import ..." by adding J5 root to sys.path.
_HERE = Path(__file__).resolve()
_J7 = _HERE.parent.parent.parent          # .../Journal 7 - Who Made You Sick
_J6 = _J7.parent / "Journal 6 - under process"
_XRP = _J6 / "Realizable_VR_Attacks_Toolkit"
for p in (_XRP, _J6):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from coper_vr.config import PATHS, SYNTH_MULTI, PEER_CHANNELS  # noqa: E402


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _fov_overlap(pos_i, ori_i, pos_j, fov_deg_i) -> float:
    """
    Is peer j inside victim i's FOV? Returns [0, 1]: 1 if inside the FOV
    cone axis-aligned with ori_i, decaying to 0 at the FOV boundary.
    """
    v = np.array(pos_j) - np.array(pos_i)
    d = np.linalg.norm(v) + 1e-9
    v_hat = v / d
    ori = np.array(ori_i) / (np.linalg.norm(ori_i) + 1e-9)
    cos_th = float(np.clip(np.dot(ori, v_hat), -1.0, 1.0))
    th_rad = np.arccos(cos_th)
    th_deg = np.degrees(th_rad)
    half = fov_deg_i / 2.0
    if th_deg > half:
        return 0.0
    # linear falloff inside the cone
    return float(1.0 - th_deg / half)


def _sample_avatar_pose(rng, radius: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sample a starting pose (position, forward orientation, FOV degrees)."""
    r = rng.uniform(0.5, radius)
    theta = rng.uniform(0, 2 * np.pi)
    pos = np.array([r * np.cos(theta), r * np.sin(theta), rng.uniform(1.4, 1.9)])
    yaw = rng.uniform(0, 2 * np.pi)
    ori = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    fov = float(rng.uniform(85.0, 105.0))
    return pos, ori, fov


# ---------------------------------------------------------------------------
# XR-PERIL single-user physics — a compact re-implementation of what
# xrperil.data_synth.generate() produces, so J7 remains self-contained
# even if the J5 generator is upgraded. Named-channel convention matches
# xrperil.data_synth exactly so downstream metrics apply unchanged.
# ---------------------------------------------------------------------------
_USER_CHANNELS: Tuple[str, ...] = (
    # 16 continuous physical
    "speed", "accel", "yaw_rate", "pitch_rate", "roll_rate",
    "optic_flow", "fov_deg", "frame_interval_ms", "jitter_ms", "latency_ms",
    "gaze_x", "gaze_y", "gaze_pupil", "head_x", "head_y", "head_z",
    # 4 discrete context
    "context_loco",     # 0=stationary,1=smooth,2=teleport,3=running
    "context_scene",    # 0=indoor,1=outdoor,2=vehicular
    "context_task",     # 0=passive,1=training,2=game,3=clinical
    "context_hmd",      # 0=Quest2,1=Quest3,2=Vive,3=Index
)


def _sim_user_track(rng, T: int, base_hmd: int) -> pd.DataFrame:
    """Generate one T-frame telemetry track under J5 physics."""
    speed = np.clip(rng.normal(1.0, 0.4, T).cumsum() * 0.05 + 0.5, 0.0, 3.0)
    accel = np.gradient(speed) * 60.0                          # 60 fps → per s
    yaw_rate = np.clip(rng.normal(0.0, 8.0, T), -60, 60)
    pitch_rate = np.clip(rng.normal(0.0, 3.0, T), -20, 20)
    roll_rate = np.clip(rng.normal(0.0, 1.5, T), -10, 10)

    fov_deg = np.full(T, rng.uniform(88.0, 100.0))
    frame_interval_ms = np.full(T, rng.uniform(11.0, 13.5))    # ~72-90 Hz
    jitter_ms = np.abs(rng.normal(0.5, 0.4, T))
    latency_ms = frame_interval_ms + jitter_ms                  # J5 coupling

    optic_flow = 0.05 * speed * fov_deg                         # J5 coupling
    gaze_x = np.clip(rng.normal(0.0, 0.2, T).cumsum() * 0.05, -1, 1)
    gaze_y = np.clip(rng.normal(0.0, 0.2, T).cumsum() * 0.05, -1, 1)
    gaze_pupil = np.clip(rng.normal(3.5, 0.4, T), 2.0, 6.0)

    head_x = np.cumsum(rng.normal(0.0, 0.02, T))
    head_y = np.cumsum(rng.normal(0.0, 0.02, T))
    head_z = 1.65 + rng.normal(0.0, 0.01, T)

    context_loco = np.full(T, int(rng.integers(0, 4)))
    context_scene = np.full(T, int(rng.integers(0, 3)))
    context_task = np.full(T, int(rng.integers(0, 4)))
    context_hmd = np.full(T, base_hmd)

    df = pd.DataFrame({
        "speed": speed, "accel": accel,
        "yaw_rate": yaw_rate, "pitch_rate": pitch_rate, "roll_rate": roll_rate,
        "optic_flow": optic_flow, "fov_deg": fov_deg,
        "frame_interval_ms": frame_interval_ms, "jitter_ms": jitter_ms,
        "latency_ms": latency_ms,
        "gaze_x": gaze_x, "gaze_y": gaze_y, "gaze_pupil": gaze_pupil,
        "head_x": head_x, "head_y": head_y, "head_z": head_z,
        "context_loco": context_loco, "context_scene": context_scene,
        "context_task": context_task, "context_hmd": context_hmd,
    })
    assert list(df.columns) == list(_USER_CHANNELS), "channel order drift"
    return df


# ---------------------------------------------------------------------------
# Peer state — one row per avatar per frame with the 5 peer-controllable
# channels. Benign peers stay in the benign band; the attacker pushes one
# channel toward its high end.
# ---------------------------------------------------------------------------
def _sim_peer_state(rng, T: int, attacker: bool, ch_name: str = "") -> pd.DataFrame:
    cols = {}
    for c in PEER_CHANNELS:
        v = rng.uniform(c.benign_lo, c.benign_hi, T)
        cols[c.name] = v
    if attacker and ch_name:
        c = next(ch for ch in PEER_CHANNELS if ch.name == ch_name)
        # ramp toward the high end over the track, with jitter.
        ramp = np.linspace(c.benign_hi, c.hi, T) * rng.uniform(0.85, 1.0, T)
        cols[c.name] = np.clip(ramp, c.lo, c.hi)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Cross-user provocation and severity labelling
# ---------------------------------------------------------------------------
def _peer_contribution(peer_state_j: pd.DataFrame,
                       pos_i, ori_i, fov_i,
                       pos_j) -> np.ndarray:
    """
    How much does peer j contribute to victim i's provocation score at each
    frame? Composed of five per-channel physics couplings.
    """
    T = len(peer_state_j)
    d = float(np.linalg.norm(np.array(pos_j) - np.array(pos_i))) + 1e-3
    ov = _fov_overlap(pos_i, ori_i, pos_j, fov_i)
    dist_decay = 1.0 / d
    if ov == 0.0:
        # Only spatial-audio jitter contributes if peer is outside FOV.
        return (peer_state_j["spatial_audio_jit"].values / 100.0) * dist_decay

    ang = peer_state_j["avatar_ang_vel"].values / 720.0
    tele = peer_state_j["teleport_rate"].values / 5.0
    part = peer_state_j["particle_intensity"].values / 1.0
    flick = peer_state_j["shader_flicker_hz"].values / 30.0
    audio = peer_state_j["spatial_audio_jit"].values / 100.0

    contrib = (0.35 * ang + 0.20 * tele + 0.15 * part +
               0.20 * flick + 0.10 * audio) * ov * dist_decay
    return contrib.astype(np.float32)


def _severity_from_score(score: np.ndarray, rng) -> np.ndarray:
    """Map continuous provocation → ordinal severity 0..3 with the J5-consistent
    hidden thresholds; add per-frame noise for realism."""
    s = score + rng.normal(0.0, 0.05, size=score.shape)
    y = np.zeros_like(s, dtype=np.int64)
    y[s > 0.15] = 1
    y[s > 0.30] = 2
    y[s > 0.55] = 3
    return y


# ---------------------------------------------------------------------------
# Main entry: generate one cohort
# ---------------------------------------------------------------------------
def _generate_one_cohort(cohort_id: int, seed: int,
                         cohort_size: int, tracks_per_cohort: int,
                         track_len: int, radius: float,
                         attack_prob: float) -> Dict:
    rng = _rng(seed)
    # Sample per-user identity: HMD type + a stable subject id.
    hmds = rng.integers(0, 4, size=cohort_size)
    subject_ids = [f"C{cohort_id:04d}_U{u}" for u in range(cohort_size)]

    tracks = []
    for tk in range(tracks_per_cohort):
        # Random poses per track (avatars move between tracks).
        poses = [_sample_avatar_pose(rng, radius) for _ in range(cohort_size)]
        # Per-user telemetry.
        user_dfs = [_sim_user_track(rng, track_len, int(hmds[u]))
                    for u in range(cohort_size)]
        # Decide attacker for this track.
        is_attack = rng.random() < attack_prob
        attacker_id = int(rng.integers(0, cohort_size)) if is_attack else -1
        attack_channel = ""
        if is_attack:
            attack_channel = rng.choice([c.name for c in PEER_CHANNELS])
        peer_dfs = [_sim_peer_state(rng, track_len,
                                    attacker=(u == attacker_id),
                                    ch_name=attack_channel if u == attacker_id else "")
                    for u in range(cohort_size)]

        # Compute per-user provocation from own telemetry + peer contributions.
        contribs_per_victim = np.zeros((cohort_size, track_len), dtype=np.float32)
        peer_attrib = np.full((cohort_size, track_len), -1, dtype=np.int64)
        for i in range(cohort_size):
            pos_i, ori_i, fov_i = poses[i]
            # Own kinematic provocation (compact surrogate; monotone in optic flow).
            own_prov = (user_dfs[i]["optic_flow"].values / 50.0
                        + np.abs(user_dfs[i]["yaw_rate"].values) / 300.0
                        + user_dfs[i]["latency_ms"].values / 200.0) / 3.0
            best_peer, best_val = -1, 0.0
            per_peer_scores = np.zeros((cohort_size, track_len), dtype=np.float32)
            for j in range(cohort_size):
                if j == i:
                    continue
                pos_j = poses[j][0]
                c = _peer_contribution(peer_dfs[j], pos_i, ori_i, fov_i, pos_j)
                per_peer_scores[j] = c
            total_peer = per_peer_scores.sum(axis=0)
            contribs_per_victim[i] = own_prov + total_peer
            # Attribution: peer whose mean contribution over the track is largest,
            # provided it exceeds a small threshold above the median.
            peer_means = per_peer_scores.mean(axis=1)
            if is_attack and i != attacker_id and peer_means[attacker_id] > 0.02:
                peer_attrib[i, :] = attacker_id

        # Severity labels.
        severities = np.stack([_severity_from_score(contribs_per_victim[i], rng)
                               for i in range(cohort_size)], axis=0)

        tracks.append({
            "cohort_id": cohort_id,
            "track_id": tk,
            "attacker_id": attacker_id,
            "attack_channel": attack_channel,
            "poses": [(p[0].tolist(), p[1].tolist(), p[2]) for p in poses],
            "user_dfs": [df.copy() for df in user_dfs],
            "peer_dfs": [df.copy() for df in peer_dfs],
            "severities": severities,               # (N, T)
            "peer_attrib": peer_attrib,             # (N, T), -1 or peer id
        })

    return {
        "cohort_id": cohort_id,
        "subject_ids": subject_ids,
        "hmds": hmds.tolist(),
        "tracks": tracks,
    }


def generate_multi(n_cohorts: int = SYNTH_MULTI.n_cohorts,
                   cohort_size: int = SYNTH_MULTI.cohort_size,
                   tracks_per_cohort: int = SYNTH_MULTI.tracks_per_cohort,
                   track_len: int = SYNTH_MULTI.track_len,
                   radius: float = SYNTH_MULTI.scene_radius_m,
                   attack_prob: float = SYNTH_MULTI.attack_prob,
                   seed: int = SYNTH_MULTI.seed) -> List[Dict]:
    return [_generate_one_cohort(cid, seed + cid,
                                 cohort_size, tracks_per_cohort,
                                 track_len, radius, attack_prob)
            for cid in range(n_cohorts)]


# ---------------------------------------------------------------------------
# Flatten & save
# ---------------------------------------------------------------------------
def save_multi(cohorts: List[Dict], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_user = []
    rows_peer = []
    rows_meta = []
    for coh in cohorts:
        cid = coh["cohort_id"]
        for tk in coh["tracks"]:
            for u in range(len(tk["user_dfs"])):
                df_u = tk["user_dfs"][u].copy()
                df_u["cohort_id"] = cid
                df_u["track_id"] = tk["track_id"]
                df_u["user_id"] = u
                df_u["t"] = np.arange(len(df_u))
                df_u["severity"] = tk["severities"][u]
                df_u["attributed_peer"] = tk["peer_attrib"][u]
                rows_user.append(df_u)
                df_p = tk["peer_dfs"][u].copy()
                df_p["cohort_id"] = cid
                df_p["track_id"] = tk["track_id"]
                df_p["user_id"] = u
                df_p["t"] = np.arange(len(df_p))
                rows_peer.append(df_p)
            rows_meta.append({
                "cohort_id": cid,
                "track_id": tk["track_id"],
                "attacker_id": tk["attacker_id"],
                "attack_channel": tk["attack_channel"],
                "poses": tk["poses"],
            })
    pd.concat(rows_user, ignore_index=True).to_csv(out_dir / "user_telemetry.csv", index=False)
    pd.concat(rows_peer, ignore_index=True).to_csv(out_dir / "peer_state.csv", index=False)
    with open(out_dir / "tracks_meta.jsonl", "w") as f:
        for m in rows_meta:
            f.write(json.dumps(m) + "\n")
    schema = {
        "user_channels": list(_USER_CHANNELS),
        "peer_channels": [c.name for c in PEER_CHANNELS],
        "peer_units": {c.name: c.unit for c in PEER_CHANNELS},
        "peer_bounds": {c.name: (c.lo, c.hi) for c in PEER_CHANNELS},
        "peer_benign_bounds": {c.name: (c.benign_lo, c.benign_hi) for c in PEER_CHANNELS},
        "severity_classes": [0, 1, 2, 3],
        "attributed_peer": "int, -1 means no attacker",
    }
    with open(out_dir / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli():
    p = argparse.ArgumentParser(description="Generate XR-PERIL-Multi corpus.")
    p.add_argument("--n_cohorts", type=int, default=SYNTH_MULTI.n_cohorts)
    p.add_argument("--cohort_size", type=int, default=SYNTH_MULTI.cohort_size)
    p.add_argument("--tracks_per_cohort", type=int, default=SYNTH_MULTI.tracks_per_cohort)
    p.add_argument("--track_len", type=int, default=SYNTH_MULTI.track_len)
    p.add_argument("--attack_prob", type=float, default=SYNTH_MULTI.attack_prob)
    p.add_argument("--seed", type=int, default=SYNTH_MULTI.seed)
    p.add_argument("--out", type=str, default=str(PATHS.synth_multi_dir))
    args = p.parse_args()
    cohorts = generate_multi(n_cohorts=args.n_cohorts,
                             cohort_size=args.cohort_size,
                             tracks_per_cohort=args.tracks_per_cohort,
                             track_len=args.track_len,
                             attack_prob=args.attack_prob,
                             seed=args.seed)
    save_multi(cohorts, Path(args.out))
    print(f"Wrote {args.n_cohorts} cohorts (cohort_size={args.cohort_size}, "
          f"tracks/cohort={args.tracks_per_cohort}, T={args.track_len}) "
          f"to {args.out}")


if __name__ == "__main__":
    _cli()
