"""
coper_vr.config
===============
Single source of truth for J7 hyperparameters, paths, and seeds.
Nothing else in the package hard-codes a number.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Repository layout — resolved relative to this file.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent            # .../Journal 7/coper_vr
_J7_ROOT = _HERE.parent                            # .../Journal 7
_JOURNAL_ROOT = _J7_ROOT.parent                    # .../journal - under proccess
_J6_ROOT = _JOURNAL_ROOT / "Journal 6 - under process"
_XRPERIL_ROOT = _J6_ROOT / "Realizable_VR_Attacks_Toolkit"


@dataclass(frozen=True)
class Paths:
    j7_root: Path = _J7_ROOT
    j6_root: Path = _J6_ROOT
    xrperil_root: Path = _XRPERIL_ROOT
    xrperil_pkg: Path = _XRPERIL_ROOT / "xrperil"        # J5 package
    raad_pkg: Path = _J6_ROOT / "raad_xr"                # J6 package
    uffcs_raw: Path = _J7_ROOT / "UFFCSData-master_raw_source"   # raw XML (E1)
    synth_dir: Path = _XRPERIL_ROOT / "synthetic"         # J5 single-user synth
    synth_multi_dir: Path = _J7_ROOT / "synthetic_multi"  # J7 multi-agent corpus
    results_dir: Path = _J7_ROOT / "results_j7"


PATHS = Paths()


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
SEEDS: Tuple[int, ...] = (0, 1, 2, 42, 2024)


# ---------------------------------------------------------------------------
# Multi-agent synthetic corpus
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SynthMultiConfig:
    n_cohorts: int = 60
    cohort_size: int = 4                # sweep: {2, 3, 4, 6, 8}
    tracks_per_cohort: int = 8
    track_len: int = 48                 # frames per track
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    scene_radius_m: float = 8.0         # avatars sampled within this radius
    fov_deg_default: float = 90.0
    audio_range_m: float = 6.0
    attack_prob: float = 0.35           # fraction of tracks with an attacker
    max_attackers_per_cohort: int = 1   # keep attribution unambiguous
    seed: int = 0


SYNTH_MULTI = SynthMultiConfig()


# ---------------------------------------------------------------------------
# Peer-controllable channels (new in J7). All in physical units.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PeerChannel:
    name: str
    unit: str
    lo: float
    hi: float
    benign_lo: float
    benign_hi: float


PEER_CHANNELS: Tuple[PeerChannel, ...] = (
    PeerChannel("avatar_ang_vel",     "deg/s",   0.0, 720.0,   0.0,  60.0),
    PeerChannel("teleport_rate",      "1/s",     0.0,   5.0,   0.0,   0.5),
    PeerChannel("particle_intensity", "0-1",     0.0,   1.0,   0.0,   0.2),
    PeerChannel("shader_flicker_hz",  "Hz",      0.0,  30.0,   0.0,   1.0),
    PeerChannel("spatial_audio_jit",  "ms",      0.0, 100.0,   0.0,   5.0),
)


# ---------------------------------------------------------------------------
# T-GAT
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TGATConfig:
    input_dim: int = 5                 # 4 logits + 1 uncertainty
    hidden_dim: int = 32
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.10
    edge_feat_dim: int = 4             # dist, fov_overlap, audio_prox, edge_type
    temporal_window: int = 8           # K frames for GRU
    gru_hidden: int = 64


TGAT = TGATConfig()


# ---------------------------------------------------------------------------
# Federation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FedConfig:
    aggregator: str = "trimmed_median"  # {fedavg, median, multi_krum, trimmed_median}
    trim_beta: float = 0.4
    local_epochs: int = 2
    rounds: int = 50
    lr: float = 3e-4
    byzantine_fracs: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4)


FED = FedConfig()


# ---------------------------------------------------------------------------
# Safety thresholds — reused from J3.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SafetyConfig:
    d_fnr: float = 0.05        # Neyman-Pearson budget on false-negative rate
    d_fpr: float = 0.10        # secondary budget on false-positive rate
    d_ux: float = 0.10         # unnecessary-intervention rate
    cvar_alpha: float = 0.05


SAFETY = SafetyConfig()


# ---------------------------------------------------------------------------
# Attack parameters (inherit J5 epsilons, extend to peer channels).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AttackConfig:
    epsilons: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    pgd_steps: int = 20
    pgd_adaptive_steps: int = 50
    pgd_adaptive_restarts: int = 5


ATTACKS = AttackConfig()


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    for p in [PATHS.synth_multi_dir, PATHS.results_dir]:
        p.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("Paths:")
    for k, v in PATHS.__dict__.items():
        exists = "OK" if Path(v).exists() else "MISSING"
        print(f"  {k:15s}  {exists:8s}  {v}")
    print(f"Seeds: {SEEDS}")
    print(f"Cohort size default: {SYNTH_MULTI.cohort_size}")
    print(f"Peer channels: {[c.name for c in PEER_CHANNELS]}")
