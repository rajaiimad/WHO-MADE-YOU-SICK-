"""Manuscript figures for main_revised4 (generator v2). Every plotted value is read
from outputs_v2/*.json; nothing is typed in. Run with XRP_GEN=v2.
-> ../manuscript/figures_v4/*.png"""
from __future__ import annotations

import json
import os

os.environ.setdefault("XRP_GEN", "v2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import J7, OUT, PEER_NAMES

FIG = J7 / "manuscript" / "figures_v4"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300,
                     "axes.facecolor": "white", "figure.facecolor": "white"})
J = lambda p: json.loads((OUT / p).read_text())
CH = {"avatar_ang_vel": "Angular\nvelocity", "teleport_rate": "Teleport", "particle_intensity": "Particle",
      "shader_flicker_hz": "Flicker", "spatial_audio_jit": "Spatial\naudio"}


def box(ax, x, y, w, h, text, fc="white", ec=INK2, fs=6.6, bold=False, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec,
                                lw=1.0, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=INK2))


def fig_method():
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")
    for x, t in ((0.15, "(a) On each headset"), (0.49, "(b) Co-presence graph"), (0.84, "(c) Two models")):
        ax.text(x, 0.96, t, fontsize=8, fontweight="bold", color=INK, ha="center")
    box(ax, 0.01, 0.62, 0.28, 0.22, "Own telemetry, 20 channels\n(motion, display, gaze, context)\nstays on the device")
    box(ax, 0.01, 0.32, 0.28, 0.22, "Frozen per-user backbone:\n4 severity log-probabilities\n+ 1 uncertainty value")
    box(ax, 0.01, 0.02, 0.28, 0.22, "Shared with the session:\nbackbone outputs + the 5\npeer-controllable channels")
    arrow(ax, (0.15, 0.62), (0.15, 0.54)); arrow(ax, (0.15, 0.32), (0.15, 0.24))
    box(ax, 0.35, 0.60, 0.28, 0.24, "Directed edge: peer j → victim i\ndistance, field-of-view overlap,\naudio proximity\n(from avatar poses)")
    box(ax, 0.35, 0.06, 0.28, 0.42, "Victim i, frame t: each\nincoming edge carries\npeer channels, excess over\nthe public benign band,\npeer backbone outputs,\nand edge geometry")
    arrow(ax, (0.29, 0.13), (0.35, 0.22)); arrow(ax, (0.49, 0.60), (0.49, 0.48))
    box(ax, 0.69, 0.50, 0.31, 0.36, "Detection: model each edge,\nthen aggregate over peers\n(a) MPNN: learned message\n(b) HGB: explicit interaction terms\n→ P(unsafe at frame t)", fc="#fdf0ea",
        ec=ORANGE)
    box(ax, 0.69, 0.06, 0.31, 0.34, "Attribution (per-edge classifier)\nscores every incoming edge;\ntop-scoring peer = candidate\nsource; channel-agnostic\nfeatures for unseen channels", fc="#eaf2fb", ec=BLUE)
    arrow(ax, (0.63, 0.30), (0.69, 0.66)); arrow(ax, (0.63, 0.25), (0.69, 0.22))
    fig.savefig(FIG / "Fig_method_pipeline.png", bbox_inches="tight"); plt.close(fig)


def fig_protocol():
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.00, 0.62, 0.21, 0.30, "XR-PERIL-Multi v2\ngenerator (fixed)\nall corpora on\ndisjoint seed ranges", bold=False)
    box(ax, 0.27, 0.62, 0.34, 0.30, "Development corpus: 60 cohorts\n5 repeats × 5 cohort-disjoint folds\ninner split per fold:\n36 train / 12 validation cohorts")
    box(ax, 0.67, 0.62, 0.33, 0.30, "Validation cohorts only: hyper-\nparameters, early stopping, Platt\ncalibration, decision thresholds")
    box(ax, 0.27, 0.10, 0.34, 0.34, "Final models: trained once per seed\n(45 train / 15 validation cohorts)\nconfiguration fixed by the CV\n(pre-registered before scoring)")
    box(ax, 0.67, 0.10, 0.33, 0.34, "Lock-box: 120 unseen cohorts,\nscored once  +  stress corpora\n(cohort size, generator shift,\nadaptive attacker, decoupling)", fc="#fdf0ea", ec=ORANGE)
    box(ax, 0.00, 0.10, 0.21, 0.34, "11 automated\nleakage checks\n(all pass)", fc="#eaf2fb", ec=BLUE)
    arrow(ax, (0.21, 0.77), (0.27, 0.77)); arrow(ax, (0.61, 0.77), (0.67, 0.77))
    arrow(ax, (0.44, 0.62), (0.44, 0.44)); arrow(ax, (0.61, 0.27), (0.67, 0.27))
    fig.savefig(FIG / "Fig_protocol.png", bbox_inches="tight"); plt.close(fig)


def fig_lockbox():
    F = {m: J(f"final_external/{m}.json")["lockbox"] for m in
         ("victim_bb__MLPpub", "proto__MLPpub", "rule__RULE", "gcngeo__GCNGEO", "edgefeat__HGB", "gat__GAT", "mpnn__MPNN",
          "edgefeatraw__HGB", "ORACLE")}
    lab = {"victim_bb__MLPpub": "Victim-\nonly", "proto__MLPpub": "Prototype\n(orig. MLP)", "rule__RULE": "Rule\ndetector",
           "gcngeo__GCNGEO": "GCN +\ngeometry", "edgefeat__HGB": "Excess-only\nint. HGB", "gat__GAT": "GAT", "mpnn__MPNN": "MPNN",
           "edgefeatraw__HGB": "Raw-channel\nint. HGB"}
    col = {m: (ORANGE if m in ("mpnn__MPNN", "edgefeatraw__HGB") else BLUE) for m in lab}
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.6))
    for ax, met, ttl in ((axs[0], "roc_auc", "ROC-AUC"), (axs[1], "pr_auc", "PR-AUC")):
        ms = list(lab)
        for i, m in enumerate(ms):
            v = F[m]["by_seed"][met][0]; lo, hi = F[m][f"ci95_{met}"]
            ax.bar(i, v, width=0.6, color=col[m], alpha=1.0 if col[m] == ORANGE else 0.55, edgecolor="white", lw=2)
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=INK, capsize=3, lw=1)
            ax.text(i, hi + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
        o = F["ORACLE"]["by_seed"][met][0]
        ax.axhline(o, color=INK2, ls="--", lw=1); ax.text(-0.4, o + 0.01, f"generator ceiling {o:.3f}", ha="left", va="bottom", fontsize=7, color=INK2)
        if met == "pr_auc":
            p = F["mpnn__MPNN"]["by_seed"]["prevalence"][0]
            ax.axhline(p, color=INK2, ls=":", lw=1); pass
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([lab[m] for m in ms], fontsize=6, rotation=35, ha="right")
        ax.set_ylim(0.4 if met == "roc_auc" else 0, 1.08); ax.set_title(ttl, fontsize=9, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / "Fig_lockbox_detection.png", bbox_inches="tight"); plt.close(fig)


def fig_channels():
    pc = J("per_channel_detection_lockbox.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.5))
    xs = range(len(PEER_NAMES)); w = 0.36
    for ax, k, ttl in ((axs[0], 0, "ROC-AUC by attack channel"), (axs[1], 1, "PR-AUC by attack channel")):
        for off, m, c, l in ((-w / 2, "proto__MLPpub", BLUE, "Prototype (original MLP)"), (w / 2, "mpnn__MPNN", ORANGE, "MPNN")):
            ax.bar([x + off for x in xs], [pc[ch][m][k] for ch in PEER_NAMES], width=w, color=c, edgecolor="white", lw=2, label=l)
        ax.scatter(list(xs), [pc[ch]["ORACLE"][k] for ch in PEER_NAMES], marker="_", s=260, color=INK, zorder=3, label="Generator ceiling")
        ax.set_xticks(list(xs)); ax.set_xticklabels([CH[c] for c in PEER_NAMES], fontsize=7)
        ax.set_ylim(0.4 if k == 0 else 0, 1.02); ax.set_title(ttl, fontsize=9)
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(FIG / "Fig_per_channel.png", bbox_inches="tight"); plt.close(fig)


def fig_robust():
    F = {m: J(f"final_external/{m}.json") for m in ("proto__MLPpub", "rule__RULE", "edgefeat__HGB", "gat__GAT", "mpnn__MPNN",
                                                    "edgefeatraw__HGB", "ORACLE")}
    SER = (("proto__MLPpub", BLUE, "Prototype (original MLP)", "-"), ("rule__RULE", "#7a7a7a", "Rule detector", "-"),
           ("edgefeat__HGB", AQUA, "Excess-only int. HGB", "-"), ("gat__GAT", "#9b59b6", "GAT", "-"),
           ("mpnn__MPNN", ORANGE, "MPNN", "-"), ("edgefeatraw__HGB", "#b5341c", "Raw-channel int. HGB", ":"),
           ("ORACLE", INK2, "Generator ceiling", "--"))
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.6))
    sizes = [(2, "size_N2"), (3, "size_N3"), (4, "lockbox"), (6, "size_N6"), (8, "size_N8")]
    for m, c, l, ls in SER:
        axs[0].plot([n for n, _ in sizes], [F[m][k]["by_seed"]["roc_auc"][0] for _, k in sizes], marker="o", ms=3, lw=1.6,
                    color=c, ls=ls, label=l)
    axs[0].axvline(4, color=GRID, lw=1); axs[0].text(4.1, 0.25, "training size", fontsize=7, color=INK2)
    axs[0].axhline(0.5, color=INK2, lw=0.8, ls=":"); axs[0].text(6.2, 0.52, "chance", fontsize=7, color=INK2)
    axs[0].set_xlabel("Users per cohort at test time"); axs[0].set_ylabel("ROC-AUC"); axs[0].set_ylim(0.1, 1.05)
    axs[0].set_title("(a) Transfer to unseen cohort sizes", fontsize=9)
    al = [1.0, 0.8, 0.6, 0.4, 0.2]
    for m, c, l, ls in SER:
        axs[1].plot(al, [F[m][f"adapt_a{a}"]["by_seed"]["fpr10_recall"][0] for a in al], marker="o", ms=3, lw=1.6, color=c,
                    ls=ls, label=l)
    axs[1].invert_xaxis(); axs[1].set_xlabel("Attacker intensity α (lower = stealthier)")
    axs[1].set_ylabel("Recall at 10% false-alarm budget"); axs[1].set_ylim(0, 1.05)
    axs[1].set_title("(b) Adaptive attacker, labels re-simulated", fontsize=9)
    for ax in axs:
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.11, 1, 1)); fig.savefig(FIG / "Fig_robustness.png", bbox_inches="tight"); plt.close(fig)


def fig_loco():
    A = J("attribution_summary.json")["loco"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    xs = range(len(PEER_NAMES)); w = 0.36
    for off, fs, c, l in ((-w / 2, "paper", BLUE, "Channel-specific features"), (w / 2, "agnostic", AQUA, "Channel-agnostic features")):
        ax.bar([x + off for x in xs], [A[f"{fs}|{ch}"]["top1"][0] for ch in PEER_NAMES], width=w, color=c, edgecolor="white", lw=2, label=l)
    ax.axhline(1 / 3, color=INK2, ls=":", lw=1); ax.text(4.45, 1 / 3 + 0.02, "chance (1/3)", fontsize=6.5, color=INK2, ha="right")
    ax.set_xticks(list(xs)); ax.set_xticklabels([CH[c] for c in PEER_NAMES], fontsize=6.5)
    ax.set_ylabel("Top-1 attribution accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Held-out (unseen) attack channel", fontsize=8.5)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "Fig_loco_attribution.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_method, fig_protocol, fig_lockbox, fig_channels, fig_robust, fig_loco):
        f()
    print("figures ->", FIG)
