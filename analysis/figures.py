"""The five figures of the main text, drawn from results/summary/ (run analysis/tables.py first).

Writes results/figures/fig_{pipeline,ladder,detail_probes,planted,regime_map}.pdf at the
article's 31 pc text width.

    uv run analysis/figures.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "qenc-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle

from qenc.utils import RESULTS

SUMMARY, OUT = RESULTS / "summary", RESULTS / "figures"
WIDTH = 372 / 72.27  # sn-jnl text width, 31 pc, in inches
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAMILY = {"angle": ("#2a78d6", "o"), "amplitude": ("#eb6834", "s"), "qsm": ("#1baf7a", "D")}
DEPTH_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
TASKS = ["ECG", "EEG", "MNIST1D", "PneumoniaMNIST", "NoduleMNIST3D"]
LABEL = {"ECG": "ECG", "EEG": "EEG", "MNIST1D": "MNIST1D", "PneumoniaMNIST": "Pneumonia", "NoduleMNIST3D": "Nodule 3D"}
LADDER_ROWS = {("PneumoniaMNIST", "angle"): "angle8", ("PneumoniaMNIST", "amplitude"): "amplitude16",
               ("PneumoniaMNIST", "qsm"): "native16", ("NoduleMNIST3D", "angle"): "angle8",
               ("NoduleMNIST3D", "amplitude"): "amplitude64", ("NoduleMNIST3D", "qsm"): "native64"}
GAPS = [("info_loss L0-L1", "Pooling before\nencoding", "$L_0-L_1$"),
        ("state_vs_input L2tr-L1", "Encoded state\nagainst its input", "$L_2-L_1$"),
        ("unused_state L2tr-L4tr", "State unused by\nthe trained readout", "$L_2-L_4$"),
        ("vs_random L4tr-L5", "Learning beyond\nrandom starts", "$L_4-\\bar L_5$")]
RC = {"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
      "axes.spines.top": False, "axes.spines.right": False}


def save(fig, stem, **kwargs):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", **kwargs)
    plt.close(fig)


def style(ax, surface):
    ax.set_facecolor(surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def pipeline():
    """Figure 1: the representation interface; grey elements are fixed by design, white ones are trained."""
    line, fixed, fixed_edge, frame = "#2b3035", "#d9dee3", "#4a535b", "#f7f8f9"
    bottom, top, yc = 0.14, 1.94, 1.27  # axes units are inches
    fig = plt.figure(figsize=(WIDTH, top - bottom))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(bottom, top))
    ax.axis("off")

    def arrow(x0, x1):
        ax.add_patch(FancyArrowPatch((x0, yc), (x1, yc), arrowstyle="-|>", mutation_scale=6, lw=0.7, color=line,
                                     shrinkA=0, shrinkB=0))

    def block(x0, x1, half, label, is_fixed, size):
        ax.add_patch(Rectangle((x0, yc - half), x1 - x0, 2 * half, facecolor=fixed if is_fixed else "white",
                               edgecolor=fixed_edge if is_fixed else line, lw=0.7, zorder=3))
        ax.text((x0 + x1) / 2, yc, label, ha="center", va="center", fontsize=size, color=INK, zorder=4)

    def meter(xc, y, w=0.17, h=0.14):
        ax.add_patch(Rectangle((xc - w / 2, y - h / 2), w, h, facecolor="white", edgecolor=line, lw=0.6, zorder=3))
        ax.add_patch(Arc((xc, y - h / 2 + 0.025), w * 0.72, h * 1.1, theta1=0, theta2=180, lw=0.5, color=line,
                         zorder=4))
        ax.plot([xc, xc + w * 0.28], [y - h / 2 + 0.025, y + h / 2 - 0.025], color=line, lw=0.5, zorder=4)

    def caption(x, text, y=yc + 0.29):
        ax.text(x, y, text, ha="center", va="bottom", fontsize=7, color=MUTED)

    def callout(x_anchor, y_anchor, x_text, title, lines, rule=2.3):
        ax.plot([x_anchor, x_anchor], [y_anchor, 0.69], color=MUTED, lw=0.5)
        ax.plot([x_anchor], [y_anchor], marker="o", ms=2.2, color=MUTED)
        ax.plot([x_text, x_text + rule], [0.69, 0.69], color=MUTED, lw=0.5)
        ax.text(x_text, 0.58, title, ha="left", va="center", fontsize=7.5, color=INK, fontweight="bold")
        for i, text in enumerate(lines):
            ax.text(x_text, 0.41 - 0.16 * i, text, ha="left", va="center", fontsize=7, color=MUTED)

    ax.text(0.12, yc, "$x$", ha="center", va="center", fontsize=10, color=INK)
    arrow(0.22, 0.4)
    block(0.4, 0.8, 0.17, "$P_c$", is_fixed=True, size=9.5)
    caption(0.6, "preprocessing")
    arrow(0.8, 0.96)
    ax.text(1.06, yc, "$z$", ha="center", va="center", fontsize=10, color=INK)
    arrow(1.15, 1.28)

    ax.add_patch(FancyBboxPatch((1.28, yc - 0.45), 2.19, 0.9, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=frame, edgecolor="#aab2b9", lw=0.6, zorder=1))
    caption(2.375, r"circuit $W_{c,\theta}(z)$", y=yc + 0.5)
    wires = (yc - 0.23, yc, yc + 0.23)
    for y in wires:
        ax.plot([1.58, 3.3], [y, y], color=line, lw=0.7, zorder=2)
        ax.text(1.555, y, r"$|0\rangle$", ha="right", va="center", fontsize=7.5, color=INK)
        ax.text(3.115, y, r"$\cdots$", ha="center", va="center", fontsize=8.5, color=INK, zorder=4,
                bbox={"facecolor": frame, "edgecolor": "none", "pad": 1})
        meter(3.3, y)
    for x0, label, is_fixed in [(1.63, "$S(z)$", True), (2.0, r"$W_{\theta_1}$", False),
                                (2.37, "$S(z)$", True), (2.74, r"$W_{\theta_2}$", False)]:
        block(x0, x0 + 0.29, 0.32, label, is_fixed, size=8)
    arrow(3.47, 3.62)

    ax.text(3.9, yc, r"$\phi_{c,\theta}(x)$", ha="center", va="center", fontsize=9.5, color=INK)
    caption(3.9, "features")
    arrow(4.17, 4.32)
    block(4.32, 5.08, 0.17, r"$b+w^{\mathsf{T}}\phi$", is_fixed=False, size=9)
    caption(4.7, "affine readout")

    callout(1.06, yc - 0.12, 0.08, "Information access",
            [r"fine grid $F$ vs. expanded coarse grid $PF$,", "same circuit, initialization and schedule"])
    callout(3.9, yc - 0.14, 2.8, "Learned accessibility",
            [r"features $\phi$ at initialization vs. after training,", "same linear probe on the same examples"])
    save(fig, "fig_pipeline", pad_inches=0.02)


def ladder():
    """Figure 2: paired gaps between ladder levels with cluster-bootstrap intervals."""
    gaps = pd.read_csv(SUMMARY / "ladder_gaps.csv")
    entries = [(t, f, LADDER_ROWS.get((t, f), f)) for t in TASKS for f in ("angle", "amplitude", "qsm")]
    fig, axes = plt.subplots(1, len(GAPS), figsize=(WIDTH, 2.72), sharey=True, layout="constrained")
    y = {(task, cand): i + TASKS.index(task) * 0.6 for i, (task, _, cand) in enumerate(entries)}
    for ax, (key, title, symbol) in zip(axes, GAPS):
        g = gaps[gaps.gap == key].set_index(["dataset", "candidate"])
        for task, family, cand in entries:
            r = g.loc[(task, cand)]
            color, marker = FAMILY[family]
            yy = y[(task, cand)]
            ax.plot([100 * r.ci_low, 100 * r.ci_high], [yy, yy], color=color, linewidth=1.1, solid_capstyle="round")
            ax.plot(100 * r["mean"], yy, marker=marker, color=color, markersize=3.6, markeredgecolor="white",
                    markeredgewidth=0.5, linestyle="none")
        ax.axvline(0, color=MUTED, linewidth=0.7)
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title(f"{title}\n{symbol}", fontsize=7, color=INK, loc="left")
        ax.tick_params(colors=MUTED, labelsize=6.5)
        ax.set_xlabel("percentage points", fontsize=6.5, color=MUTED)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
    centers = [np.mean([y[(t, c)] for tt, _, c in entries if tt == t]) for t in TASKS]
    axes[0].set_yticks(centers, [LABEL[t] for t in TASKS], fontsize=7, color=INK)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=c, marker=m, linewidth=1.1, markersize=3.6, label=f)
               for f, (c, m) in FAMILY.items()]
    fig.legend(handles=handles, labels=["Angle (eight means)", "Amplitude", "QSM"], loc="outside lower center",
               ncol=3, frameon=False, fontsize=7)
    save(fig, "fig_ladder")


def detail_probes():
    """Figure 3: retained detail at fixed capacity and learned accessibility, paired seeds."""
    detail = pd.read_csv(SUMMARY / "detail_contrasts.csv")
    probes = pd.read_csv(SUMMARY / "probe_contrasts.csv")
    fig = plt.figure(figsize=(WIDTH, 3.2), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.15], hspace=.05)
    axes = top.subplots(1, 2, sharey=True)
    for ax, task, label in zip(axes, TASKS[-2:], ["PneumoniaMNIST (2D)", "NoduleMNIST3D (3D)"]):
        for i, family in enumerate(["amplitude", "qsm"]):
            row = detail[(detail.dataset == task) & (detail.family == family)].iloc[0]
            values = np.array(json.loads(row["values"])) * 100
            color = FAMILY[family][0]
            ax.scatter(i + np.linspace(-.1, .1, 5), values, color=color, s=10, zorder=3)
            ax.plot([i - .22, i + .22], [values.mean()] * 2, color=color, lw=1.8)
            ax.text(i + .27, values.mean(), f"{values.mean():+.2f}", va="center", fontsize=6.5)
        ax.axhline(0, color="#77828a", ls=":", lw=.8)
        ax.set(xticks=[0, 1], xticklabels=["Amplitude", "QSM"], xlim=(-.45, 1.75))
        ax.set_title(label, fontsize=7)
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Fine minus expanded\ncoarse BA (pp)")
    top.suptitle("(a) Retained detail at fixed circuit capacity", x=0.01, ha="left", fontsize=8)
    selected = probes[(~probes.dataset.isin(TASKS[-2:]))
                      | probes.candidate.isin(["angle8", "amplitude16", "amplitude64", "native16", "native64"])]
    axes = bottom.subplots(1, 5, sharey=True)
    for ax, task, label in zip(axes, TASKS, ["ECG", "EEG", "MNIST1D", "Pneumonia", "Nodule3D"]):
        for i, family in enumerate(FAMILY):
            row = selected[(selected.dataset == task) & (selected.family == family)].iloc[0]
            values = np.array(json.loads(row["values"])) * 100
            color = FAMILY[family][0]
            ax.scatter(i + np.linspace(-.13, .13, 5), values, s=7, color=color, zorder=3)
            ax.plot([i - .27, i + .27], [values.mean()] * 2, lw=1.6, color=color)
        ax.axhline(0, color="#77828a", ls=":", lw=.8)
        ax.set(xticks=[0, 1, 2], xticklabels=["Angle", "Ampl.", "QSM"], xlim=(-.5, 2.5))
        ax.set_title(label, fontsize=7)
        ax.tick_params(axis="x", labelsize=6.5, labelrotation=40)
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Trained minus initial\nprobe BA (pp)")
    bottom.suptitle("(b) Learned accessibility under the same probe", x=0.01, ha="left", fontsize=8)
    save(fig, "fig_detail_probes")


def planted():
    """Figure 4: function-class labels, kernel ceilings, monomial surrogates and trained models."""
    summary = pd.read_csv(SUMMARY / "planted_summary.csv")
    surface, blue, orange = "#fcfcfb", "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(WIDTH, 2.75), layout="constrained", facecolor=surface)
    style(ax, surface)
    ax.tick_params(labelsize=6.5)
    f = summary[summary.label.isin(["F2", "F4"])]
    rows = sorted({(r.dataset, r.grid, r.label) for r in f.itertuples()}, key=lambda t: (t[2], t[0], t[1]))
    series = (("amp1_ceiling", blue, "o", True, "one copy, kernel ceiling"),
              ("amp2_ceiling", orange, "o", True, "two copies, kernel ceiling"),
              ("surrogate2", blue, "D", True, "degree-2 monomials, logistic"),
              ("surrogate4", orange, "D", True, "degree-4 monomials, logistic"),
              ("amp1", blue, "o", False, "one copy, trained"),
              ("amp2", orange, "o", False, "two copies, trained"))
    for (arm, color, marker, filled, name), dy in zip(series, np.linspace(-0.3, 0.3, len(series))):
        sel = f[f.arm == arm]
        ys = [rows.index((r.dataset, r.grid, r.label)) + dy for r in sel.itertuples()]
        ax.errorbar(sel.test_bacc, ys, xerr=[sel.test_bacc - sel.ci_low, sel.ci_high - sel.test_bacc], fmt=marker,
                    color=color, mfc=color if filled else surface, mec=color, ms=4, elinewidth=1, capsize=0,
                    label=name)
    ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
    short = {"NoduleMNIST3D": "Nodule 3D", "PneumoniaMNIST": "Pneumonia"}
    ax.set_yticks(range(len(rows)), [f"{label}, {short[d]}, {int(g)} values"
                                     + ("\n(exploratory)" if label == "F4" and int(g) == 16 else "")
                                     for d, g, label in rows], fontsize=6.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("test balanced accuracy (mean over seeds, 95% CI)", fontsize=6.5, color=MUTED)
    ax.set_title("Function class: F2 is in the one-copy class, F4 needs two copies", fontsize=7.5, color=INK,
                 loc="left")
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=3, labelcolor=INK,
              handletextpad=0.3, columnspacing=1.0)
    save(fig, "fig_planted", facecolor=surface)


def regime_map():
    """Figure 5: median kernel drifts of the regime map against width."""
    summary = pd.read_csv(SUMMARY / "regime_summary.csv")
    families = [("angle", "Angle"), ("amplitude", "Amplitude"), ("qsm", "QSM")]
    metrics = [("drift_ntk", "Tangent-kernel drift"), ("drift_fid", "Fidelity-kernel drift")]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 2.9), layout="constrained", sharey="row")
    for col, (family, name) in enumerate(families):
        g = summary[summary.family == family]
        depths = sorted(g.depth.unique())
        colors = dict(zip(depths, DEPTH_RAMP[-len(depths):]))
        for row, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            for depth in depths:
                d = g[g.depth == depth].sort_values("qubits")
                ax.plot(d.qubits, d[metric], marker="o", ms=2.6, lw=1.1, color=colors[depth], label=f"{depth}")
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_xticks(sorted(g.qubits.unique()))
            ax.grid(color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED, labelsize=6.5)
            if row == 0:
                ax.set_title(name, fontsize=7.5, color=INK, loc="left")
                ax.legend(fontsize=6, title="depth", title_fontsize=6, frameon=False, ncol=2,
                          handlelength=1.2, columnspacing=0.8, loc="upper left")
            else:
                ax.set_xlabel("qubits", fontsize=7, color=MUTED)
            if col == 0:
                ax.set_ylabel(f"{label}\n(median, epoch 100)", fontsize=7, color=MUTED)
    save(fig, "fig_regime_map")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for draw, needs in ((ladder, "ladder_gaps.csv"), (detail_probes, "probe_contrasts.csv"),
                        (planted, "planted_summary.csv"), (regime_map, "regime_summary.csv")):
        if (SUMMARY / needs).exists():
            with plt.rc_context(RC):
                draw()
    with plt.rc_context({**RC, "font.size": 8, "mathtext.fontset": "cm"}):
        pipeline()
    print("figures in", OUT)
