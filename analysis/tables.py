"""Every table and number of the paper from the per-job results.

Reads results/<bundle>/*/ and writes results/summary/*.csv, plus the LaTeX bodies of the
SI tables in results/summary/tex/. Bundles that are missing are skipped, so a partial
rerun still summarizes what it has. Bootstrap intervals use the same seeds and draw
order as the paper.

    uv run analysis/tables.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from qenc.utils import RESULTS

SUMMARY, N_BOOT = RESULTS / "summary", 2000
TEX = SUMMARY / "tex"
TASKS = ["ECG", "EEG", "MNIST1D", "PneumoniaMNIST", "NoduleMNIST3D"]
LABEL = {"ECG": "ECG", "EEG": "EEG", "MNIST1D": "MNIST1D", "PneumoniaMNIST": "Pneumonia", "NoduleMNIST3D": "Nodule 3D"}
NAMES = {"angle": "Angle", "amplitude": "Amplitude", "qsm": "QSM"}


def jobs(name, marker="complete.json"):
    return sorted(p for p in (RESULTS / name).glob("*") if (p / marker).exists())


def concat(name, file, marker="complete.json", job=False):
    """All jobs' ``file`` of one bundle in folder order; ``job=True`` adds the folder name as ``job``."""
    folders = [f for f in jobs(name, marker) if (f / file).exists()]
    if not folders:
        return None
    return pd.concat([pd.read_csv(f / file).assign(**({"job": f.name} if job else {})) for f in folders],
                     ignore_index=True)


def split_job(df):
    """Dataset and seed from job names of the form DATASET-SEED."""
    return df.assign(dataset=df.job.str.rsplit("-", n=1).str[0], seed=df.job.str.rsplit("-", n=1).str[1].astype(int))


def save(df, name):
    df.to_csv(SUMMARY / f"{name}.csv", index=False)
    return df


def tabular(name, spec, header, rows):
    TEX.mkdir(parents=True, exist_ok=True)
    body = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", header, "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]
    (TEX / f"{name}.tex").write_text("\n".join(body) + "\n")


def mean_sd(values):
    return f"${100 * np.mean(values):.1f}\\pm{100 * np.std(values, ddof=1):.1f}$"


# ---------------------------------------------------------------- five-task study, volume follow-up, spatial pilot

def five_task():
    folders = jobs("five-task")
    if not folders:
        return
    rows, policies, quantum = [], [], []
    for folder in folders:
        dataset, seed = folder.name.rsplit("-", 1)
        rows.append(pd.read_csv(folder / "representation_scores.csv").assign(dataset=dataset, seed=int(seed)))
        if not (folder / "quantum_scores.csv").exists():
            continue
        q = pd.read_csv(folder / "quantum_scores.csv")
        quantum.append(q.assign(dataset=dataset, seed=int(seed)))
        choice = json.loads((folder / "encoding-selection.json").read_text())
        for policy, selected, expanded in [
                ("Historical thresholds", choice["legacy_selected"], False),
                ("Validation, common candidates", choice["common_selected"], False),
                ("Validation, additional image orientation", choice["selected"], True),
                ("QSM when feasible, amplitude otherwise", "qsm" if "qsm" in set(q.candidate) else "amplitude", False),
                ("Always angle", "angle", False), ("Always amplitude", "amplitude", False)]:
            pool = q if expanded else q[q.candidate != "angle4x2"]
            score = float(pool.loc[pool.candidate == selected, "test_bacc"].iloc[0])
            regret = float(pool.test_bacc.max()) - score
            policies.append(dict(dataset=dataset, seed=int(seed), policy=policy, selected=selected, test_bacc=score,
                                 regret=regret, hit=abs(regret) < 1e-12))
    representation = pd.concat(rows, ignore_index=True)
    full = representation[representation.representation == "full"][["dataset", "seed", "cv_bacc", "test_bacc"]]
    representation = representation.merge(full, on=["dataset", "seed"], suffixes=("", "_full"))
    representation["cv_gap"] = representation.cv_bacc_full - representation.cv_bacc
    representation["test_gap"] = representation.test_bacc_full - representation.test_bacc
    save(representation.groupby(["dataset", "representation"], sort=False).agg(
        n_features=("n_features", "first"), mean_cv=("cv_bacc", "mean"), mean_test=("test_bacc", "mean"),
        sd_test=("test_bacc", "std"), G_A=("cv_gap", "mean"), test_gap=("test_gap", "mean"),
        selections=("selected", "sum")).reset_index(), "five_task_pooling_screen")
    if quantum:
        q = pd.concat(quantum, ignore_index=True)
        save(q.groupby(["dataset", "candidate"], sort=False).test_bacc.agg(["mean", "std", "count"]).reset_index(),
             "five_task_outcomes")
        p = pd.DataFrame(policies)
        save(p.groupby("policy", sort=False).agg(runs=("seed", "size"), hits=("hit", "sum"),
                                                 mean_regret=("regret", "mean")).reset_index(), "validation_selection")


def volume_followup():
    v = concat("volume-followup", "scores.csv")
    if v is None:
        return
    v["test_regret"] = v.groupby("seed").test_bacc.transform("max") - v.test_bacc
    save(v.groupby("candidate", sort=False).agg(features=("features", "first"), qubits=("n_qubits", "first"),
                                                mean_test=("test_bacc", "mean"), sd_test=("test_bacc", "std"),
                                                selections=("selected", "sum")).reset_index(), "volume_followup")
    selected = v[v.selected]
    print(f"volume follow-up: selected pipeline test-best in {int((selected.test_regret.abs() < 1e-12).sum())} "
          f"of {len(selected)} runs")


def spatial_pilot():
    q, c = concat("spatial-pilot", "scores.csv"), concat("spatial-pilot", "classical.csv")
    if q is None:
        return
    save(q.groupby(["dataset", "candidate"], sort=False).test_bacc.agg(["mean", "std", "count"]).reset_index(),
         "spatial_outcomes")
    save(c.groupby(["dataset", "candidate"], sort=False).test_bacc.agg(["mean", "std", "count"]).reset_index(),
         "spatial_rbf")


def outcomes_table():
    """Table S3: complete-pipeline test BA of the three families (mean +- sample SD over five seeds)."""
    q, s = concat("five-task", "quantum_scores.csv", job=True), concat("spatial-pilot", "scores.csv")
    if q is None or s is None or not {"ECG", "EEG", "MNIST1D"} <= set(split_job(q).dataset):
        return
    q = split_job(q)
    lines = []
    for task in ("ECG", "EEG", "MNIST1D"):
        g = q[q.dataset == task]
        cells = [mean_sd(g[g.candidate == c].test_bacc) for c in ("angle", "amplitude", "qsm")]
        lines.append(f"{task} & 8 means / full signal & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    for task, grids in (("PneumoniaMNIST", (("Coarse, $2\\times4$", "8"), ("Fine, $4\\times4$", "16"))),
                        ("NoduleMNIST3D", (("Coarse, $2^3$", "8"), ("Fine, $4^3$", "64")))):
        g = s[s.dataset == task]
        for label, d in grids:
            angle = mean_sd(g[g.candidate == "angle8"].test_bacc) if d == "8" else "{Not tested}"
            lines.append(f"{task} & {label} & {angle} & {mean_sd(g[g.candidate == f'amplitude{d}'].test_bacc)} & "
                         f"{mean_sd(g[g.candidate == f'native{d}'].test_bacc)} \\\\")
    tabular("outcomes_table", "llccc", "Task & Supplied representation & Angle-8 & Amplitude & QSM \\\\", lines)


# ---------------------------------------------------------------- detail and probes

def detail():
    q = concat("detail", "quantum_scores.csv", job=True)
    if q is None:
        return
    q, rows = split_job(q), []
    for (dataset, family), g in q.groupby(["dataset", "family"], sort=False):
        wide = g.pivot(index="seed", columns="arm", values="test_bacc")
        diff = wide.F - wide.PF
        rows.append(dict(dataset=dataset, family=family, n=len(diff), mean=diff.mean(), sd=diff.std(ddof=1),
                         wins=int((diff > 0).sum()), ties=int((diff == 0).sum()), values=json.dumps(diff.tolist())))
    save(pd.DataFrame(rows), "detail_contrasts")


def probes():
    s = concat("probes", "scores.csv")
    if s is None:
        return
    rows = []
    for (dataset, candidate, family), g in s.groupby(["dataset", "candidate", "family"]):
        p = g.pivot(index="seed", columns="stage", values="test_bacc")
        diff = p.trained - p.initial
        rows.append(dict(dataset=dataset, candidate=candidate, family=family, n=len(diff),
                         initial_mean=p.initial.mean(), trained_mean=p.trained.mean(), input_mean=p.input.mean(),
                         mean=diff.mean(), sd=diff.std(ddof=1), wins=int((diff > 1e-12).sum()),
                         ties=int((diff.abs() <= 1e-12).sum()), values=json.dumps(diff.tolist())))
    contrasts = save(pd.DataFrame(rows), "probe_contrasts")
    lines = []
    for task in TASKS:
        for r in contrasts[contrasts.dataset == task].itertuples():
            change = 0.0 if abs(r.mean) < 1e-12 else 100 * r.mean
            lines.append(f"{task} & {r.candidate} & {100 * r.initial_mean:.2f} & {100 * r.trained_mean:.2f} & "
                         f"${change:+.2f}\\pm{100 * r.sd:.2f}$ & {r.wins}/5 & {r.ties} \\\\")
        lines.append("\\midrule")
    tabular("probe_table", "llrrrrr",
            "Task & Candidate & Initial & Trained & $\\Delta\\pm\\mathrm{SD}$ & Positive & Ties \\\\", lines[:-1])
    resources = (s[s.stage == "initial"].groupby(["dataset", "candidate"])
                 [["n_qubits", "n_features", "parameters"]].first())
    lines = []
    for task in TASKS:
        for candidate, r in resources.loc[task].iterrows():
            lines.append(f"{task} & {candidate} & {r.n_qubits} & {r.n_features} & {r.parameters} \\\\")
        lines.append("\\midrule")
    tabular("resource_table", "llrrr", "Task & Candidate & Qubits & Measured features & Parameters \\\\", lines[:-1])


# ---------------------------------------------------------------- ladder

GAPS = {  # name: (minuend, subtrahend); "L5" is the mean over the 20 random draws. Order fixes the bootstrap stream.
    "info_loss L0-L1": ("classical|L0|raw", "{c}|L1|input"),
    "state_vs_input L2tr-L1": ("{c}|L2|trained", "{c}|L1|input"),
    "readout_gap L3tr-L4tr": ("{c}|L3|trained", "{c}|L4|trained"),
    "unused_state L2tr-L4tr": ("{c}|L2|trained", "{c}|L4|trained"),
    "learn L4tr-L4init": ("{c}|L4|trained", "{c}|L4|initial"),
    "vs_random L4tr-L5": ("{c}|L4|trained", "L5"),
    "state_learn L2tr-L2init": ("{c}|L2|trained", "{c}|L2|initial"),
}
LADDER_ROWS = {("PneumoniaMNIST", "angle"): "angle8", ("PneumoniaMNIST", "amplitude"): "amplitude16",
               ("PneumoniaMNIST", "qsm"): "native16", ("NoduleMNIST3D", "angle"): "angle8",
               ("NoduleMNIST3D", "amplitude"): "amplitude64", ("NoduleMNIST3D", "qsm"): "native64"}


def _weighted_ba(correct, y, weights):
    per_class = []
    for cls in np.unique(y):
        mask = y == cls
        w = weights[:, mask]
        per_class.append((w @ correct[:, mask].T) / w.sum(1, keepdims=True))
    return np.mean(per_class, axis=0).mean(1)


def _ladder_bootstrap(preds, dataset, candidate, rng):
    """Seeds resampled with replacement, then test examples within each drawn seed; all gaps paired."""
    seeds = sorted(k for k in preds if k.startswith(dataset + "-"))
    n = len(seeds)
    pick = rng.integers(0, n, (N_BOOT, n))
    result = {}
    for gap, (a, b) in GAPS.items():
        point, boot = [], np.empty((n, n, N_BOOT))
        for si, s in enumerate(seeds):
            p, y = preds[s], preds[s]["yt"]

            def correct(key):
                keys = ([f"{candidate}|L5|random{k:02d}" for k in range(20)] if key == "L5"
                        else [key.format(c=candidate)])
                return np.stack([p[k] == y for k in keys]).astype(float)

            ca, cb, ones = correct(a), correct(b), np.ones((1, len(y)))
            point.append(_weighted_ba(ca, y, ones)[0] - _weighted_ba(cb, y, ones)[0])
            for slot in range(n):
                w = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=N_BOOT).astype(float)
                boot[si, slot] = _weighted_ba(ca, y, w) - _weighted_ba(cb, y, w)
        draws = boot[pick, np.arange(n)[None, :], np.arange(N_BOOT)[:, None]].mean(1)
        result[gap] = (float(np.mean(point)), *np.percentile(draws, [2.5, 97.5]))
    return result


def ladder():
    folders = jobs("ladder")
    if not folders:
        return
    s = pd.concat([pd.read_csv(f / "scores.csv") for f in folders], ignore_index=True)
    preds = {}
    for f in folders:
        with np.load(f / "predictions.npz") as z:
            preds[f.name] = {k: z[k] for k in z.files}
    t = s.copy()
    t["stage"] = t.stage.where(~t.stage.str.startswith("random"), "random")
    wide = (t.groupby(["dataset", "seed", "candidate", "family", "level", "stage"]).test_bacc.mean()
            .unstack(["level", "stage"]))
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    raw = t[t.level == "L0"].set_index(["dataset", "seed"]).test_bacc.rename("L0_raw")
    wide = wide.reset_index().merge(raw.reset_index(), on=["dataset", "seed"], suffixes=("_drop", ""))
    wide = wide.loc[:, ~wide.columns.str.endswith("_drop")]
    kta = t[t.level == "L2"].pivot_table(index=["dataset", "seed", "candidate"], columns="stage", values="kta")
    kta.columns = [f"kta_{c}" for c in kta.columns]
    wide = wide.merge(kta.reset_index(), on=["dataset", "seed", "candidate"])
    cols = [c for c in wide.columns if c not in ("dataset", "seed", "candidate", "family")]
    # Summaries are stored at four decimals, as in the paper, and the LaTeX table reads the stored values.
    summary = save(wide.groupby(["dataset", "candidate", "family"])[cols].mean().reset_index().round(4),
                   "ladder_levels")
    rng, gaps = np.random.default_rng(0), []
    for (d, c), _ in summary.groupby(["dataset", "candidate"]):
        for g, (pt, lo, hi) in _ladder_bootstrap(preds, d, c, rng).items():
            gaps.append(dict(dataset=d, candidate=c, gap=g, mean=pt, ci_low=lo, ci_high=hi))
    save(pd.DataFrame(gaps).round(4), "ladder_gaps")
    top = {}
    for (task, cand), g in s[s.level.isin(["L4", "L5"])].groupby(["dataset", "candidate"]):
        wins = sum(int(x[(x.level == "L4") & (x.stage == "trained")].test_bacc.iloc[0]
                       > x[x.level == "L5"].test_bacc.max()) for _, x in g.groupby("seed"))
        top[(task, cand)] = (wins, g.seed.nunique())
    rows, idx = [], summary.set_index(["dataset", "candidate"])
    for task in TASKS:
        for family in ("angle", "amplitude", "qsm"):
            cand = LADDER_ROWS.get((task, family), family)
            if (task, cand) not in idx.index:
                continue
            r = idx.loc[(task, cand)]
            levels = [100 * r[c] for c in ("L0_raw", "L1_input", "L2_trained", "L3_trained", "L4_trained",
                                           "L4_initial", "L5_random")]
            rows.append(f"{LABEL[task] if family == 'angle' else ''} & "
                        f"{'Angle-8' if family == 'angle' else NAMES[family]} & "
                        + " & ".join(f"{v:.1f}" for v in levels)
                        + f" & {r.kta_initial:.2f}$\\to${r.kta_trained:.2f} & "
                        f"{top[(task, cand)][0]}/{top[(task, cand)][1]} \\\\")
        rows.append("\\addlinespace")
    tabular("ladder_table", "llrrrrrrrcc",
            "Task & Family & $L_0$ & $L_1$ & $L_2$ & $L_3$ & $L_4$ & $L_4^{(0)}$ & $\\bar L_5$ & KTA & Top \\\\",
            rows[:-1])


# ---------------------------------------------------------------- planted labels

def _planted_predictions():
    bound = {k: min(1.0, 0.5 + 4 / (2 * k)) for k in (2, 4, 8, 16)}
    predictions = {"F2": {a: ("learnable", np.nan)
                          for a in ("amp1_ceiling", "amp2_ceiling", "surrogate2", "surrogate4")}}
    for k in (2, 4, 8, 16):
        if k <= 4:
            predictions[f"K{k}"] = {a: ("learnable", np.nan) for a in ("angle_ceiling", "fourier4", "fourier16")}
        else:
            predictions[f"K{k}"] = {**{a: ("bounded", bound[k]) for a in ("angle", "angle_ceiling", "fourier4")},
                                    "fourier16": ("learnable", np.nan)}
    predictions["F4"] = {}
    return predictions


def planted():
    folders = jobs("planted")
    if not folders:
        return
    runs = {}
    for f in folders:
        dataset, seed, label, *grid = f.name.split("-")
        with np.load(f / "predictions.npz") as z:
            runs[f.name] = dict(dataset=dataset, seed=int(seed), label=label, grid=int(grid[0][1:]) if grid else None,
                                preds={k: z[k] for k in z.files})
    rng, weights = np.random.default_rng(0), {}
    for j in runs.values():  # one weight matrix per (dataset, seed): every contrast is paired
        if (j["dataset"], j["seed"]) not in weights:
            n = len(j["preds"]["yt"])
            weights[(j["dataset"], j["seed"])] = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(float)
    seeds = sorted({j["seed"] for j in runs.values()})
    pick = rng.integers(0, len(seeds), (N_BOOT, len(seeds)))

    def bacc(y, p, w):
        return np.mean([(w[:, y == c] @ (p[y == c] == c)) / w[:, y == c].sum(1) for c in np.unique(y)], axis=0)

    def draws(per_seed):
        return np.stack([per_seed[s] for s in seeds])[pick, np.arange(N_BOOT)[:, None]].mean(1)

    groups = sorted({(j["dataset"], j["grid"], j["label"]) for j in runs.values()}, key=str)
    predicted, rows, values = _planted_predictions(), [], {}
    for dataset, grid, label in groups:
        members = [j for j in runs.values() if (j["dataset"], j["grid"], j["label"]) == (dataset, grid, label)]
        share = min(np.bincount(j["preds"]["yt"], minlength=2).min() / len(j["preds"]["yt"]) for j in members)
        for arm in sorted({a for j in members for a in j["preds"] if a != "yt"}):
            point, per_seed = [], {}
            for j in members:
                y, p = j["preds"]["yt"], j["preds"][arm]
                point.append(bacc(y, p, np.ones((1, len(y))))[0])
                per_seed[j["seed"]] = bacc(y, p, weights[(dataset, j["seed"])])
            values[(dataset, grid, label, arm)] = per_seed
            lo, hi = np.percentile(draws(per_seed), [2.5, 97.5])
            kind, limit = predicted[label].get(arm, ("", np.nan))
            verdict = ("degenerate label" if share < 0.05
                       else ("learned" if lo > 0.5 else "not learned") if kind == "learnable"
                       else ("within bound" if lo <= limit else "VIOLATED") if kind == "bounded" else "")
            rows.append(dict(dataset=dataset, grid=grid, label=label, arm=arm, test_bacc=np.mean(point), ci_low=lo,
                             ci_high=hi, predicted=kind, bound=limit, verdict=verdict))
    contrasts = []
    for name, a, b, f4_minus_f2 in (
            ("F4: amp2_ceiling - amp1_ceiling", "amp2_ceiling", "amp1_ceiling", False),
            ("F4: surrogate4 - surrogate2", "surrogate4", "surrogate2", False),
            ("F4 minus F2: amp2_ceiling - amp1_ceiling", "amp2_ceiling", "amp1_ceiling", True),
            ("F2: amp2_ceiling - amp1_ceiling (reference)", "amp2_ceiling", "amp1_ceiling", False),
            ("F2: surrogate4 - surrogate2 (reference)", "surrogate4", "surrogate2", False)):
        for dataset, grid in sorted({(d, g) for d, g, label in groups if label in ("F2", "F4")}, key=str):
            labels = ("F4", "F2") if f4_minus_f2 else ("F4",) if name.startswith("F4") else ("F2",)
            if not all((dataset, grid, label, arm) in values for label in labels for arm in (a, b)):
                continue
            gain = {label: {s: values[(dataset, grid, label, a)][s] - values[(dataset, grid, label, b)][s]
                            for s in seeds} for label in labels}
            per_seed = {s: gain["F4"][s] - gain["F2"][s] for s in seeds} if f4_minus_f2 else gain[labels[0]]
            lo, hi = np.percentile(draws(per_seed), [2.5, 97.5])
            predicted_cell = name.startswith("F4") and grid == 8
            contrasts.append(dict(contrast=name, dataset=dataset, grid=grid,
                                  estimate=np.mean([v.mean() for v in per_seed.values()]), ci_low=lo, ci_high=hi,
                                  predicted=("positive" if predicted_cell
                                             else "exploratory" if name.startswith("F4") else ""),
                                  verdict=("confirmed" if lo > 0 else "not confirmed") if predicted_cell else ""))
    summary = save(pd.DataFrame(rows).round(4), "planted_summary")
    contrasts = save(pd.DataFrame(contrasts).round(4), "planted_contrasts")
    verdicts = pd.concat([summary.verdict, contrasts.verdict])
    print("planted labels:", verdicts[verdicts != ""].value_counts().to_dict())


# ---------------------------------------------------------------- regime map

def regime_map():
    metrics = concat("regime-map", "metrics.csv", marker="metrics.csv")  # job, family, qubits, ... columns included
    if metrics is None:
        return
    lin = concat("regime-map", "linearized.csv", marker="linearized.csv")
    final = metrics[metrics.epoch == 100]
    single = metrics[(metrics.family == "amplitude") | (metrics.depth == 1)]
    verdicts = {"P1": {"max_single_upload_fidelity_drift": float(single.drift_fid.max()),
                       "runs": int(single.groupby(["job", "seed"]).ngroups)}}
    verdicts["P1"]["verdict"] = "holds" if verdicts["P1"]["max_single_upload_fidelity_drift"] < 1e-4 else "VIOLATED"
    p2 = {}
    for family, g in final.groupby("family"):
        holds = []
        for _, cell in g.groupby(["depth", "lr", "init", "task"]):
            med = cell.groupby("qubits").drift_ntk.median()
            if len(med) >= 2:  # tangent drift lower at the largest width than at the smallest
                holds.append(bool(med.iloc[-1] < med.iloc[0]))
        k, n = sum(holds), len(holds)
        p = float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue) if n else float("nan")
        p2[family] = {"cells": n, "cells_holding": k, "p_one_sided": p,
                      "verdict": "holds" if n and k / n > 0.5 and p < 0.05 else "does not hold"}
    verdicts["P2"] = p2
    keys = ["qubits", "depth", "lr", "init", "task", "seed"]
    sel = final.qubits.isin([4, 8]) & final.depth.isin([2, 4, 8])
    pairs = final[sel & (final.family == "angle")][keys + ["drift_fid"]].merge(
        final[sel & (final.family == "qsm")][keys + ["drift_fid"]], on=keys, suffixes=("_angle", "_qsm"))
    k, n = int((pairs.drift_fid_qsm > pairs.drift_fid_angle).sum()), len(pairs)
    p = float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue) if n else float("nan")
    verdicts["P3"] = {"pairs": n, "qsm_larger": k, "p_one_sided": p,
                      "median_ratio": (float(np.median(pairs.drift_fid_qsm / pairs.drift_fid_angle.clip(lower=1e-12)))
                                       if n else float("nan")),
                      "verdict": "holds" if n and k / n > 0.5 and p < 0.05 else "does not hold"}
    if lin is not None:
        runs = final.merge(lin[["job", "seed", "agreement", "ba_test_trained", "ba_test_linearized"]],
                           on=["job", "seed"])
        runs["disagreement"] = 1 - runs.agreement
        rng, unique, boots = np.random.default_rng(0), runs.job.unique(), []
        for _ in range(N_BOOT):  # resample whole jobs
            sample = pd.concat([runs[runs.job == j] for j in rng.choice(unique, len(unique))])
            boots.append(stats.spearmanr(sample.drift_ntk, sample.disagreement).statistic)
        lo, hi = (float(v) for v in np.nanpercentile(boots, [2.5, 97.5]))
        verdicts["P4"] = {"runs": len(runs),
                          "spearman": float(stats.spearmanr(runs.drift_ntk, runs.disagreement).statistic),
                          "ci": [lo, hi], "verdict": "holds" if lo > 0 else "does not hold"}
        gain = runs.ba_test_trained - runs.ba_test_linearized
        verdicts["exploratory"] = {
            "E2_init": {"|".join(map(str, k)): v
                        for k, v in final.groupby(["family", "init"]).drift_ntk.median().round(4).items()},
            "E5_richness_pays_spearman": float(stats.spearmanr(runs.drift_ntk, gain).statistic),
            "E5_median_gain_over_linearized": float(gain.median())}
    (SUMMARY / "regime_verdicts.json").write_text(json.dumps(verdicts, indent=2) + "\n")
    if "P4" in verdicts and set(p2) == {"angle", "amplitude", "qsm"}:  # the complete map
        v = verdicts
        rows = [("R1", "single-upload fidelity drift below $10^{-4}$",
                 f"largest drift ${v['P1']['max_single_upload_fidelity_drift']:.1e}$ over {v['P1']['runs']} runs"
                 .replace("e-05", "\\times10^{-5}"),
                 v["P1"]["verdict"]),
                ("R2", "tangent drift lower at largest width",
                 ", ".join(f"{NAMES[f]} {p2[f]['cells_holding']}/{p2[f]['cells']} ($p={p2[f]['p_one_sided']:.3f}$)"
                           for f in ("angle", "amplitude", "qsm")),
                 "holds" if all(p2[f]["verdict"] == "holds" for f in p2) else "does not hold"),
                ("R3", "QSM fidelity drift exceeds angle",
                 (f"{v['P3']['qsm_larger']}/{v['P3']['pairs']} pairs, median ratio {v['P3']['median_ratio']:.2f}, "
                  f"$p={v['P3']['p_one_sided']:.3f}$"),
                 v["P3"]["verdict"]),
                ("R4", "tangent drift tracks linearization gap",
                 (f"$\\rho={v['P4']['spearman']:.2f}$, 95\\% interval {v['P4']['ci'][0]:.2f} "
                  f"to {v['P4']['ci'][1]:.2f}, {v['P4']['runs']} runs"),
                 v["P4"]["verdict"])]
        tabular("regime_verdict_table", "p{0.06\\linewidth}p{0.27\\linewidth}p{0.45\\linewidth}p{0.12\\linewidth}",
                "& Prediction & Result & Verdict \\\\", [" & ".join(r) + " \\\\" for r in rows])
    summary = save(final.groupby(["family", "qubits", "depth"])
                   [["drift_ntk", "drift_fid", "drift_proj", "cka_features", "ba_test"]]
                   .median().reset_index().round(5), "regime_summary")
    rows = []
    for family in ("angle", "amplitude", "qsm"):
        for _, r in summary[summary.family == family].iterrows():
            rows.append(f"{NAMES[family]} & {r.qubits} & {r.depth} & {r.drift_ntk:.3f} & {r.drift_fid:.3f} & "
                        f"{r.drift_proj:.3f} & {r.cka_features:.3f} & {100 * r.ba_test:.1f} \\\\")
        rows.append("\\midrule")
    tabular("regime_summary_table", "lrrrrrrr",
            "Family & Qubits & Depth & Tangent drift & Fidelity drift & Projected drift & CKA & Test BA \\\\",
            rows[:-1])
    if (RESULTS / "regime-map" / "ntk_norm.csv").exists():
        g = (pd.read_csv(RESULTS / "regime-map" / "ntk_norm.csv").groupby(["family", "qubits", "depth", "init"])
             .k0_fro_per_probe.median().unstack("init"))
        rows = []
        for family in ("angle", "amplitude", "qsm"):
            for (f, n, d), r in g.loc[[family]].iterrows():
                rows.append(f"{NAMES[f]} & {n} & {d} & {r[0.1]:.3g} & {r[1.0]:.3g} \\\\")
            rows.append("\\midrule")
        tabular("regime_ntk_norm_table", "lrrrr", "Family & Qubits & Depth & init 0.1 & init 1.0 \\\\", rows[:-1])
    print("regime map: P1", verdicts["P1"]["verdict"], "| P2", {f: x["verdict"] for f, x in verdicts["P2"].items()},
          "| P3", verdicts["P3"]["verdict"], "| P4", verdicts.get("P4", {}).get("verdict"))


# ---------------------------------------------------------------- controls

def controls():
    w = RESULTS / "witnesses"
    if (w / "label_witness.csv").exists():
        save(pd.read_csv(w / "phase_witness.csv"), "witness_phase")
        save(pd.read_csv(w / "label_witness.csv").groupby(["dimensions", "label_rule", "representation"])
             .agg(projection_loss=("projection_loss", "mean"), test_bacc=("test_bacc", "mean")).reset_index(),
             "witness_label_rule")
    eeg = concat("eeg-controls", "scores.csv")
    if eeg is not None:
        save(eeg.groupby(["split", "features", "model"]).test_bacc.agg(["mean", "std", "count"]).reset_index(),
             "eeg_controls")
        qsm = eeg[eeg.model == "QSM"].pivot(index="seed", columns="split", values="test_bacc")
        diff = qsm.recording - qsm.random
        shared = np.mean([json.loads((f / "complete.json").read_text())["random_split_shared_recording"]
                          for f in jobs("eeg-controls")])
        print(f"EEG: QSM recording minus random split {diff.mean():+.3f} "
              f"+- {diff.std(ddof=1) / np.sqrt(len(diff)):.3f} (SE); "
              f"random split shares a recording for {100 * shared:.0f}% of test windows")
    hls = concat("hls", "scores.csv")
    if hls is not None:
        save(hls.groupby(["representation", "model"]).test_bacc.agg(["mean", "std", "count"]).reset_index(), "hls")
    fits = RESULTS / "rank" / "fits.csv"
    if fits.exists():
        df = pd.read_csv(fits)
        df["near_best"] = df.r2 >= df.groupby(["target_rank", "target_seed", "k"]).r2.transform("max") - 0.01
        best = df.groupby(["target_rank", "target_seed", "k"]).r2.max().groupby(["target_rank", "k"]).mean()
        table = save(df.groupby(["target_rank", "k"]).agg(ceiling=("ceiling", "mean"), reach=("near_best", "mean"))
                     .reset_index().assign(best_r2=lambda t: [best[(r, k)] for r, k in zip(t.target_rank, t.k)]),
                     "rank_restarts")
        rows = []
        for rank, g in table.groupby("target_rank"):
            cells = [f"{r.best_r2:.3f} ({r.ceiling:.3f}) [{100 * r.reach:.0f}\\%]"
                     for r in g.sort_values("k").itertuples()]
            rows.append(f"{rank} & {rank * (18 - rank)} & " + " & ".join(cells) + " \\\\")
        tabular("rank_table", "cccccc", "rank & d.o.f. & $k{=}0$ & $k{=}1$ & $k{=}2$ & $k{=}3$ \\\\", rows)
    if (RESULTS / "checkerboard" / "scores.csv").exists():
        save(pd.read_csv(RESULTS / "checkerboard" / "scores.csv"), "checkerboard")


if __name__ == "__main__":
    SUMMARY.mkdir(parents=True, exist_ok=True)
    for step in (five_task, volume_followup, spatial_pilot, outcomes_table, detail, probes, ladder, planted, regime_map,
                 controls):
        step()
    print("summaries in", SUMMARY)
