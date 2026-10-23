"""Regime map: how lazily the circuits train (main text Section 3.4, SI S12).

One class-balanced PneumoniaMNIST draw (seed 7: 1000 fit, 300 validation, the full
624-image test split) at MedMNIST+ sizes 28, 64 and 224. Each job trains one family,
width, depth, learning rate (0.005, 0.05), initialization scale (0.1, 1.0) and task
(real labels, or a same-architecture teacher) for 100 epochs (Adam, batch 64), over
five seeds (three at 16 qubits). At epochs 0, 5, 20 and 100 it records, on the first 256
test images, the relative drift ||K_t - K_0||_F / ||K_0||_F of the fidelity, projected
and tangent kernels (the tangent kernel on the first 128), their KTA, the CKA of the
features and the balanced accuracies. Core jobs (up to 12 qubits) also train the
first-order linearization around initialization. The QSM jobs at 10 qubits and the
16-qubit jobs need a GPU with about 24 GB or a large host memory.

``--ntk-norm`` runs the post hoc diagnostic of the initial tangent-kernel size.

    uv run experiments/regime_map.py --list | --job angle-q8-d4-lr0.05-init0.1-real | --all | --ntk-norm
"""
from __future__ import annotations

import copy
import sys

import pandas as pd
import torch

from qenc.fastsim import data, kernels, models, sim
from qenc.utils import bundle, cli

OUT = bundle("regime-map")
DRAW_SEED, N_FIT, N_VAL = 7, 1000, 300
EPOCHS, BATCH, CHECKPOINTS = 100, 64, (0, 5, 20, 100)
LRS, INITS, TASKS = (0.005, 0.05), (0.1, 1.0), ("real", "teacher")
SEEDS = {"core": (0, 1, 2, 3, 4), "scale": (0, 1, 2)}
PROBE, NTK_PROBE = 256, 128
TEACHER_SEED, TEACHER_INIT = 10_000, 1.0
# family -> qubits -> (image size, grid of block means)
GEOMETRY = {
    "angle": {4: (28, (2, 2)), 8: (28, (2, 4)), 16: (28, (4, 4))},
    "amplitude": {4: (28, (4, 4)), 8: (64, (16, 16)), 12: (64, (64, 64)), 16: (224, (224, 224))},
    "qsm": {4: (28, (4, 4)), 6: (64, (8, 8)), 8: (64, (16, 16)), 10: (64, (32, 32))},
}
ARMS = {  # family -> (qubits, depths): uploads for angle and QSM, trainable layers for amplitude
    "core": {"angle": ([4, 8], [1, 2, 4, 8, 16]), "amplitude": ([4, 8, 12], [1, 3, 9]),
             "qsm": ([4, 6, 8, 10], [1, 2, 4, 8])},
    "scale": {"angle": ([16], [1, 4, 8]), "amplitude": ([16], [1, 3])},
}


def job_name(family, n, depth, lr, init, task):
    return f"{family}-q{n}-d{depth}-lr{lr}-init{init}-{task}"


ARM_OF = {job_name(f, n, d, lr, i, t): arm for arm, families in ARMS.items() for f, (qubits, depths) in families.items()
          for n in qubits for d in depths for lr in LRS for i in INITS for t in TASKS}
JOBS = list(ARM_OF)


def parse(job):
    family, q, d, lr, init, task = job.split("-")
    return family, int(q[1:]), int(d[1:]), float(lr[2:]), float(init[4:]), task


def build(family, n, depth, init, seed, device):
    torch.manual_seed(seed)
    if family == "angle":
        model = models.AngleReupload(n, depth, readout="z+zz", init_scale=init)
    elif family == "amplitude":
        grid = GEOMETRY["amplitude"][n][1]
        model = models.Amplitude(grid[0] * grid[1], n_layers=depth, init_scale=init)
    else:
        model = models.QSM(n, depth, init_scale=init, dtype=torch.float32)
    return model.to(device)


class Inputs:
    """One split's inputs, on the device or, for 10-qubit QSM uploads, in pinned host memory."""

    def __init__(self, x, device, resident):
        self.device, self.resident = device, resident
        self.x = x.to(device) if resident else (x.pin_memory() if torch.cuda.is_available() else x)

    def take(self, ix):
        return self.x[ix if self.resident else ix.cpu()].to(self.device, non_blocking=True)

    def batches(self, size=256):
        for start in range(0, len(self.x), size):
            yield self.take(torch.arange(start, min(start + size, len(self.x)), device=self.x.device))


def prepare(family, n, device):
    size, grid = GEOMETRY[family][n]
    images = data.load_images(size, data.draw_indices(DRAW_SEED, N_FIT, N_VAL))
    resident = not (family == "qsm" and n >= 10)
    return {split: (Inputs(data.representation(family, img, grid, device=device), device, resident), y.to(device))
            for split, (img, y) in images.items()}


@torch.no_grad()
def logits(model, inputs):
    return torch.cat([model(x) for x in inputs.batches()])


def teacher_labels(family, n, depth, splits, device):
    """Labels of a same-architecture teacher: its logit margin above the fitting median."""
    teacher = build(family, n, depth, TEACHER_INIT, TEACHER_SEED, device)
    margins = {}
    for split, (inputs, _) in splits.items():
        z = logits(teacher, inputs)
        margins[split] = z[:, 1] - z[:, 0]
    threshold = margins["fit"].median()
    return {split: (inputs, (margins[split] > threshold).long()) for split, (inputs, _) in splits.items()}


def bacc(y, pred):
    return float(torch.stack([(pred[y == c] == c).float().mean() for c in torch.unique(y)]).mean())


@torch.no_grad()
def probe_state(model, x_probe):
    psi = torch.cat([model.state(x) for x in x_probe.split(64)])
    return psi, sim.rdm_1q(psi, model.n_qubits), torch.cat([model.features(x) for x in x_probe.split(64)])


def measure(model, x_probe, x_ntk, y_probe, y_ntk, ref, gamma):
    psi, rdm, feats = probe_state(model, x_probe)
    k_fid = kernels.fidelity_kernel(psi).double()
    k_proj = kernels.projected_kernel(rdm, gamma=gamma).double()
    k_ntk = kernels.tangent_kernel(model, x_ntk, names=["weights"], chunk=16 if model.n_qubits <= 12 else 4).double()
    w = model.weights.detach().double()
    row = {"kta_fid": kernels.kta(k_fid, y_probe), "kta_proj": kernels.kta(k_proj, y_probe),
           "kta_ntk": kernels.kta(k_ntk, y_ntk)}
    if ref is None:
        ref = {"fid": k_fid, "proj": k_proj, "ntk": k_ntk, "feats": feats.double(), "w": w}
    row.update(drift_fid=kernels.relative_change(ref["fid"], k_fid),
               drift_proj=kernels.relative_change(ref["proj"], k_proj),
               drift_ntk=kernels.relative_change(ref["ntk"], k_ntk),
               cka_features=kernels.linear_cka(ref["feats"], feats.double()),
               param_displacement=float((w - ref["w"]).norm() / ref["w"].norm()))
    return row, ref


def projected_gamma(model, x_probe):
    """1 / median squared distance between one-qubit reduced states at initialization."""
    _, rdm, _ = probe_state(model, x_probe)
    v = rdm.reshape(len(rdm), -1)
    sq = ((v[:, None] - v[None]).abs() ** 2).sum(-1)
    return float(1.0 / sq[~torch.eye(len(v), dtype=torch.bool, device=sq.device)].median().clamp_min(1e-12))


def train(model, splits, lr, seed, on_checkpoint):
    fit_inputs, y = splits["fit"]
    generator, opt = torch.Generator().manual_seed(seed), torch.optim.Adam(model.parameters(), lr=lr)
    on_checkpoint(0, float("nan"))
    for epoch in range(1, EPOCHS + 1):
        order, total = torch.randperm(len(y), generator=generator), 0.0
        for start in range(0, len(y), BATCH):
            ix = order[start:start + BATCH].to(y.device)
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(fit_inputs.take(ix)), y[ix])
            loss.backward()
            opt.step()
            total += loss.item() * len(ix)
        if epoch in CHECKPOINTS:
            on_checkpoint(epoch, total / len(y))


def run(job):
    folder = OUT / job
    if (folder / "metrics.csv").exists():
        return
    folder.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    family, n, depth, lr, init, task = parse(job)
    arm = ARM_OF[job]
    splits = prepare(family, n, device)
    if task == "teacher":
        splits = teacher_labels(family, n, depth, splits, device)
    test_inputs, y_test = splits["test"]
    x_probe = test_inputs.take(torch.arange(PROBE).to(test_inputs.x.device))
    x_ntk, y_probe, y_ntk = x_probe[:NTK_PROBE], y_test[:PROBE], y_test[:NTK_PROBE]
    rows, lin_rows = [], []
    for seed in SEEDS[arm]:
        model = build(family, n, depth, init, seed, device)
        initial, gamma, state = copy.deepcopy(model), projected_gamma(model, x_probe), {"ref": None}

        def on_checkpoint(epoch, loss):
            preds = {s: logits(model, inputs).argmax(1) for s, (inputs, _) in splits.items()}
            row, state["ref"] = measure(model, x_probe, x_ntk, y_probe, y_ntk, state["ref"], gamma)
            rows.append(dict(row, seed=seed, epoch=epoch, loss_fit=loss,
                             **{f"ba_{s}": bacc(y, preds[s]) for s, (_, y) in splits.items()}))

        train(model, splits, lr, seed, on_checkpoint)
        if arm == "core":
            lin = kernels.Linearized(initial, output="logits")
            trained_pred = logits(model, test_inputs).argmax(1)
            train(lin, splits, lr, seed, lambda e, l: None)
            lin_pred = logits(lin, test_inputs).argmax(1)
            lin_rows.append({"seed": seed, "ba_test_trained": bacc(y_test, trained_pred),
                             "ba_test_linearized": bacc(y_test, lin_pred),
                             "agreement": float((trained_pred == lin_pred).float().mean())})
        print(job, "seed", seed, f"test {rows[-1]['ba_test']:.3f} tangent drift {rows[-1]['drift_ntk']:.3f}",
              flush=True)
    config = dict(job=job, arm=arm, family=family, qubits=n, depth=depth, lr=lr, init=init, task=task)
    if lin_rows:
        pd.DataFrame(lin_rows).assign(**config).to_csv(folder / "linearized.csv", index=False)
    pd.DataFrame(rows).assign(**config).to_csv(folder / "metrics.csv", index=False)


NTK_CELLS = {"angle": [(4, (1, 4, 8)), (8, (1, 4, 8)), (16, (1, 4, 8))],
             "amplitude": [(4, (1, 3)), (8, (1, 3)), (12, (1, 3)), (16, (1, 3))],
             "qsm": [(4, (2, 8)), (6, (2, 8)), (8, (2, 8))]}


def ntk_norm():
    """Size of the initial tangent kernel per probe, ||K_0||_F / 128, seeds 0-2, both initialization scales."""
    if (OUT / "ntk_norm.csv").exists():
        print(OUT / "ntk_norm.csv", "exists, skipping")
        return
    device, rows = torch.device("cuda" if torch.cuda.is_available() else "cpu"), []
    for family, sizes in NTK_CELLS.items():
        for n, depths in sizes:
            test = prepare(family, n, device)["test"][0]
            x = test.take(torch.arange(NTK_PROBE).to(test.x.device))
            for depth in depths:
                for init in INITS:
                    for seed in (0, 1, 2):
                        model = build(family, n, depth, init, seed, device)
                        k = kernels.tangent_kernel(model, x, names=["weights"], chunk=16 if n <= 12 else 4).double()
                        rows.append(dict(family=family, qubits=n, depth=depth, init=init, seed=seed,
                                         k0_fro_per_probe=float(torch.linalg.matrix_norm(k)) / len(x)))
                        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "ntk_norm.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1:] == ["--ntk-norm"]:
        ntk_norm()
    else:
        cli(JOBS, run, __doc__)
