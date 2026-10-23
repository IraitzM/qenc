"""Datasets and the data draws of every experiment.

Archives are downloaded on first use into ``data/`` (set QENC_DATA to use another
folder) and checked against their MD5 checksums. The draws reproduce the archived
runs exactly: same seeds, same order of random calls, same per-sample normalization.
"""
from __future__ import annotations

import hashlib
import http.client
import os
import pickle
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split

DATA = Path(os.environ.get("QENC_DATA", Path(__file__).resolve().parents[2] / "data"))

MNIST1D_URL = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
EEG_URL = ("https://huggingface.co/datasets/ankislyakov/EpilepticSeizureRecognition/"
           "resolve/main/Epileptic%20Seizure%20Recognition.csv")
MEDMNIST_URL = "https://zenodo.org/records/10519652/files/{}.npz?download=1"
MITDB_URL = "https://physionet.org/files/mitdb/1.0.0/{}"
HLS_URL = "https://zenodo.org/records/15376628/files/{}.zip?download=1"
MD5 = {
    "mnist1d_data.pkl": "5bf44b67062b653caea0043bc236811a",
    "eeg_seizure_uci388.csv": "6fd90d64def591ae3c8de83bc8dc7348",
    "pneumoniamnist": "28209eda62fecd6e6a2d98b1501bb15f",
    "pneumoniamnist_64": "8f4eceb4ccffa70c672198ea285246c6",
    "pneumoniamnist_224": "d6a3c71de1b945ea11211b03746c1fe1",
    "nodulemnist3d": "8755a7e9e05a4d9ce80a24c3e7a256f3",
    "HS": "ac5a2390390f40e7abfe03fff7d73222",
    "LS": "b27336a3e53c42fcc3c4629e0c8f23d8",
}


def download(url: str, path: Path, md5: str | None = None, attempts: int = 3) -> Path:
    """Fetch ``url`` once into ``path``, retrying broken transfers, and check its MD5 when one is given."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    for attempt in range(attempts):
        try:
            urllib.request.urlretrieve(url, part)
            break
        except (OSError, http.client.HTTPException):  # dropped connections and short reads
            if attempt == attempts - 1:
                raise
    if md5 is not None and hashlib.md5(part.read_bytes()).hexdigest() != md5:
        part.unlink()
        raise RuntimeError(f"{url} does not match the archived checksum {md5}")
    part.rename(path)
    return path


def medmnist(name: str) -> Path:
    """A MedMNIST archive, e.g. ``pneumoniamnist`` or ``pneumoniamnist_64`` (Zenodo record 10519652)."""
    return download(MEDMNIST_URL.format(name), DATA / "medmnist" / f"{name}.npz", MD5[name])


# ---------------------------------------------------------------- normalizations

def znorm_signal(a: np.ndarray) -> np.ndarray:
    """Per-sample z-score along the last axis, as used for signals and the five-task images."""
    return (a - a.mean(axis=-1, keepdims=True)) / (a.std(axis=-1, keepdims=True) + 1e-8)


def znorm_flat(x: np.ndarray) -> np.ndarray:
    """Per-sample z-score over all pixels or voxels, as used for the spatial pilot."""
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    sd = x.std(1, keepdims=True)
    return (x - x.mean(1, keepdims=True)) / np.where(sd > 1e-12, sd, 1.)


def subsample(x, y, n_per_class, rng):
    """``n_per_class`` examples of every class, in random order."""
    idx = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        idx.append(idx_c[:n_per_class])
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return x[idx], y[idx]


# ---------------------------------------------------------------- signals

def mnist1d() -> dict:
    """The official MNIST1D release (Greydanus and Wang, 2024), length-40 sequences."""
    path = download(MNIST1D_URL, DATA / "mnist1d_data.pkl", MD5["mnist1d_data.pkl"])
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {"x_train": np.asarray(raw["x"], dtype=np.float64), "y_train": np.asarray(raw["y"], dtype=np.int64),
            "x_test": np.asarray(raw["x_test"], dtype=np.float64), "y_test": np.asarray(raw["y_test"], dtype=np.int64)}


ECG_TRAIN_RECORDS, ECG_TEST_RECORDS = ("106", "119", "200", "208"), ("203", "223")
MITDB_MD5 = {  # header, signal and annotation files of MIT-BIH release 1.0.0
    "106": ("6b10ebd69e6c620936a3538cd5eb1304", "1b8c2b8257b37998dfa494e51ffaf37c", "7fc08cf64136459e88ac183b27d557df"),
    "119": ("1ae3b1cafcbb6eb1943419f310a8ed8a", "229be04be2eb3401e863b35fc396ab76", "a2c794187aeccf451fd3f36847e373bd"),
    "200": ("c31bdb3ab7d16b4435d77822a335286b", "62df06a5f8916615183ef8fe8abba75d", "086f8d7d884def961e51d7e94ea2213f"),
    "208": ("1821f934ba767d5aa8b1f56ad30b2048", "6cf051b70b8948aaa1739ffd43d16160", "28ed7d9a6c20c847d2ddf4377386c31c"),
    "203": ("1433212c0925529947dc02a8b2d05583", "62914128ee4dc122ec4c6d2e1f11f547", "5e86a68eea5677d224bc2acbfd16270b"),
    "223": ("1b2828b84089260cc1207abe0b1ba06c", "db7b767dfd6507b7527fe9511b11b816", "47900bbcfa8985cdf51f5f6e2b2523e5"),
}


def _beats(record: str, max_per_class: int):
    """Length-40 windows centred on annotated Normal (0) and ventricular (1) beats, lead MLII, z-scored."""
    import wfdb
    for extension, md5 in zip(("hea", "dat", "atr"), MITDB_MD5[record]):
        download(MITDB_URL.format(f"{record}.{extension}"), DATA / "mitdb" / f"{record}.{extension}", md5)
    signal = wfdb.rdrecord(str(DATA / "mitdb" / record)).p_signal[:, 0]
    ann = wfdb.rdann(str(DATA / "mitdb" / record), "atr")
    beats, labels, counts = [], [], [0, 0]
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol not in ("N", "V"):
            continue
        lo, hi = sample - 20, sample + 20
        if lo < 0 or hi > len(signal):
            continue
        label = 0 if symbol == "N" else 1
        if counts[label] >= max_per_class:
            continue
        window = signal[lo:hi].astype(np.float64)
        beats.append((window - window.mean()) / (window.std() + 1e-8))
        labels.append(label)
        counts[label] += 1
    return np.asarray(beats), np.asarray(labels, dtype=np.int64)


def ecg_beats() -> dict:
    """MIT-BIH Normal against PVC beats with patient-disjoint train (4 records) and test (2 records)."""
    path = DATA / "ecg_beats.npz"
    if not path.exists():
        data = {}
        for split, records, cap in (("train", ECG_TRAIN_RECORDS, 400), ("test", ECG_TEST_RECORDS, 100)):
            xs, ys = zip(*(_beats(r, cap) for r in records))
            data[f"x_{split}"], data[f"y_{split}"] = np.concatenate(xs), np.concatenate(ys)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def eeg():
    """UCI Epileptic Seizure Recognition (Bonn recordings): raw windows, seizure labels, recording ids.

    The 11,500 rows are 500 recordings cut into 23 windows each. ``X21.V1.791`` names
    window 21 of recording ``V1.791``.
    """
    path = download(EEG_URL, DATA / "eeg_seizure_uci388.csv", MD5["eeg_seizure_uci388.csv"])
    df = pd.read_csv(path)
    x = df[[c for c in df.columns if c.startswith("X")]].to_numpy(dtype=np.float64)
    y = (df["y"].to_numpy(dtype=np.int64) == 1).astype(np.int64)
    recordings = df["Unnamed"].astype(str).str.replace(r"^X\d+\.", "", regex=True).to_numpy()
    return x, y, recordings


def eeg_grouped_split(seed: int):
    """Five-task EEG draw: recordings split 60/20/20 per class, then 150/50/100 windows per class.

    Returns the raw windows, labels, recording ids and the (fit, validation, test) row indices.
    """
    x, y, groups = eeg()
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for cls in (0, 1):
        recordings = np.unique(groups[y == cls])
        rng.shuffle(recordings)
        blocks = np.split(recordings, [int(.6 * len(recordings)), int(.8 * len(recordings))])
        for part, block, count in zip(parts, blocks, (150, 50, 100)):
            ix = np.where(np.isin(groups, block))[0]
            rng.shuffle(ix)
            part.extend(ix[:count].tolist())
    parts = [np.asarray(p)[rng.permutation(len(p))] for p in parts]
    assert all(not set(groups[a]) & set(groups[b]) for a, b in ((parts[0], parts[1]), (parts[0], parts[2]),
                                                                  (parts[1], parts[2])))
    return x, y, groups, parts


def eeg_split(seed: int, n_train: int = 200, n_test: int = 100, split: str = "recording",
              znorm: bool = True, shuffle: bool = True):
    """Train and test windows for the EEG controls, ``n_train``/``n_test`` per class.

    ``recording`` assigns whole recordings to one side before drawing windows; ``random``
    draws windows directly, so sibling windows of one recording fall on both sides.
    ``shuffle=False, znorm=False`` gives the raw draw of the amplitude control.
    """
    x, y, rec = eeg()
    rng = np.random.default_rng(seed)
    if split == "recording":
        train, test = [], []
        for cls in (1, 0):
            rows = np.where(y == cls)[0]
            recordings = np.unique(rec[rows])
            rng.shuffle(recordings)
            cut = min(max(int(round(n_train / (n_train + n_test) * len(recordings))), 1), len(recordings) - 1)
            train_pool = rows[np.isin(rec[rows], list(set(recordings[:cut])))]
            test_pool = rows[np.isin(rec[rows], list(set(recordings[cut:])))]
            rng.shuffle(train_pool)
            rng.shuffle(test_pool)
            train.append(train_pool[:n_train])
            test.append(test_pool[:n_test])
        train, test = np.concatenate(train), np.concatenate(test)
    else:
        pos, neg = rng.permutation(np.where(y == 1)[0]), rng.permutation(np.where(y == 0)[0])
        train = np.concatenate([pos[:n_train], neg[:n_train]])
        test = np.concatenate([pos[n_train:n_train + n_test], neg[n_train:n_train + n_test]])
    if shuffle:
        rng.shuffle(train)
        rng.shuffle(test)
    norm = znorm_signal if znorm else np.asarray
    return norm(x[train]), y[train], norm(x[test]), y[test], rec[train], rec[test]


# ---------------------------------------------------------------- MedMNIST

def _arrays(name: str):
    with np.load(medmnist(name)) as raw:
        return {s: (raw[f"{s}_images"], raw[f"{s}_labels"].ravel().astype(np.int64)) for s in ("train", "test")}


def _balanced(y, n_per_class, rng):
    idx = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        idx.append(idx_c[:n_per_class])
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


def pneumonia_pool(n_train: int = 300, n_test: int = 150, seed: int = 42):
    """Class-balanced PneumoniaMNIST subset in [0, 1] from which the five-task draws are taken."""
    arrays, rng, out = _arrays("pneumoniamnist"), np.random.default_rng(seed), []
    for split, n in (("train", n_train), ("test", n_test)):
        images, labels = arrays[split]
        idx = _balanced(labels, n, rng)
        out += [images[idx].astype(np.float64) / 255.0, labels[idx]]
    return out


def nodule_draw(seed: int, n_train: int = 150, n_test: int = 60):
    """Class-balanced NoduleMNIST3D volumes in [0, 1], flattened."""
    arrays, rng, out = _arrays("nodulemnist3d"), np.random.default_rng(seed), []
    for split, n in (("train", n_train), ("test", n_test)):
        images, labels = arrays[split]
        idx = _balanced(labels, n, rng)
        out += [(images[idx].astype(np.float64) / 255.0).reshape(len(idx), -1), labels[idx]]
    return out


def spatial_draw(name: str, seed: int) -> dict:
    """Spatial-pilot draw: 120 training and 60 test images or volumes per class, z-scored per example.

    The training part splits 180 fit / 60 validation, stratified.
    """
    rng = np.random.default_rng(seed)
    out = {}
    with np.load(medmnist(name.lower())) as raw:
        for split, count in (("train", 120), ("test", 60)):
            y = raw[f"{split}_labels"].ravel().astype(int)
            ix = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:count] for c in (0, 1)])
            ix = rng.permutation(ix)
            x = raw[f"{split}_images"][ix].astype(float) / 255.
            out[split] = (znorm_flat(x), y[ix], ix, x.shape[1:])
    (x, y, ids, shape), (xt, yt, test_ids, _) = out["train"], out["test"]
    fit, val = train_test_split(np.arange(len(y)), test_size=.25, random_state=seed, stratify=y)
    return dict(x=x, y=y, xt=xt, yt=yt, shape=shape, fit=fit, val=val, train_ids=ids, test_ids=test_ids)


# ---------------------------------------------------------------- HLS-CMDS

def hls_root() -> Path:
    """HLS-CMDS heart (HS) and lung (LS) recordings (Torabi et al., Zenodo record 15376628)."""
    root = DATA / "hls_cmds"
    for part in ("HS", "LS"):
        if not (root / part).exists():
            archive = download(HLS_URL.format(part), root / f"{part}.zip", MD5[part])
            with zipfile.ZipFile(archive) as z:
                z.extractall(root / part)
    return root


def hls_cmds(seed: int, representation: str = "waveform", target_len: int = 40, test_frac: float = 0.3):
    """Heart (0) against lung (1) windows, split by recording.

    Each recording gives 1 s windows (2 per heart and 4 per lung recording, evenly spaced),
    resampled to ``target_len`` (waveform) or reduced to the log power-spectral envelope (spectral).
    """
    root, rng = hls_root(), np.random.default_rng(seed)

    def recordings(category):
        found = []
        for folder, _, files in os.walk(root / category):
            if "__MACOSX" not in folder:
                found += [os.path.join(folder, f) for f in files
                          if f.lower().endswith(".wav") and not f.startswith("._")]
        return sorted(found)

    def windows(path, n_windows):
        sr, x = wavfile.read(path)
        x = x.astype(np.float64) / 32768.0  # 16-bit PCM scaled as soundfile does
        if x.ndim > 1:
            x = x.mean(axis=1)
        w = int(1.0 * sr)
        if len(x) < w:
            x = np.pad(x, (0, w - len(x)))
        out = []
        for start in np.linspace(0, len(x) - w, n_windows).astype(int):
            segment = x[start:start + w]
            if representation == "spectral":
                magnitude = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
                feature = np.log1p(np.maximum(resample_poly(magnitude, target_len, len(magnitude)), 0.0))
            else:
                feature = resample_poly(segment, target_len, w)
            feature = (feature[:target_len] if len(feature) >= target_len
                       else np.pad(feature, (0, target_len - len(feature))))
            out.append(feature.astype(np.float64))
        return out

    split = {"train": [], "test": []}
    for category, label, n_windows in (("HS", 0, 2), ("LS", 1, 4)):
        paths = recordings(category)
        rng.shuffle(paths)
        n_test = int(round(len(paths) * test_frac))
        for part, chosen in (("test", paths[:n_test]), ("train", paths[n_test:])):
            for path in chosen:
                split[part] += [(segment, label) for segment in windows(path, n_windows)]

    def pack(pairs):
        order = rng.permutation(len(pairs))
        return np.stack([pairs[i][0] for i in order]), np.array([pairs[i][1] for i in order], dtype=np.int64)

    return (*pack(split["train"]), *pack(split["test"]))


# ---------------------------------------------------------------- checkerboard

def checkerboard(n_train: int = 600, n_test: int = 300, overlap: float = 0.0, seed: int = 42):
    """The '2c' template of kernel-trainer: a 4x4 checkerboard of 16 ellipsoids in the unit cube.

    Vendored from kernel_trainer.dataset.DataGenerator v0.1.9 (MIT licence, I. Montalban).
    The original constructs the test generator before drawing the training set, and both
    seed Python's global ``random``, so the training draw starts from ``seed + 1`` and the
    test draw continues the same stream.
    """
    random.seed(seed + 1)
    return (*_two_c(n_train, overlap), *_two_c(n_test, overlap))


def _two_c(samples: int, overlap: float, radius: float = 0.15):
    n1 = int((samples * 0.5) / 8)
    n2 = int((samples - n1 * 8) / 8)
    base = np.linspace(radius, 1 - radius, 4)
    adjusted = 0.5 + (base - 0.5) * (1 - min(max(overlap, 0.0), 1.0))
    points, labels = [], []
    for iy, cy in enumerate(adjusted):
        for ix, cx in enumerate(adjusted):
            label = 1 if (ix + iy) % 2 == 0 else 0
            for _ in range(n1 if label == 1 else n2):
                r = np.cbrt(random.random())
                theta = np.arccos(1 - 2 * random.random())
                phi = 2 * np.pi * random.random()
                points.append([radius * r * np.sin(theta) * np.cos(phi) + cx,
                               radius * r * np.sin(theta) * np.sin(phi) + cy,
                               radius * r * np.cos(theta) + (1 - cx)])
                labels.append(label)
    return np.asarray(points, dtype=np.float64), np.asarray(labels, dtype=np.int64)
