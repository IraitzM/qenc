"""Result paths, small file helpers and the command line shared by the experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("QENC_RESULTS", ROOT / "results"))


def bundle(name: str) -> Path:
    """Folder of one result bundle, created on demand."""
    path = RESULTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def digest(a: np.ndarray) -> str:
    """SHA-256 of an array's bytes, used to check that a data draw reproduces the archived one."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def cli(jobs: list[str], run: Callable[[str], None], doc: str | None = None) -> None:
    """``--list`` prints the job names, ``--job NAME`` runs one (repeatable) and ``--all`` runs every job.

    A job whose folder holds ``complete.json`` is skipped, so an interrupted ``--all`` resumes.
    """
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="print the job names")
    parser.add_argument("--job", action="append", choices=jobs, metavar="NAME", help="run one job")
    parser.add_argument("--all", action="store_true", help="run every job in order")
    args = parser.parse_args()
    if args.list:
        print("\n".join(jobs))
    elif args.all or args.job:
        for job in jobs if args.all else args.job:
            run(job)
    else:
        parser.error("use --list, --job NAME or --all")
