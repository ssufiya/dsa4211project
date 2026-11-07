"""Run a grid of (problem, method, seed) experiments described by a YAML config.

Usage
-----
    python code/run_experiments.py --config configs/pilot.yaml --workers 4

Each run writes one JSON file to results/<experiment name>/. Existing files are
skipped, so an interrupted experiment can be resumed by re-running the command.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

# One BLAS thread per process: keeps wall-clock timings comparable and avoids
# oversubscription when running several workers.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods import METHODS  # noqa: E402
from problems import make_problem  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def expand_grid(cfg: dict) -> list[dict]:
    p = cfg["problems"]
    keys = ["base", "D", "d_e", "alignment", "eps"]
    vals = [p.get(k, [0.0] if k == "eps" else None) for k in keys]
    jobs = []
    for combo in itertools.product(*vals):
        prob = dict(zip(keys, combo))
        if prob["d_e"] > prob["D"]:
            continue
        for m in cfg["methods"]:
            for seed in cfg["seeds"]:
                jobs.append({"problem": prob, "method": dict(m), "seed": int(seed),
                             "budget": int(cfg["budget"]), "n_init": int(cfg["n_init"])})
    return jobs


def run_id(job: dict) -> str:
    p, m = job["problem"], job["method"]
    label = m.get("label", m["name"])
    return (f"{label}__{p['base']}_D{p['D']}_de{p['d_e']}_{p['alignment']}"
            f"_eps{p['eps']:g}_s{job['seed']}")


def run_one(args) -> str:
    job, out_dir = args
    out = Path(out_dir) / f"{run_id(job)}.json"
    if out.exists():
        return f"skip {out.name}"
    m = dict(job["method"])
    name = m.pop("name")
    m.pop("label", None)
    problem = make_problem(job["problem"], job["seed"])
    try:
        res = METHODS[name](problem, job["budget"], job["n_init"], job["seed"], **m)
    except Exception:  # record failures instead of crashing the whole grid
        res = {"error": traceback.format_exc()}
    res["n_evals_counted_by_problem"] = problem.n_evals
    record = {"job": job, "run_id": run_id(job), "result": res}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(out)
    status = "ERROR" if "error" in res else f"best={min(res['y']):.3g}"
    return f"done {out.name} {status}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="list jobs without running")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = ROOT / "results" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = expand_grid(cfg)
    print(f"{len(jobs)} runs -> {out_dir}")
    if args.dry_run:
        for j in jobs:
            print(run_id(j))
        return

    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out_dir / "environment.json").write_text(json.dumps({
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workers": args.workers,
    }, indent=2))

    t0 = time.time()
    work = [(j, str(out_dir)) for j in jobs]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, msg in enumerate(pool.imap_unordered(run_one, work), 1):
                print(f"[{i}/{len(work)}] {msg}", flush=True)
    else:
        for i, w in enumerate(work, 1):
            print(f"[{i}/{len(work)}] {run_one(w)}", flush=True)
    print(f"finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
