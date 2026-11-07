"""Follow-up checks on unexpected results, computed from saved runs.

Usage
-----
    python code/investigate.py

Writes results/investigations.txt with:

1. TuRBO lengthscale capping. For aligned problems in main_dimension, how often
   one of the two truly relevant inputs ends with its ARD lengthscale at TuRBO's
   upper bound (2.0, as in Eriksson et al., 2019), i.e. indistinguishable from an
   irrelevant input. Compared with GP-BO (ARD), whose upper bound is 100.
2. Box geometry. For x uniform on the box [-1, 1]^D, the distribution of one
   relevant coordinate z = a^T x is uniform on [-1, 1] when aligned, but close to
   Gaussian (same variance 1/3) when rotated, so more of its mass lies near the
   centre of the box. Reported: P(|z| < 0.25) under each alignment.
3. CMA-ES search shape after 3000 evaluations (followup_cmaes_long): fraction of
   the search covariance's variance lying in the relevant subspace, aligned vs
   rotated. Exact rotation invariance would make the two distributions equal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problems import EmbeddedProblem  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def lengthscale_capping(lines: list[str]) -> None:
    lines.append("1. Relevant input with lengthscale at the method's upper bound "
                 "(aligned problems, main_dimension)")
    for method, cap in (("turbo", 2.0), ("gp_ard", 100.0)):
        for base in ("ackley", "rosenbrock"):
            hits, total = 0, 0
            for D in (10, 30, 100):
                for s in range(15):
                    f = RES / "main_dimension" / f"{method}__{base}_D{D}_de2_aligned_eps0_s{s}.json"
                    if not f.exists():
                        continue
                    ls = np.array(json.loads(f.read_text())["result"]["diagnostics"]["lengthscales"])
                    rc = EmbeddedProblem(base, D, 2, "aligned", s).relevant_coords
                    hits += int(np.max(ls[rc]) >= 0.995 * cap)
                    total += 1
            lines.append(f"   {method:7s} {base:11s} {hits:3d}/{total} runs")


def box_geometry(lines: list[str]) -> None:
    lines.append("2. P(|z| < 0.25) for one relevant coordinate z = a^T x, x uniform on [-1,1]^D")
    rng = np.random.default_rng(0)
    for D in (10, 30, 100):
        p = EmbeddedProblem("sphere", D, 2, "rotated", 0)
        z = rng.uniform(-1.0, 1.0, size=(200_000, D)) @ p.A[:, 0]
        lines.append(f"   D={D:3d}: aligned 0.250 (uniform), rotated {np.mean(np.abs(z) < 0.25):.3f} "
                     f"(variance {z.var():.3f} vs 1/3)")


def cmaes_shape(lines: list[str]) -> None:
    lines.append("3. CMA-ES relevant-variance fraction after 3000 evaluations "
                 "(median over seeds; isotropic value = d_e/D)")
    for base in ("ackley", "rosenbrock"):
        for D in (10, 30, 100):
            vals = {}
            for al in ("aligned", "rotated"):
                fr = []
                for s in range(15):
                    f = RES / "followup_cmaes_long" / f"cmaes__{base}_D{D}_de2_{al}_eps0_s{s}.json"
                    if f.exists():
                        fr.append(json.loads(f.read_text())["result"]["diagnostics"]["relevant_variance_fraction"])
                vals[al] = np.median(fr) if fr else float("nan")
            lines.append(f"   {base:11s} D={D:3d}: aligned {vals['aligned']:.4f}, "
                         f"rotated {vals['rotated']:.4f}, isotropic {2 / D:.4f}")


def main() -> None:
    lines: list[str] = []
    lengthscale_capping(lines)
    box_geometry(lines)
    cmaes_shape(lines)
    text = "\n".join(lines) + "\n"
    (RES / "investigations.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
