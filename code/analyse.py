"""Turn saved runs into figures and summary tables.

Usage
-----
    python code/analyse.py --exp pilot

Reads results/<exp>/*.json and writes
    report/figures/<exp>_*.pdf          figures used in the report
    results/<exp>/summary_*.csv         numbers quoted in the report

All quantities are computed from the *true* (noise-free) objective, which equals
the simple regret because every problem has minimum value 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FLOOR = 1e-8  # regret floor so log10 is defined

# Okabe-Ito palette: distinguishable with common colour-vision deficiencies.
COLORS = {
    "random": "#999999", "cmaes": "#D55E00", "gp_ard": "#0072B2",
    "gp_iso": "#56B4E9", "turbo": "#009E73",
}
LINESTYLE = {"aligned": "-", "rotated": "--"}


def color(label: str) -> str:
    for k, c in COLORS.items():
        if label == k or label.startswith(k):
            return c
    return "#000000"


def load(exp: str) -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "results" / exp).glob("*.json")):
        if f.name == "environment.json":
            continue
        rec = json.loads(f.read_text())
        job, res = rec["job"], rec["result"]
        if "error" in res:
            print(f"warning: {f.name} failed, excluded:\n{res['error'].splitlines()[-1]}")
            continue
        y = np.asarray(res["y"])
        rows.append({
            **job["problem"],
            "method": job["method"].get("label", job["method"]["name"]),
            "seed": job["seed"],
            "budget": job["budget"],
            "best_so_far": np.minimum.accumulate(y),
            "final": float(y.min()),
            "wall_time_s": res["wall_time_s"],
            **{f"diag_{k}": v for k, v in res["diagnostics"].items()
               if np.isscalar(v)},
        })
    if not rows:
        raise SystemExit(f"no results found in results/{exp}")
    df = pd.DataFrame(rows)
    df["log_final"] = np.log10(np.maximum(df["final"], FLOOR))
    return df


def setting_key(r) -> str:
    return f"{r['base']}, D={r['D']}, d_e={r['d_e']}" + (f", eps={r['eps']:g}" if r["eps"] else "")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_convergence(df: pd.DataFrame, exp: str, out: Path) -> None:
    """Median and interquartile range of log10 best-so-far regret vs evaluations."""
    df = df.assign(setting=df.apply(setting_key, axis=1))
    # Order panels numerically (D=10, 30, 100), not alphabetically (10, 100, 30).
    order = (df.drop_duplicates("setting")
               .sort_values(["base", "D", "d_e", "eps"])["setting"])
    settings = order.tolist()
    ncol = min(3, len(settings))
    nrow = int(np.ceil(len(settings) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for ax, s in zip(axes.flat, settings):
        sub = df[df["setting"] == s]
        for (method, align), g in sub.groupby(["method", "alignment"]):
            curves = np.log10(np.maximum(np.vstack(g["best_so_far"].to_list()), FLOOR))
            t = np.arange(1, curves.shape[1] + 1)
            med = np.median(curves, axis=0)
            lo, hi = np.percentile(curves, [25, 75], axis=0)
            ax.plot(t, med, color=color(method), ls=LINESTYLE.get(align, "-"),
                    lw=1.6, label=f"{method} ({align})")
            ax.fill_between(t, lo, hi, color=color(method), alpha=0.12, lw=0)
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("objective evaluations")
        ax.set_ylabel("log10 best regret")
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(settings):]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.savefig(out / f"{exp}_convergence.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_final_vs_dimension(df: pd.DataFrame, exp: str, out: Path) -> None:
    """Final regret against ambient dimension, aligned (solid) vs rotated (dashed)."""
    if df["D"].nunique() < 2:
        return
    groups = list(df.groupby(["base", "d_e"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.3), squeeze=False)
    for ax, ((base, d_e), g) in zip(axes.flat, groups):
        for (method, align), h in g.groupby(["method", "alignment"]):
            stats = h.groupby("D")["log_final"].quantile([0.25, 0.5, 0.75]).unstack()
            ax.errorbar(stats.index, stats[0.5],
                        yerr=[stats[0.5] - stats[0.25], stats[0.75] - stats[0.5]],
                        color=color(method), ls=LINESTYLE.get(align, "-"), marker="o",
                        ms=3, capsize=2, label=f"{method} ({align})")
        ax.set_xscale("log")
        ax.set_title(f"{base}, d_e={d_e}", fontsize=9)
        ax.set_xlabel("ambient dimension D")
        ax.set_ylabel("log10 final regret (median, IQR)")
        ax.grid(alpha=0.3)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.savefig(out / f"{exp}_final_vs_D.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_ecdf(df: pd.DataFrame, exp: str, out: Path, n_targets: int = 31) -> None:
    """COCO-style runtime ECDF: fraction of (run, target) pairs solved vs evaluations.

    Targets are log-uniform between 10^2 and 10^-6 regret, pooled over all
    settings in the experiment, separately for each method and alignment.
    """
    targets = np.logspace(2, -6, n_targets)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for (method, align), g in df.groupby(["method", "alignment"]):
        curves = np.vstack(g["best_so_far"].to_list())  # runs x evals
        hit = curves[:, :, None] <= targets[None, None, :]
        frac = hit.mean(axis=(0, 2))
        ax.plot(np.arange(1, len(frac) + 1), frac, color=color(method),
                ls=LINESTYLE.get(align, "-"), label=f"{method} ({align})")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("fraction of (run, target) pairs")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"{exp}_ecdf.pdf")
    plt.close(fig)


def fig_time(df: pd.DataFrame, exp: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for method, g in df.groupby("method"):
        s = g.groupby("D")["wall_time_s"].median()
        ax.plot(s.index, s.values, marker="o", color=color(method), label=method)
    ax.set_yscale("log")
    ax.set_xlabel("ambient dimension D")
    ax.set_ylabel("median wall time per run (s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"{exp}_walltime.pdf")
    plt.close(fig)


def fig_rotation_effect(rot: pd.DataFrame, exp: str, out: Path) -> None:
    """Rotation effect (rotated minus aligned median log10 regret) with 95% CI.

    One panel per base function; within a panel, methods along the x axis and
    one marker per setting (D, d_e or eps, whichever varies). Values above the
    dashed zero line mean the method is worse on rotated problems.
    """
    if rot.empty:
        return
    vary = next((k for k in ("D", "d_e", "eps") if rot[k].nunique() > 1), "D")
    levels = sorted(rot[vary].unique())
    methods = [m for m in COLORS if m in set(rot["method"])] + \
              sorted(set(rot["method"]) - set(COLORS))
    bases = sorted(rot["base"].unique())
    fig, axes = plt.subplots(1, len(bases), figsize=(4.6 * len(bases), 3.3),
                             squeeze=False, sharey=True)
    markers = ["o", "s", "^", "D", "v"]
    width = 0.6
    for ax, base in zip(axes.flat, bases):
        sub = rot[rot["base"] == base]
        for j, lev in enumerate(levels):
            offset = (j - (len(levels) - 1) / 2) * width / max(len(levels), 1)
            for i, m in enumerate(methods):
                r = sub[(sub["method"] == m) & (sub[vary] == lev)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = r["rotated_minus_aligned"]
                ax.errorbar(i + offset, y,
                            yerr=[[y - r["ci_low"]], [r["ci_high"] - y]],
                            color=color(m), marker=markers[j % len(markers)],
                            ms=4, capsize=2, lw=1,
                            label=f"{vary}={lev:g}" if i == 0 else None)
        ax.axhline(0, color="black", ls="--", lw=0.8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_title(base, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.set_xlim(-0.6, len(methods) - 0.4)
        # Plain-language direction labels in the top and bottom corners.
        ax.text(0.015, 0.97, "↑ worse when rotated", transform=ax.transAxes,
                fontsize=7.5, color="0.3", va="top")
        ax.text(0.015, 0.03, "↓ better when rotated", transform=ax.transAxes,
                fontsize=7.5, color="0.3", va="bottom")
    # Symmetric y range so the zero line sits in the middle of every panel.
    lim = max(abs(v) for ax in axes.flat for v in ax.get_ylim())
    lim = max(lim, 1.0)
    for ax in axes.flat:
        ax.set_ylim(-lim, lim)
    axes.flat[0].set_ylabel("rotated $-$ aligned\nmedian log10 regret (95% CI)")
    # Right-hand axis: the same values as ratios of regret (rotated / aligned).
    right = axes.flat[-1].twinx()
    right.set_ylim(-lim, lim)
    ticks = [t for t in (-2, -1, 0, 1, 2) if abs(t) <= lim]
    names = {-2: "×1/100", -1: "×1/10", 0: "×1", 1: "×10", 2: "×100"}
    right.set_yticks(ticks)
    right.set_yticklabels([names[t] for t in ticks], fontsize=8)
    right.set_ylabel("regret when rotated\nrelative to aligned", fontsize=9)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=8,
               frameon=False, title=vary)
    fig.savefig(out / f"{exp}_rotation_effect.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def bootstrap_ci(x: np.ndarray, stat=np.median, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    boots = stat(x[rng.integers(len(x), size=(n_boot, len(x)))], axis=1)
    return float(stat(x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def table_rotation_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Median log10 final regret, rotated minus aligned, with a bootstrap 95% CI.

    Aligned and rotated instances are different problems, so this uses an
    unpaired bootstrap of the difference in medians.
    """
    rows = []
    rng = np.random.default_rng(1)
    for keys, g in df.groupby(["base", "D", "d_e", "eps", "method"]):
        a = g.loc[g["alignment"] == "aligned", "log_final"].to_numpy()
        r = g.loc[g["alignment"] == "rotated", "log_final"].to_numpy()
        if len(a) == 0 or len(r) == 0:
            continue
        boots = [np.median(rng.choice(r, len(r))) - np.median(rng.choice(a, len(a)))
                 for _ in range(5000)]
        rows.append(dict(zip(["base", "D", "d_e", "eps", "method"], keys),
                         median_aligned=np.median(a), median_rotated=np.median(r),
                         rotated_minus_aligned=np.median(r) - np.median(a),
                         ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5),
                         n_aligned=len(a), n_rotated=len(r)))
    return pd.DataFrame(rows)


def table_vs_reference(df: pd.DataFrame, reference: str = "cmaes") -> pd.DataFrame:
    """Paired comparison with a reference method on the same problem instance.

    For each seed the two methods face the identical instance and initial design,
    so the per-seed difference in log10 final regret is a paired observation.
    """
    if reference not in set(df["method"]):
        return pd.DataFrame()
    rows = []
    key = ["base", "D", "d_e", "eps", "alignment"]
    for keys, g in df.groupby(key):
        ref = g[g["method"] == reference].set_index("seed")["log_final"]
        for method, h in g.groupby("method"):
            if method == reference:
                continue
            other = h.set_index("seed")["log_final"]
            common = ref.index.intersection(other.index)
            if len(common) < 2:
                continue
            diff = (other.loc[common] - ref.loc[common]).to_numpy()
            med, lo, hi = bootstrap_ci(diff)
            rows.append(dict(zip(key, keys), method=method, reference=reference,
                             median_diff=med, ci_low=lo, ci_high=hi,
                             frac_better=float(np.mean(diff < 0)), n=len(common)))
    return pd.DataFrame(rows)


def table_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("diag_")]
    if not cols:
        return pd.DataFrame()
    return (df.groupby(["method", "base", "D", "d_e", "alignment"])[cols]
              .median(numeric_only=True).dropna(how="all").reset_index())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment name (folder in results/)")
    ap.add_argument("--reference", default="cmaes")
    args = ap.parse_args()

    df = load(args.exp)
    fig_dir = ROOT / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir = ROOT / "results" / args.exp

    fig_convergence(df, args.exp, fig_dir)
    fig_final_vs_dimension(df, args.exp, fig_dir)
    fig_ecdf(df, args.exp, fig_dir)
    fig_time(df, args.exp, fig_dir)

    summary = (df.groupby(["base", "D", "d_e", "eps", "alignment", "method"])
                 .agg(n=("seed", "count"), median_log_final=("log_final", "median"),
                      q25=("log_final", lambda s: s.quantile(0.25)),
                      q75=("log_final", lambda s: s.quantile(0.75)),
                      median_wall_time_s=("wall_time_s", "median"))
                 .reset_index())
    summary.to_csv(res_dir / "summary_final_regret.csv", index=False)
    rot = table_rotation_effect(df)
    rot.to_csv(res_dir / "summary_rotation_effect.csv", index=False)
    fig_rotation_effect(rot, args.exp, fig_dir)
    table_vs_reference(df, args.reference).to_csv(res_dir / "summary_vs_reference.csv", index=False)
    table_diagnostics(df).to_csv(res_dir / "summary_diagnostics.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nfigures -> {fig_dir}\ntables  -> {res_dir}")


if __name__ == "__main__":
    main()
