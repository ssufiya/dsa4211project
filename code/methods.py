"""Optimisers compared in the project.

All methods minimise a problem on the box [-1, 1]^D under the same evaluation
budget and return the full sequence of evaluated objective values, so the
runner can compute best-so-far curves in terms of evaluations. Every method
evaluates the *same* Latin-hypercube initial design of `n_init` points for a
given seed, and those evaluations count towards the budget.

Methods
-------
random     Latin-hypercube initial design followed by uniform random sampling.
cmaes      CMA-ES (pycma) started from the best initial point (Hansen, 2016).
gp_ard     GP-BO with a Matern-5/2 kernel and one lengthscale per input (ARD),
           Expected Improvement acquisition (Frazier, 2018).
gp_iso     GP-BO with a single shared lengthscale (isotropic kernel).
turbo      TuRBO-1: BO inside a single trust region whose side lengths follow the
           GP's ARD lengthscales, with Thompson sampling (Eriksson et al., 2019).

GP-BO details (gp_*): scikit-learn Gaussian process with normalised outputs,
hyperparameters fitted by marginal likelihood, Expected Improvement evaluated in
log space for numerical stability, and the acquisition maximised over a
candidate set (uniform points plus perturbations of the best observed points).
Candidate-set maximisation is a simplification of the multi-start gradient
optimisation used by libraries such as BoTorch; this is a known limitation and
should be stated in the report.
"""

from __future__ import annotations

import time
import warnings

import cma
import numpy as np
from scipy.special import log_ndtr
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def initial_design(n: int, dim: int, seed: int) -> np.ndarray:
    """Scrambled Latin hypercube on [-1, 1]^dim, shared across methods per seed."""
    sampler = qmc.LatinHypercube(d=dim, seed=np.random.default_rng([seed, dim, 31]))
    return 2.0 * sampler.random(n) - 1.0


def log_expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
    """log EI for minimisation, stable when EI underflows (cf. Ament et al., 2023)."""
    sigma = np.maximum(sigma, 1e-12)
    z = (best - mu) / sigma
    out = np.empty_like(z)
    # h(z) = z * Phi(z) + phi(z), EI = sigma * h(z)
    hi = z > -5.0
    zh = z[hi]
    h = zh * np.exp(log_ndtr(zh)) + np.exp(-0.5 * zh ** 2) / np.sqrt(2 * np.pi)
    out[hi] = np.log(np.maximum(h, 1e-300))
    zl = z[~hi]
    # asymptotic expansion for very negative z: h(z) ~ phi(z) / z^2
    out[~hi] = -0.5 * zl ** 2 - 0.5 * np.log(2 * np.pi) - 2.0 * np.log(-zl)
    return np.log(sigma) + out


def _result(ys, start, diagnostics, x_best=None):
    return {
        "y": [float(v) for v in ys],
        "wall_time_s": time.perf_counter() - start,
        "x_best": None if x_best is None else [float(v) for v in x_best],
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------

def run_random(problem, budget: int, n_init: int, seed: int, **_) -> dict:
    start = time.perf_counter()
    rng = np.random.default_rng([seed, 101])
    X0 = initial_design(n_init, problem.D, seed)
    ys = [problem(x) for x in X0]
    xs = list(X0)
    while len(ys) < budget:
        x = rng.uniform(-1.0, 1.0, size=problem.D)
        xs.append(x)
        ys.append(problem(x))
    return _result(ys, start, {}, xs[int(np.argmin(ys))])


# ---------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------

def _cma_diagnostics(es, problem) -> dict:
    """How much of the search distribution's variance lies in the relevant subspace.

    For an isotropic distribution the fraction tr(A^T C A) / tr(C) equals d_e / D.
    Values below d_e / D mean the distribution has contracted in the relevant
    directions relative to the irrelevant ones (expected near convergence, since
    only relevant directions receive selection pressure).
    """
    # pycma factors the search covariance as diag(s) C diag(s), where s is a
    # coordinate-wise scaling and C the sampler's full covariance matrix.
    D = problem.D
    s = np.broadcast_to(np.asarray(getattr(es.sigma_vec, "scaling", 1.0), float), (D,))
    C_full = getattr(es.sm, "covariance_matrix", None)
    C_full = np.eye(D) if C_full is None else np.asarray(C_full)
    C = s[:, None] * C_full * s[None, :]
    A = problem.A
    frac = float(np.trace(A.T @ C @ A) / np.trace(C))
    eig = np.sort(np.linalg.eigvalsh(C))[::-1]
    return {
        "final_sigma": float(es.sigma),
        "relevant_variance_fraction": frac,
        "isotropic_reference": problem.d_e / problem.D,
        "cov_eigenvalues": [float(v) for v in eig],
        "condition_number": float(eig[0] / max(eig[-1], 1e-300)),
    }


def run_cmaes(problem, budget: int, n_init: int, seed: int, sigma0: float = 0.5,
              popsize: int | None = None, **_) -> dict:
    start = time.perf_counter()
    X0 = initial_design(n_init, problem.D, seed)
    ys = [problem(x) for x in X0]
    x_best, y_best = X0[int(np.argmin(ys))].copy(), float(np.min(ys))
    rng = np.random.default_rng([seed, 202])
    x0 = x_best.copy()
    restarts, diagnostics = 0, {}
    while len(ys) < budget:
        opts = {
            "bounds": [-1.0, 1.0],
            "seed": int(rng.integers(1, 2**31 - 1)),
            "verbose": -9,
        }
        if popsize is not None:
            opts["popsize"] = int(popsize)
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop() and len(ys) < budget:
            cands = es.ask()
            if len(ys) + len(cands) > budget:
                # Not enough budget left for a full generation: evaluate what we
                # can (these still count) and stop without a partial update.
                for x in cands[: budget - len(ys)]:
                    x = np.clip(x, -1.0, 1.0)
                    ys.append(problem(x))
                    if ys[-1] < y_best:
                        x_best, y_best = x.copy(), ys[-1]
                break
            fs = []
            for x in cands:
                x = np.clip(x, -1.0, 1.0)
                fs.append(problem(x))
                if fs[-1] < y_best:
                    x_best, y_best = x.copy(), fs[-1]
            ys.extend(fs)
            es.tell(cands, fs)
        diagnostics = _cma_diagnostics(es, problem)
        diagnostics["stop_reason"] = {k: str(v) for k, v in es.stop().items()}
        if len(ys) < budget:  # restart from a uniformly random point
            restarts += 1
            x0 = rng.uniform(-1.0, 1.0, size=problem.D)
    diagnostics["restarts"] = restarts
    return _result(ys, start, diagnostics, x_best)



# ---------------------------------------------------------------------------
# GP-based Bayesian optimisation
# ---------------------------------------------------------------------------

class GPBO:
    """Minimal GP-BO loop on [-1, 1]^dim with EI and a candidate-set optimiser."""

    def __init__(self, dim: int, ard: bool, seed: int, n_restarts: int = 2,
                 refit_every: int = 1, n_candidates: int = 5000,
                 y_transform: str = "none"):
        self.dim, self.ard = dim, ard
        self.rng = np.random.default_rng([seed, 303])
        self.n_restarts, self.refit_every = n_restarts, refit_every
        self.n_candidates, self.y_transform = n_candidates, y_transform
        ls = np.full(dim, 0.5) if ard else 0.5
        self.kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                       * Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                       + WhiteKernel(1e-6, (1e-9, 1e-2)))
        self.gp = None
        self.n_fits = 0
        self.fit_time = 0.0
        self.acq_time = 0.0

    def _transform(self, y: np.ndarray) -> np.ndarray:
        if self.y_transform == "none":
            return y
        if self.y_transform == "log":
            # Monotone, uses only observed data (not the known minimum).
            return np.log(y - y.min() + 1e-3 * (np.std(y) + 1e-12))
        raise ValueError(f"unknown y_transform {self.y_transform!r}")

    def fit(self, X: np.ndarray, y: np.ndarray, it: int) -> None:
        t0 = time.perf_counter()
        yt = self._transform(y)
        refit = self.gp is None or it % self.refit_every == 0
        kernel = self.kernel if self.gp is None else self.gp.kernel_
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            optimizer="fmin_l_bfgs_b" if refit else None,
            n_restarts_optimizer=self.n_restarts if refit else 0,
            random_state=int(self.rng.integers(2**31 - 1)),
        )
        gp.fit(X, yt)
        self.gp, self.yt = gp, yt
        self.n_fits += int(refit)
        self.fit_time += time.perf_counter() - t0

    def propose(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        n_glob = self.n_candidates
        cands = [self.rng.uniform(-1.0, 1.0, size=(n_glob, self.dim))]
        # Local candidates around the best points. Perturb a random subset of
        # coordinates (as in TuRBO) so local moves stay local in high dimension.
        order = np.argsort(self.yt)[: min(5, len(self.yt))]
        p = min(1.0, 20.0 / self.dim)
        per = n_glob // (2 * len(order))
        for idx in order:
            for scale in (0.1, 0.02):
                mask = self.rng.random((per, self.dim)) < p
                mask[np.arange(per), self.rng.integers(self.dim, size=per)] = True
                step = scale * self.rng.standard_normal((per, self.dim)) * mask
                cands.append(np.clip(X[idx] + step, -1.0, 1.0))
        C = np.vstack(cands)
        mu, sd = self.gp.predict(C, return_std=True)
        lei = log_expected_improvement(mu, sd, best=float(self.yt.min()))
        self.acq_time += time.perf_counter() - t0
        return C[int(np.argmax(lei))]

    def lengthscales(self) -> list[float]:
        ls = self.gp.kernel_.k1.k2.length_scale
        return [float(v) for v in np.atleast_1d(ls)]


def _bo_loop(f, dim, X0, budget, seed, ard, **gp_kw):
    bo = GPBO(dim, ard=ard, seed=seed, **gp_kw)
    X = np.array(X0, dtype=float)
    y = np.array([f(x) for x in X])
    it = 0
    while len(y) < budget:
        bo.fit(X, y, it)
        x_new = bo.propose(X)
        X = np.vstack([X, x_new])
        y = np.append(y, f(x_new))
        it += 1
    bo.fit(X, y, 0)  # final fit for diagnostics
    diag = {
        "lengthscales": bo.lengthscales(),
        "gp_fit_time_s": bo.fit_time,
        "acq_time_s": bo.acq_time,
        "n_hyperparameter_fits": bo.n_fits,
    }
    return X, y, diag


def run_gp(problem, budget, n_init, seed, ard: bool = True, n_restarts: int = 2,
           refit_every: int = 1, n_candidates: int = 5000, y_transform: str = "none", **_):
    start = time.perf_counter()
    X0 = initial_design(n_init, problem.D, seed)
    X, y, diag = _bo_loop(problem, problem.D, X0, budget, seed, ard,
                          n_restarts=n_restarts, refit_every=refit_every,
                          n_candidates=n_candidates, y_transform=y_transform)
    if ard and problem.relevant_coords is not None:
        # For aligned problems: do the shortest lengthscales pick out the relevant
        # coordinates? (fraction of the d_e shortest that are truly relevant)
        ls = np.array(diag["lengthscales"])
        shortest = set(np.argsort(ls)[: problem.d_e].tolist())
        diag["relevant_recovered_frac"] = len(shortest & set(problem.relevant_coords.tolist())) / problem.d_e
    return _result(y, start, diag, X[int(np.argmin(y))])


def run_gp_ard(problem, budget, n_init, seed, **kw):
    kw["ard"] = True
    return run_gp(problem, budget, n_init, seed, **kw)


def run_gp_iso(problem, budget, n_init, seed, **kw):
    kw["ard"] = False
    return run_gp(problem, budget, n_init, seed, **kw)




# ---------------------------------------------------------------------------
# TuRBO-1 (Eriksson et al., 2019, "Scalable Global Optimization via Local
# Bayesian Optimization", Section 3; single trust region, batch size 1)
# ---------------------------------------------------------------------------

def _turbo_gp(U: np.ndarray, y: np.ndarray, rng, prev_kernel=None, refit: bool = True):
    """GP on the unit cube with the hyperparameter ranges used in the paper."""
    dim = U.shape[1]
    kernel = prev_kernel if prev_kernel is not None else (
        ConstantKernel(1.0, (5e-2, 20.0))
        * Matern(length_scale=np.full(dim, 0.5), length_scale_bounds=(5e-3, 2.0), nu=2.5)
        + WhiteKernel(1e-4, (1e-8, 1e-1)))
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True,
        optimizer="fmin_l_bfgs_b" if refit else None,
        n_restarts_optimizer=1 if (refit and prev_kernel is None) else 0,
        random_state=int(rng.integers(2**31 - 1)))
    gp.fit(U, y)
    return gp


def _thompson_sample(gp, C: np.ndarray, rng) -> int:
    """Index of the minimiser of one joint posterior sample over the candidates."""
    mu, cov = gp.predict(C, return_cov=True)
    cov = 0.5 * (cov + cov.T)
    jitter = 1e-8 * max(float(np.mean(np.diag(cov))), 1e-12)
    for _ in range(6):
        try:
            L = np.linalg.cholesky(cov + jitter * np.eye(len(C)))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:  # fall back to independent marginals if the covariance is degenerate
        L = np.diag(np.sqrt(np.maximum(np.diag(cov), 0.0)))
    sample = mu + L @ rng.standard_normal(len(C))
    return int(np.argmin(sample))


def run_turbo(problem, budget, n_init, seed, length_init: float = 0.8,
              length_min: float = 0.5 ** 7, length_max: float = 1.6,
              succ_tol: int = 3, fail_tol: int | None = None,
              n_candidates: int | None = None, refit_every: int = 1, **_) -> dict:
    """TuRBO-1 with the default settings of the paper.

    The algorithm works on the unit cube [0, 1]^D (x = 2u - 1), so the length
    settings are those of the paper: initial side length 0.8, minimum 2^-7,
    maximum 1.6. The trust region is a box centred on the best point found in
    the current run. Its side length along coordinate i is

        L_i = lambda_i * L / (prod_j lambda_j)^(1/D),

    where lambda_i are the GP's ARD lengthscales, so the box is longer along
    inputs the GP believes matter less. The box is therefore always axis-aligned:
    this is why TuRBO is *not* rotation-invariant, which is the property tested
    in this project.

    After each evaluation: an improvement counts as a success, otherwise a failure.
    succ_tol successes in a row double L (up to length_max); fail_tol failures in a
    row halve it (default fail_tol = max(4, D), as in the paper). When L falls below
    length_min the trust region has converged: TuRBO restarts with a fresh initial
    design and a GP that forgets the old data. The global best is kept throughout.

    Deviation from the paper: Thompson samples are drawn exactly by a Cholesky
    factorisation over at most 2000 candidates (the paper uses up to 5000 and a
    scalable approximation), to keep the cost manageable with scikit-learn.
    """
    start = time.perf_counter()
    D = problem.D
    rng = np.random.default_rng([seed, 505])
    fail_tol = max(4, D) if fail_tol is None else int(fail_tol)
    n_cand = min(100 * D, 2000) if n_candidates is None else int(n_candidates)
    p_perturb = min(1.0, 20.0 / D)

    def f_unit(u):
        return problem(2.0 * u - 1.0)

    all_y, x_best, y_best = [], None, np.inf
    length_history, lengthscales = [], None
    n_restarts, n_shrinks, n_expands, fit_time = 0, 0, 0, 0.0

    U0 = (initial_design(n_init, D, seed) + 1.0) / 2.0  # shared design, unit cube
    while len(all_y) < budget:
        # --- (re)start a trust region ---------------------------------------
        U = U0[: budget - len(all_y)]
        y = np.array([f_unit(u) for u in U])
        all_y.extend(y.tolist())
        length, succ, fail = length_init, 0, 0
        gp_kernel, it = None, 0
        while len(all_y) < budget and length >= length_min:
            t0 = time.perf_counter()
            gp = _turbo_gp(U, y, rng, gp_kernel, refit=(it % refit_every == 0))
            gp_kernel, it = gp.kernel_, it + 1
            fit_time += time.perf_counter() - t0
            ls = np.atleast_1d(gp.kernel_.k1.k2.length_scale).astype(float)
            ls = np.broadcast_to(ls, (D,))
            lengthscales = ls

            # Trust region: centred on the best point of this run.
            centre = U[int(np.argmin(y))]
            weights = ls / np.exp(np.mean(np.log(ls)))  # geometric mean = 1
            lb = np.clip(centre - weights * length / 2.0, 0.0, 1.0)
            ub = np.clip(centre + weights * length / 2.0, 0.0, 1.0)

            # Candidates: perturb a random subset of coordinates of the centre
            # (at least one) with values drawn uniformly inside the trust region.
            pert = lb + (ub - lb) * rng.random((n_cand, D))
            mask = rng.random((n_cand, D)) < p_perturb
            mask[np.arange(n_cand), rng.integers(D, size=n_cand)] = True
            C = np.where(mask, pert, centre)

            u_new = C[_thompson_sample(gp, C, rng)]
            y_new = f_unit(u_new)
            all_y.append(y_new)
            length_history.append(length)

            # Success / failure counters (improvement threshold from the paper).
            if y_new < y.min() - 1e-3 * abs(y.min()):
                succ, fail = succ + 1, 0
            else:
                succ, fail = 0, fail + 1
            if succ == succ_tol:
                length, succ, n_expands = min(2.0 * length, length_max), 0, n_expands + 1
            elif fail == fail_tol:
                length, fail, n_shrinks = length / 2.0, 0, n_shrinks + 1

            U = np.vstack([U, u_new])
            y = np.append(y, y_new)

        i = int(np.argmin(y))
        if y[i] < y_best:
            x_best, y_best = 2.0 * U[i] - 1.0, float(y[i])
        if len(all_y) < budget:  # converged: restart with a new random design
            n_restarts += 1
            U0 = rng.random((n_init, D))

    diag = {
        "restarts": n_restarts,
        "n_expands": n_expands,
        "n_shrinks": n_shrinks,
        "final_length": length_history[-1] if length_history else length_init,
        "length_history": length_history,
        "lengthscales": [float(v) for v in lengthscales] if lengthscales is not None else [],
        "gp_fit_time_s": fit_time,
    }
    if problem.relevant_coords is not None and lengthscales is not None:
        shortest = set(np.argsort(lengthscales)[: problem.d_e].tolist())
        diag["relevant_recovered_frac"] = len(shortest & set(problem.relevant_coords.tolist())) / problem.d_e
    return _result(all_y, start, diag, x_best)


METHODS = {
    "random": run_random,
    "cmaes": run_cmaes,
    "gp_ard": run_gp_ard,
    "gp_iso": run_gp_iso,
    "turbo": run_turbo,
}
