"""Test problems with low effective dimension embedded in a higher ambient dimension.

Every problem has the form

    f(x) = g(A^T x - z_star) + eps * ||(I - A A^T) x||^2,     x in [-1, 1]^D

where
    * D      is the ambient dimension (number of inputs the optimiser controls),
    * d_e    is the effective dimension (number of columns of A),
    * A      is a D x d_e matrix with orthonormal columns,
    * g      is a base function on R^{d_e} with minimum value 0 at the origin,
    * z_star shifts the optimum away from the centre of the box,
    * eps    optionally makes the "irrelevant" directions matter slightly.

Alignment
---------
* "aligned": A selects d_e randomly chosen coordinates, so the relevant subspace is
  spanned by coordinate axes (ARD lengthscales can, in principle, detect it).
* "rotated": A has random orthonormal columns (QR of a Gaussian matrix), so every
  coordinate contributes a little and no coordinate can be switched off.

Note on scale: if x is uniform on [-1, 1]^D then each component of z = A^T x has
variance 1/3 in both cases, so the typical scale of the base function's input is
the same. The *range* of z differs, however: in the rotated case the box projects
onto a larger region of R^{d_e}. This is one way that box constraints break the
symmetry between the aligned and rotated problems.

The optimum x_star is drawn uniformly from [-0.5, 0.5]^D and z_star = A^T x_star, so
the global minimum value is exactly 0 and lies strictly inside the box for both
alignments. The value f(x) is therefore the simple regret.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Base functions on R^{d_e}. Each has global minimum 0 at u = 0.
# The inner scaling maps the typical range u in [-1, 1] onto a sensible region
# of the classical function's domain.
# ---------------------------------------------------------------------------

def sphere(u: np.ndarray) -> float:
    return float(np.sum(u ** 2))


def rosenbrock(u: np.ndarray) -> float:
    # Classical minimum at v = (1, ..., 1); v = 2u + 1 moves it to u = 0.
    v = 2.0 * u + 1.0
    if v.size == 1:  # Rosenbrock needs >= 2 dims; fall back to a 1-D valley
        return float((1.0 - v[0]) ** 2)
    return float(np.sum(100.0 * (v[1:] - v[:-1] ** 2) ** 2 + (1.0 - v[:-1]) ** 2))


def ackley(u: np.ndarray) -> float:
    v = 5.0 * u
    d = v.size
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    term1 = -a * np.exp(-b * np.sqrt(np.sum(v ** 2) / d))
    term2 = -np.exp(np.sum(np.cos(c * v)) / d)
    return float(term1 + term2 + a + np.e)


def rastrigin(u: np.ndarray) -> float:
    v = 3.0 * u
    return float(10.0 * v.size + np.sum(v ** 2 - 10.0 * np.cos(2.0 * np.pi * v)))


BASE_FUNCTIONS = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
    "ackley": ackley,
    "rastrigin": rastrigin,
}


# ---------------------------------------------------------------------------
# Embedded problem
# ---------------------------------------------------------------------------

@dataclass
class EmbeddedProblem:
    base: str
    D: int
    d_e: int
    alignment: str  # "aligned" or "rotated"
    seed: int
    eps: float = 0.0
    A: np.ndarray = field(init=False, repr=False)
    x_star: np.ndarray = field(init=False, repr=False)
    z_star: np.ndarray = field(init=False, repr=False)
    relevant_coords: np.ndarray | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.base not in BASE_FUNCTIONS:
            raise ValueError(f"unknown base function {self.base!r}")
        if not 1 <= self.d_e <= self.D:
            raise ValueError("need 1 <= d_e <= D")
        # The problem instance depends only on (base, D, d_e, alignment, seed), so
        # every method sees the *same* instance for a given seed (paired design).
        rng = np.random.default_rng([self.seed, self.D, self.d_e, 7919])
        if self.alignment == "aligned":
            coords = np.sort(rng.choice(self.D, size=self.d_e, replace=False))
            A = np.zeros((self.D, self.d_e))
            A[coords, np.arange(self.d_e)] = 1.0
            self.relevant_coords = coords
        elif self.alignment == "rotated":
            G = rng.standard_normal((self.D, self.d_e))
            Q, R = np.linalg.qr(G)
            Q = Q * np.sign(np.diag(R))  # make the decomposition unique
            A = Q
            self.relevant_coords = None
        else:
            raise ValueError("alignment must be 'aligned' or 'rotated'")
        self.A = A
        self.x_star = rng.uniform(-0.5, 0.5, size=self.D)
        self.z_star = self.A.T @ self.x_star
        self._g = BASE_FUNCTIONS[self.base]
        self.n_evals = 0

    @property
    def lower(self) -> np.ndarray:
        return -np.ones(self.D)

    @property
    def upper(self) -> np.ndarray:
        return np.ones(self.D)

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.D,):
            raise ValueError(f"expected shape ({self.D},), got {x.shape}")
        if np.any(x < -1.0 - 1e-9) or np.any(x > 1.0 + 1e-9):
            raise ValueError("point outside the box [-1, 1]^D")
        self.n_evals += 1
        z = self.A.T @ x
        val = self._g(z - self.z_star)
        if self.eps > 0.0:
            # The optimum is defined to stay at x_star, so penalise distance from
            # x_star in the orthogonal complement (minimum value remains 0).
            r = x - self.x_star
            r_perp = r - self.A @ (self.A.T @ r)
            val += self.eps * float(r_perp @ r_perp)
        return val

    def describe(self) -> dict:
        return {
            "base": self.base,
            "D": self.D,
            "d_e": self.d_e,
            "alignment": self.alignment,
            "seed": self.seed,
            "eps": self.eps,
        }


def make_problem(cfg: dict, seed: int) -> EmbeddedProblem:
    return EmbeddedProblem(
        base=cfg["base"],
        D=int(cfg["D"]),
        d_e=int(cfg["d_e"]),
        alignment=cfg["alignment"],
        seed=int(seed),
        eps=float(cfg.get("eps", 0.0)),
    )
