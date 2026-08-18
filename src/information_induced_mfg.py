"""
Information-induced Mean-Field Game evaluator.

Key modelling correction:
the information model does NOT generate the exogenous arrival process.
It changes behavioural decisions such as jockeying/switching and
reneging/abandonment.  Thus lambda0 is fixed, while information enters
the behavioural flow terms.

The script numerically evaluates:
  * an information-induced stationary MFE;
  * equilibrium residuals;
  * a local uniqueness/contraction diagnostic;
  * the linear DDE matrices A and B;
  * characteristic roots and delay-dependent stability;
  * a numerical Hopf-crossing diagnostic;
  * simple dimensionality comparisons.

The numerical tests do not constitute proofs.  The theorem hypotheses
must still be established analytically.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import numpy as np
from numpy.linalg import norm
from scipy.optimize import root, minimize, brentq


Array = np.ndarray


@dataclass
class MFGParams:
    N: int
    lambda0: Array          # fixed/exogenous arrivals
    mu0: Array
    mu_slope: Array
    alpha_max: Array
    k_alpha: Array
    alpha_threshold: Array
    beta: float
    a: Array
    gamma: float            # jockeying intensity
    tau: float = 0.0
    U: Optional[Array] = None

    def __post_init__(self):
        arrays = [
            "lambda0", "mu0", "mu_slope", "alpha_max",
            "k_alpha", "alpha_threshold", "a"
        ]
        for name in arrays:
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        if self.U is None:
            self.U = np.zeros(self.N)
        else:
            self.U = np.asarray(self.U, dtype=float)

        for name in arrays + ["U"]:
            if len(getattr(self, name)) != self.N:
                raise ValueError(f"{name} must have length N={self.N}")


#def sigmoid(x):
#    x = np.clip(x, -60.0, 60.0)
#    return 1.0 / (1.0 + np.exp(-x))


#def softmax(z):
#    z = z - np.max(z)
#    e = np.exp(z)
#    return e / np.sum(e)

@staticmethod
def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))

@staticmethod
def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if logits.size == 0:
        return logits
    x = logits - np.max(logits)
    exps = np.exp(x)

    return exps / max(float(np.sum(exps)), 1e-15)


class InformationModel:
    """
    H(q) maps the physical state to the perceived state.

    Replace H by a Markovian estimator, learned estimator, etc.
    For delayed telemetry the DDE itself supplies q(t-tau) to H.
    """
    def __init__(self, H: Callable[[Array], Array]):
        self.H = H

    def perceive(self, q):
        return np.asarray(self.H(q), dtype=float)


def service(q, p):
    # Continuous increasing example.
    return p.mu0 + p.mu_slope * q / (1.0 + q)


def d_service(q, p):
    return p.mu_slope / (1.0 + q) ** 2


def perceived_reward(I, p):
    return p.U - p.a * I


def reneging(q, I, p):
    # Information affects reneging through perceived congestion.
    return p.alpha_max * sigmoid(
        p.k_alpha * (p.a * I - p.alpha_threshold)
    )


def d_reneging_dI(q, I, p):
    s = sigmoid(p.k_alpha * (p.a * I - p.alpha_threshold))
    return p.alpha_max * p.k_alpha * p.a * s * (1.0 - s)


def jockeying_matrix(I, p):
    """
    r[j,i] = switching rate from queue j to queue i.

    Softmax/Boltzmann appears here as a behavioural response among
    already-present tenants. It is NOT an arrival splitter.
    """
    N = p.N
    R = perceived_reward(I, p)
    rates = np.zeros((N, N))

    for j in range(N):
        mask = np.ones(N, dtype=bool)
        mask[j] = False
        rates[j, mask] = p.gamma * softmax(p.beta * R[mask])

    return rates


def jockeying_net_flow(q, I, p):
    R = jockeying_matrix(I, p)
    inflow = q @ R
    outflow = q * np.sum(R, axis=1)
    return inflow - outflow


def vector_field(q, p, info):
    """
    Stationary/zero-delay population dynamics:
        qdot = lambda0
             + jockeying_net_flow(q,H(q))
             - service(q)
             - reneging(q,H(q))

    Arrival rates lambda0 are exogenous and are NOT determined by H.
    """
    I = behaviour.perceive(q)
    B = behaviour.flow(q, I, p)


    return (
        p.lambda0
        + jockeying_net_flow(q, I, p)
        - service(q, p)
        + B
    )


def find_mfe(p, info, q0=None):
    if q0 is None:
        q0 = np.maximum(p.lambda0 / np.maximum(p.mu0, 1e-8), 1e-3)

    sol = root(lambda q: vector_field(q, p, info), q0)

    if not sol.success:
        raise RuntimeError(sol.message)

    q_star = np.maximum(sol.x, 0.0)
    residual = norm(vector_field(q_star, p, info), np.inf)

    if residual > 1e-7:
        raise RuntimeError(
            f"Candidate MFE residual too large: {residual:.3e}"
        )

    return q_star


def numerical_jacobian(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    n = len(x)
    J = np.zeros((n, n))

    for k in range(n):
        h = eps * max(1.0, abs(x[k]))
        xp = x.copy()
        xm = x.copy()
        xp[k] += h
        xm[k] -= h
        J[:, k] = (f(xp) - f(xm)) / (2.0 * h)

    return J


def physical_matrix(q_star, p):
    """
    A in
        delta qdot(t) = A delta q(t) + B delta q(t-tau).

    For the delayed-information model, instantaneous physical damping
    comes from service and any direct q-dependence of departures.
    """
    return -np.diag(d_service(q_star, p))


def behavioural_matrix(q_star, p):
    """
    B in the delayed linearization.

    Information-dependent jockeying and reneging are placed in B.
    """
    def g(I):
        return (
            jockeying_net_flow(q_star, I, p)
            - reneging(q_star, I, p)
        )

    return numerical_jacobian(g, q_star)


def characteristic_matrix(lam, A, B, tau):
    n = A.shape[0]
    return lam * np.eye(n) - A - B * np.exp(-lam * tau)


def characteristic_residual(lam, A, B, tau):
    return abs(np.linalg.det(characteristic_matrix(lam, A, B, tau)))


def characteristic_roots(
    A, B, tau,
    re_grid=np.linspace(-2.0, 1.0, 15),
    im_grid=np.linspace(-8.0, 8.0, 31),
    tol=1e-7
):
    """
    Numerical root search for det Delta(lambda)=0.

    This is a diagnostic solver. It is not a rigorous DDE spectral proof.
    """
    roots = []

    def objective(x):
        lam = x[0] + 1j * x[1]
        return np.log1p(characteristic_residual(lam, A, B, tau))

    for re0 in re_grid:
        for im0 in im_grid:
            sol = minimize(
                objective,
                np.array([re0, im0]),
                method="Nelder-Mead",
                options={"maxiter": 600}
            )

            lam = sol.x[0] + 1j * sol.x[1]
            if characteristic_residual(lam, A, B, tau) < tol:
                if not any(abs(lam - z) < 1e-4 for z in roots):
                    roots.append(lam)

    return sorted(roots, key=lambda z: z.real, reverse=True)


def rightmost_root(A, B, tau):
    roots = characteristic_roots(A, B, tau)
    return roots[0] if roots else None


def stability_diagnostic(A, B, tau):
    lam = rightmost_root(A, B, tau)

    if lam is None:
        return {"root": None, "classification": "undetermined"}

    if lam.real < -1e-6:
        cls = "locally stable"
    elif lam.real > 1e-6:
        cls = "locally unstable"
    else:
        cls = "near stability boundary"

    return {"root": lam, "classification": cls}


def estimate_hopf_threshold(A, B, tau_lo=0.0, tau_hi=10.0, n_scan=40):
    """
    Locate the first numerical crossing of Re(lambda_max(tau))=0.

    A genuine Hopf theorem additionally needs a simple conjugate pair,
    nonzero frequency, and transversality; supercriticality requires
    nonlinear normal-form information.
    """
    taus = np.linspace(tau_lo, tau_hi, n_scan)
    vals = []

    for tau in taus:
        lam = rightmost_root(A, B, tau)
        vals.append(np.nan if lam is None else lam.real)

    for k in range(len(taus) - 1):
        if not np.isfinite(vals[k]) or not np.isfinite(vals[k + 1]):
            continue
        if vals[k] * vals[k + 1] > 0:
            continue

        def f(t):
            lam = rightmost_root(A, B, t)
            return np.nan if lam is None else lam.real

        try:
            tau_c = brentq(f, taus[k], taus[k + 1])
        except ValueError:
            continue

        lam_c = rightmost_root(A, B, tau_c)

        if lam_c is not None and abs(lam_c.imag) > 1e-5:
            return tau_c, lam_c

    return None, None


def local_contraction_diagnostic(p, info, q_star, eta=0.05):
    """
    Numerical diagnostic only.

    T(q)=q-eta*F(q).  If ||DT(q*)||<1 this suggests local contraction,
    but it does NOT establish global uniqueness.
    """
    def T(q):
        return q - eta * vector_field(q, p, info)

    J = numerical_jacobian(T, q_star)
    L = np.linalg.norm(J, 2)

    return {
        "local_L_estimate": L,
        "local_contraction": bool(L < 1.0),
        "interpretation": "local numerical diagnostic, not a uniqueness proof"
    }


def performance_proxy(q, lambda0):
    """
    Occupancy/arrival proxy:
        sum_i q_i / sum_i lambda0_i.

    If reneging is significant, label this an occupancy proxy rather
    than literal completed-customer sojourn time.
    """
    return float(np.sum(q) / np.sum(lambda0))


def evaluate_model(p, info, q0=None):
    q_star = find_mfe(p, info, q0)
    I_star = info.perceive(q_star)

    A = physical_matrix(q_star, p)
    B = behavioural_matrix(q_star, p)

    return {
        "q_star": q_star,
        "I_star": I_star,
        "residual_inf": norm(vector_field(q_star, p, info), np.inf),
        "service": service(q_star, p),
        "reneging": reneging(q_star, I_star, p),
        "jockeying_net": jockeying_net_flow(q_star, I_star, p),
        "A": A,
        "B": B,
        "contraction": local_contraction_diagnostic(p, info, q_star),
        "performance_proxy": performance_proxy(q_star, p.lambda0),
    }


def print_result(name, r):
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print("q* =", np.round(r["q_star"], 6))
    print("I* =", np.round(r["I_star"], 6))
    print("MFE residual =", f"{r['residual_inf']:.3e}")
    print("service =", np.round(r["service"], 6))
    print("reneging =", np.round(r["reneging"], 6))
    print("net jockeying =", np.round(r["jockeying_net"], 6))
    print("\nA =")
    print(np.round(r["A"], 6))
    print("\nB =")
    print(np.round(r["B"], 6))
    print("\ncontraction diagnostic =", r["contraction"])
    print("occupancy/arrival proxy =", r["performance_proxy"])


def make_example(N=3, tau=0.0):
    return MFGParams(
        N=N,
        lambda0=np.full(N, 0.35),
        mu0=np.full(N, 0.28),
        mu_slope=np.full(N, 0.50),
        alpha_max=np.full(N, 0.35),
        k_alpha=np.full(N, 1.5),
        alpha_threshold=np.full(N, 1.0),
        beta=1.5,
        a=np.full(N, 1.0),
        gamma=0.25,
        tau=tau,
        U=np.zeros(N),
    )


def identity_information(q):
    return q.copy()


if __name__ == "__main__":
    '''
    Base MFE Evaluation: Computes equilibrium queue lengths $q^*$, system residuals, and Jacobians $A$ and $B$ for a 3-queue system.
    Multiple-Start Equilibrium Test: Probes for potential multi-stability by solving from multiple initial conditions ($q_0$).
    Delay Stability Sweep: Tracks the rightmost characteristic root across delays $\tau \in [0, 8]$.
    Hopf Threshold Estimation: Finds candidate critical delay $\tau_c$ and crossing frequency $\omega_c$.

    Dimensionality Scaling: Tests how mean equilibrium occupancy scales across system sizes $N \in \{2, 3, 4, 5\}$.
    '''
    # --------------------------------------------------------------
    # Base information-induced MFE
    # --------------------------------------------------------------
    p = make_example(N=3, tau=0.0)
    info = InformationModel(identity_information)

    r = evaluate_model(p, info)
    print_result("BASE INFORMATION-INDUCED MFE", r)

    # --------------------------------------------------------------
    # Multiple-start test: useful for probing possible multiplicity
    # --------------------------------------------------------------
    print("\n" + "=" * 72)
    print("MULTIPLE-START EQUILIBRIUM TEST")
    print("=" * 72)

    starts = [
        np.zeros(p.N),
        np.ones(p.N),
        np.full(p.N, 2.0),
        np.array([0.1, 2.0, 4.0]),
    ]

    for q0 in starts:
        try:
            q = find_mfe(p, info, q0)
            print("start =", q0, " -> q* =", np.round(q, 6))
        except RuntimeError as e:
            print("start =", q0, " -> FAILED:", e)

    # --------------------------------------------------------------
    # Delay-dependent stability
    # --------------------------------------------------------------
    print("\n" + "=" * 72)
    print("DELAY-DEPENDENT STABILITY")
    print("=" * 72)

    for tau in np.linspace(0.0, 8.0, 9):
        d = stability_diagnostic(r["A"], r["B"], tau)
        print(f"tau={tau:5.2f}  {d}")

    # --------------------------------------------------------------
    # Numerical Hopf diagnostic
    # --------------------------------------------------------------
    tau_c, lambda_c = estimate_hopf_threshold(
        r["A"], r["B"], tau_lo=0.0, tau_hi=8.0, n_scan=25
    )

    print("\n" + "=" * 72)
    print("HOPF DIAGNOSTIC")
    print("=" * 72)

    if tau_c is None:
        print("No numerical imaginary-axis crossing detected.")
    else:
        print("estimated tau_c =", tau_c)
        print("critical root =", lambda_c)
        print("critical frequency =", abs(lambda_c.imag))
        print(
            "This only identifies a candidate Hopf crossing. "
            "It does not prove supercriticality."
        )

    # --------------------------------------------------------------
    # Dimensionality experiment
    # --------------------------------------------------------------
    print("\n" + "=" * 72)
    print("DIMENSIONALITY EXPERIMENT")
    print("=" * 72)

    for N in [2, 3, 4, 5]:
        pp = make_example(N=N, tau=0.0)
        ii = InformationModel(identity_information)
        rr = evaluate_model(pp, ii)
        print(
            f"N={N}: mean(q*)={np.mean(rr['q_star']):.5f}, "
            f"proxy={rr['performance_proxy']:.5f}"
        )
