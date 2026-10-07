from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import math
import numpy as np
import matplotlib.pyplot as plt
from math import gamma as gamma_fn

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ============================================================
# Parameters
# ============================================================
@dataclass
class Params:
    N: int = 2
    lam: np.ndarray = None      # base demand per queue
    sigma: np.ndarray = None    # trust discount
    mu: np.ndarray = None       # service rates
    kappa: float = 3.0          # demand sensitivity to delayed congestion
    r0: float = 0.5             # baseline renunciation rate
    r1: float = 0.10            # renunciation slope in delayed congestion
    k_Mk: float = 0.5
    beta: float = 1.0
    alpha_star: float = 0.4
    theta3: float = 0.6
    theta4: float = 0.1
    steep: float = 4.0          # softplus steepness for routing
    relaxation: float = 0.5
    tol: float = 1e-10
    max_iter: int = 5000

    def __post_init__(self):
        if self.lam is None:
            self.lam = np.array([2.0, 2.4])
        if self.sigma is None:
            self.sigma = np.array([0.9, 0.85])
        if self.mu is None:
            self.mu = np.array([2.0, 1.7])


# ============================================================
# Model: controls (the "HJB" best response) and moment dynamics
# ============================================================
def softplus(x, s):
    return np.log1p(np.exp(s * x)) / s


def controls(Md, p: Params, p_mk, p_ac):
    """Optimal-control surrogate given delayed (trust-discounted) congestion."""
    N = p.N
    ren = p.r0 + p.r1 * p.sigma * Md
    alpha = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            gain = p.sigma[j] * (
                p_mk * p.k_Mk / p.mu[j]
                + p_ac * p.beta * p.alpha_star * (p.theta3 - p.theta4)
            )
            alpha[i, j] = gain * softplus(p.sigma[i] * Md[i] - p.sigma[j] * Md[j], p.steep)
    return ren, alpha


def inflow(Md, p: Params):
    return p.lam / (1.0 + p.kappa * p.sigma * Md)


def rhs(M, Md, p, p_mk, p_ac):
    """Moment dynamics dM/dt = f(M, M_tau)."""
    ren, alpha = controls(Md, p, p_mk, p_ac)
    out = ren + alpha.sum(axis=1)
    into = alpha.T @ M          # sum_j alpha_{j->i} M_j
    return inflow(Md, p) - M * out + into


def stationary_given_expectation(Md, p, p_mk, p_ac):
    """Solve f(M, Md)=0 for M with Md frozen (this is the response map T)."""
    ren, alpha = controls(Md, p, p_mk, p_ac)
    out = ren + alpha.sum(axis=1)
    L = np.diag(out) - alpha.T
    return np.linalg.solve(L, inflow(Md, p))


# ============================================================
# Fixed-point iteration (forward-backward style, relaxed)
# ============================================================
def solve_equilibrium(p, p_mk, p_ac, M0):
    M = M0.astype(float).copy()
    hist = []
    for k in range(p.max_iter):
        # "backward" step: controls from current congestion
        ren, alpha = controls(M, p, p_mk, p_ac)
        # "forward" step: stationary moments under those controls
        M_next = stationary_given_expectation(M, p, p_mk, p_ac)
        err = float(np.max(np.abs(M_next - M)))
        hist.append(err)
        M = p.relaxation * M_next + (1 - p.relaxation) * M
        if err < p.tol:
            return M, True, k + 1, hist
    return M, False, p.max_iter, hist


# ============================================================
# Linearization and delay stability
# ============================================================
def jacobians(Mbar, p, p_mk, p_ac, h=1e-6):
    N = p.N
    A = np.zeros((N, N)); B = np.zeros((N, N))
    for k in range(N):
        e = np.zeros(N); e[k] = h
        A[:, k] = (rhs(Mbar + e, Mbar, p, p_mk, p_ac) - rhs(Mbar - e, Mbar, p, p_mk, p_ac)) / (2 * h)
        B[:, k] = (rhs(Mbar, Mbar + e, p, p_mk, p_ac) - rhs(Mbar, Mbar - e, p, p_mk, p_ac)) / (2 * h)
    return A, B


def z_roots(omega, A, B):
    """Solve det(i w I - A - z B) = 0 for z (generalized eigenproblem)."""
    M0 = 1j * omega * np.eye(A.shape[0]) - A
    try:
        return np.linalg.eigvals(np.linalg.solve(B, M0))
    except np.linalg.LinAlgError:
        return np.array([])


def critical_delay(A, B, omega_max=20.0, steps=20000, max_branch=10):
    omegas = np.linspace(1e-3, omega_max, steps)

    def sorted_moduli(w):
        z = z_roots(w, A, B)
        return np.sort(np.abs(z)) if z.size else None

    prev = sorted_moduli(omegas[0])
    cands: List[Dict[str, float]] = []
    for a, b in zip(omegas[:-1], omegas[1:]):
        cur = sorted_moduli(b)
        if prev is None or cur is None or prev.size != cur.size:
            prev = cur; continue
        for idx in range(cur.size):
            if (prev[idx] - 1.0) * (cur[idx] - 1.0) < 0:
                lo, hi = a, b
                for _ in range(60):  # bisection on w
                    mid = 0.5 * (lo + hi)
                    m = sorted_moduli(mid)
                    if (m[idx] - 1.0) * (prev[idx] - 1.0) > 0:
                        lo = mid
                    else:
                        hi = mid
                w = 0.5 * (lo + hi)
                z = z_roots(w, A, B)
                zc = z[np.argmin(np.abs(np.abs(z) - 1.0))]
                th = np.angle(zc)
                for n in range(max_branch):
                    tau = (-th + 2 * math.pi * n) / w
                    if tau > 0:
                        cands.append({"omega": w, "tau": tau})
        prev = cur
    if not cands:
        return None, []
    cands.sort(key=lambda c: c["tau"])
    return cands[0]["tau"], cands


def simulate_dde(A, B, tau, T=100.0, dt=0.005, xi0=None):
    n = int(T / dt)
    d = max(int(round(tau / dt)), 0)
    N = A.shape[0]
    xi = np.zeros((n + 1, N))
    xi[0] = xi0 if xi0 is not None else 0.1 * np.ones(N)
    for k in range(n):
        delayed = xi[k - d] if k - d >= 0 else xi[0]
        xi[k + 1] = xi[k] + dt * (A @ xi[k] + B @ delayed)
    return np.linspace(0, T, n + 1), xi


# ============================================================
# Full simulation -> data for the figure
# ============================================================
def run_simulation(p, p_mk, p_ac, tau):
    Mbar, converged, iters, hist = solve_equilibrium(p, p_mk, p_ac, np.array([1.0, 1.0]))
    A, B = jacobians(Mbar, p, p_mk, p_ac)
    eig0 = np.linalg.eigvals(A + B)
    tau_c, cands = critical_delay(A, B)

    # Panel B: response map T(M^e) with uniform expectation M^e
    xs = np.linspace(0.0, 10.0, 400)
    T = np.array([stationary_given_expectation(x * np.ones(p.N), p, p_mk, p_ac)[0] for x in xs])
    g = T - xs
    roots = []
    for i in range(len(xs) - 1):
        if g[i] * g[i + 1] < 0:
            r = xs[i] - g[i] * (xs[i + 1] - xs[i]) / (g[i + 1] - g[i])
            slope = (T[i + 1] - T[i]) / (xs[i + 1] - xs[i])
            roots.append((r, abs(slope) < 1))

    # Panel D: simulate delayed linear dynamics below / above tau_c
    if tau_c is not None:
        tau_lo, tau_hi = 0.5 * tau_c, 1.5 * tau_c
    else:
        tau_lo, tau_hi = tau, 3 * tau
    t_lo, x_lo = simulate_dde(A, B, tau_lo)
    t_hi, x_hi = simulate_dde(A, B, tau_hi)

    return dict(Mbar=Mbar, converged=converged, iterations=iters, hist=hist,
                A=A, B=B, eig0=eig0, tau_c=tau_c, cands=cands,
                xs=xs, T=T, roots=roots,
                tau_lo=tau_lo, tau_hi=tau_hi, t=t_lo, x_lo=x_lo, x_hi=x_hi)


# ============================================================
# Plotting (uses ONLY the simulation output)
# ============================================================
def gamma_density(w, mean, shape=2.0):
    scale = mean / shape
    return w ** (shape - 1) * np.exp(-w / scale) / (gamma_fn(shape) * scale ** shape)


def plot_panels(res, fname="fig4_combined"):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    Mbar = res["Mbar"]
    w = np.linspace(0, 10, 500)

    # A: density reconstructed from simulated mean, projected to scalar M_i
    ax = axs[0, 0]
    for i, c in enumerate(["navy", "tab:green"]):
        m = gamma_density(w, Mbar[i])
        ax.plot(w, m, color=c, lw=1.8, label=f"$m_{i+1}(w)$")
        ax.fill_between(w, 0, m, color=c, alpha=0.15)
        Mi = _trapz(w * m, w) / _trapz(m, w)
        ax.axvline(Mi, color=c, ls="--", lw=2)
        ax.text(Mi + 0.1, 0.05 + 0.1 * i, f"$M_{i+1}\\approx {Mbar[i]:.2f}$", fontsize=8, color=c)
    ax.set_title("A. Monotonicity & Scalar Coupling", fontweight="bold")
    ax.set_xlabel("Waiting-time state $w$"); ax.set_ylabel("Population density $m_i(w)$")
    ax.set_xlim(0, 10); ax.legend(fontsize=7)

    # B: response map and fixed points from the model
    ax = axs[0, 1]
    ax.plot(res["xs"], res["T"], color="darkgreen", lw=2.5, label=r"Best response $T(M^e)$")
    ax.plot(res["xs"], res["xs"], "k:", lw=1.5, label=r"Consistency $M=M^e$")
    for r, stable in res["roots"]:
        ax.plot(r, r, "o" if stable else "s", color="navy" if stable else "darkred", ms=8, zorder=5)
    ax.set_title(f"B. Equilibrium Existence & Multiplicity ({len(res['roots'])} fixed pt)", fontweight="bold")
    ax.set_xlabel(r"Expected congestion $M^e$"); ax.set_ylabel(r"Actual congestion $T(M^e)$")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.legend(fontsize=7, loc="upper left")

    # C: perturbation of reconstructed density along leading eigen-direction
    ax = axs[1, 0]
    eigvals, eigvecs = np.linalg.eig(res["A"] + res["B"])
    v = np.real(eigvecs[:, np.argmax(eigvals.real)])
    v = v / np.max(np.abs(v))
    eps = 0.3
    m0 = gamma_density(w, Mbar[0])
    mp = gamma_density(w, max(Mbar[0] + eps * v[0], 1e-3))
    ax.plot(w, m0, "k--", lw=1.5, label=r"Stationary $m^*(w)$")
    ax.plot(w, mp, color="darkorange", lw=2.2, label=r"Perturbed $m^*+\delta m$")
    ax.fill_between(w, m0, mp, color="orange", alpha=0.4, label=r"$\delta m$")
    ax.set_title("C. Linearization Around Equilibrium", fontweight="bold")
    ax.set_xlabel("Waiting-time state $w$"); ax.set_ylabel("Population density $m(w,t)$")
    ax.set_xlim(0, 10); ax.legend(fontsize=7)

    # D: simulated delayed linearized dynamics
    ax = axs[1, 1]
    tc = res["tau_c"]
    tag = f"$\\tau_c={tc:.3f}$" if tc is not None else "no crossing found"
    ax.plot(res["t"], res["x_lo"][:, 0], color="navy", lw=1.4, label=f"$\\tau={res['tau_lo']:.3f}$")
    ax.plot(res["t"], res["x_hi"][:, 0], color="darkred", lw=1.4, label=f"$\\tau={res['tau_hi']:.3f}$")
    ax.axhline(0, color="gray", ls=":")
    ax.set_title(f"D. Delay Stability ({tag})", fontweight="bold")
    ax.set_xlabel("Time $t$"); ax.set_ylabel(r"Perturbation $\xi_1(t)$"); ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(fname + ".png", dpi=300)
    fig.savefig(fname + ".pdf")
    plt.show()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    p = Params()
    res = run_simulation(p, p_mk=0.6, p_ac=0.4, tau=0.8)

    print("Converged:", res["converged"], "after", res["iterations"], "iterations")
    print("Final residual:", res["hist"][-1])
    print("M* =", res["Mbar"])
    print("A =\n", res["A"]); print("B =\n", res["B"])
    print("Eigenvalues of A+B (tau=0):", res["eig0"])
    print("tau_c =", res["tau_c"])
    print("Fixed points of T (M, stable?):", res["roots"])

    plot_panels(res)
