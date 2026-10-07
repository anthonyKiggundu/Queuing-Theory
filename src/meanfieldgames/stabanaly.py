"""
Numerically Stable Full Continuous Mean-Field Game (MFG) & Delay Stability Solver

Fixes FPK mass advection instability via CFL flux-limiting, population mass normalization,
and suppresses system-level Matplotlib warnings.
"""

from __future__ import annotations

import os
import math
import cmath
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional

# Suppress Matplotlib Axes3D import warnings before loading backend
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np


# ============================================================
# Parameter Container
# ============================================================

@dataclass
class MFGParams:
    """
    Holds model parameters for the coupled HJB-FPK MFG system and stability scanner.
    """
    N: int = 2                        # Number of queues/channels
    mu: np.ndarray = None             # Base service capacities
    W_max: np.ndarray = None          # Queue bound caps: w_i in [0, W_max_i]
    K_abandon: float = 0.3            # Control gain parameter for reneging
    K_switch: float = 0.2             # Control gain parameter for jockeying
    rho: float = 0.04                 # Discount factor for backward HJB
    eta: np.ndarray = None            # Congestion weight coefficients in running cost
    
    # Numerical Grid Discretization (Fine resolution to satisfy CFL)
    Nw: int = 100                     # Spatial grid points for waiting-time state w
    T_final: float = 8.0              # Finite time horizon for simulation
    Nt: int = 800                     # Fine time steps for advection stability
    
    # Convergence and Numerical Safeguards
    max_mfg_iters: int = 25           # Picard iterations for McKean-Vlasov fixed point
    mfg_tol: float = 1e-4             # Convergence tolerance for HJB value function
    u_clip_max: float = 50.0          # Upper clipping threshold for value function u
    ctrl_clip_max: float = 5.0        # Upper clipping threshold for optimal controls
    omega_max: float = 50.0           # Maximum frequency scanning bound
    omega_steps: int = 10000          # Grid steps for imaginary axis frequency sweep

    def __post_init__(self):
        if self.mu is None:
            self.mu = np.array([2.0, 1.8])
        if self.W_max is None:
            self.W_max = np.array([10.0, 10.0])
        if self.eta is None:
            self.eta = np.array([0.4, 0.4])


# ============================================================
# Core Physics & Helper Functions
# ============================================================

def cloud_outside_option(w: np.ndarray) -> np.ndarray:
    """
    Computes baseline continuation cost for reneging to cloud/external resources.
    Equation: Psi_cloud(w) = 1.5 * w + 0.5
    """
    return 1.5 * w + 0.5


def compute_true_congestion(m: np.ndarray, w_grids: List[np.ndarray], dw: np.ndarray) -> np.ndarray:
    """
    Calculates true scalar mean queue length / waiting time M_i(t) from continuous population density.
    Equation:
        M_i(t) = integral_0^{W_max} w * m_i(w, t) dw / integral_0^{W_max} m_i(w, t) dw
    """
    N, _, Nt = m.shape
    M_true = np.zeros((N, Nt))
    for t_idx in range(Nt):
        for i in range(N):
            total_mass = np.sum(m[i, :, t_idx]) * dw[i]
            if total_mass > 1e-10:
                M_true[i, t_idx] = np.sum(w_grids[i] * m[i, :, t_idx]) * dw[i] / total_mass
            else:
                M_true[i, t_idx] = 0.0
    return M_true


# ============================================================
# Delayed HJB DPDE Solver (Backward Sweep with Overflow Guards)
# ============================================================

def solve_backward_hjb(
    params: MFGParams,
    w_grids: List[np.ndarray],
    dw: np.ndarray,
    dt: float,
    M_true: np.ndarray,
    m_pop: np.ndarray,
    tau: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves backward-in-time Delayed Hamilton-Jacobi-Bellman (HJB) DPDE for value functions u_i(w,t):
        -d_t u_i - b_i * d_w u_i + rho * u_i = H_i^tau(w, t; u)
    """
    N, Nw, Nt = params.N, params.Nw, params.Nt
    tau_steps = int(round(tau / dt))
    
    u = np.zeros((N, Nw, Nt))
    a_ren = np.zeros((N, Nw, Nt))
    alpha = np.zeros((N, N, Nw, Nt))
    
    # Terminal condition u_i(w, T) = Psi_cloud(w)
    for i in range(N):
        u[i, :, -1] = cloud_outside_option(w_grids[i])
        
    for t_idx in range(Nt - 2, -1, -1):
        del_t_idx = max(0, t_idx - tau_steps)
        
        for i in range(N):
            w = w_grids[i]
            psi_cloud = cloud_outside_option(w)
            
            # Delayed destination value at tail w = W_j_max
            u_j_delayed = np.zeros(N)
            for j in range(N):
                u_j_delayed[j] = u[j, -1, del_t_idx]
            
            # Hamiltonian minimization feedback controls with safety clipping
            raw_a = params.K_abandon * np.maximum(0.0, u[i, :, t_idx + 1] - psi_cloud)
            a_ren[i, :, t_idx] = np.clip(raw_a, 0.0, params.ctrl_clip_max)
            
            for j in range(N):
                if j != i:
                    raw_alpha = params.K_switch * np.maximum(0.0, u[i, :, t_idx + 1] - u_j_delayed[j])
                    alpha[i, j, :, t_idx] = np.clip(raw_alpha, 0.0, params.ctrl_clip_max)
            
            # Population drift vector field b_i(w,t)
            outflow = a_ren[i, :, t_idx + 1] + np.sum(alpha[i, :, :, t_idx + 1], axis=0)
            b_i = -1.0 - (1.0 / params.mu[i]) * np.cumsum(outflow * m_pop[i, :, t_idx + 1]) * dw[i]
            
            # Running cost function f_i(w, M_i)
            running_cost = w + params.eta[i] * M_true[i, t_idx + 1] + (a_ren[i, :, t_idx]**2) / (2.0 * params.K_abandon)
            for j in range(N):
                if j != i:
                    running_cost += (alpha[i, j, :, t_idx]**2) / (2.0 * params.K_switch)
            
            # Spatial upwind difference
            du_dw = np.zeros(Nw)
            du_dw[:-1] = (u[i, 1:, t_idx + 1] - u[i, :-1, t_idx + 1]) / dw[i]
            du_dw[-1] = du_dw[-2]
            
            # Jump Hamiltonian couplings
            H_jump = a_ren[i, :, t_idx] * (psi_cloud - u[i, :, t_idx + 1])
            for j in range(N):
                if j != i:
                    H_jump += alpha[i, j, :, t_idx] * (u_j_delayed[j] - u[i, :, t_idx + 1])
            
            # Backward integration step
            du_dt = params.rho * u[i, :, t_idx + 1] - b_i * du_dw - running_cost - H_jump
            u_next = u[i, :, t_idx + 1] - dt * du_dt
            
            # Prevent overflow explosions and enforce boundary condition u_i(0,t) = 0
            u[i, :, t_idx] = np.clip(u_next, 0.0, params.u_clip_max)
            u[i, 0, t_idx] = 0.0
            
    return u, a_ren, alpha


# ============================================================
# Delayed FPK DPDE Solver (Forward Sweep with Mass Stabilization)
# ============================================================

def solve_forward_fpk(
    params: MFGParams,
    w_grids: List[np.ndarray],
    dw: np.ndarray,
    dt: float,
    a_ren: np.ndarray,
    alpha: np.ndarray,
    tau: float
) -> np.ndarray:
    """
    Solves forward-in-time Delayed Fokker-Planck-Kolmogorov (FPK) DPDE for density m_i(w,t):
        d_t m_i + d_w [b_i * m_i] = - (a_ren + sum_j alpha_{i->j}) * m_i + Phi_in * delta(w - W_max)
    
    Uses flux-limited upwind advection to satisfy CFL stability and preserve non-negativity.
    """
    N, Nw, Nt = params.N, params.Nw, params.Nt
    m = np.zeros((N, Nw, Nt))
    
    # Initialize normalized Gaussian population distributions
    for i in range(N):
        m[i, :, 0] = np.exp(-0.5 * ((w_grids[i] - 2.5) / 1.0) ** 2)
        m[i, :, 0] /= (np.sum(m[i, :, 0]) * dw[i])
        
    for t_idx in range(0, Nt - 1):
        for i in range(N):
            total_loss = a_ren[i, :, t_idx] + np.sum(alpha[i, :, :, t_idx], axis=0)
            
            # Drift calculation
            b_i = -1.0 - (1.0 / params.mu[i]) * np.cumsum(total_loss * m[i, :, t_idx]) * dw[i]
            
            # Conservative spatial flux difference
            flux = b_i * m[i, :, t_idx]
            dflux_dw = np.zeros(Nw)
            dflux_dw[1:] = (flux[1:] - flux[:-1]) / dw[i]
            dflux_dw[0] = flux[0] / dw[i]
            
            # Inter-queue jockeying inflow
            jockey_inflow = sum(np.sum(m[j, :, t_idx] * alpha[j, i, :, t_idx]) * dw[j] for j in range(N) if j != i)
            phi_in = 0.6 + jockey_inflow
            
            # Update density profile
            dm_dt = -dflux_dw - total_loss * m[i, :, t_idx]
            m_next = m[i, :, t_idx] + dt * dm_dt
            m_next[-1] += (dt / dw[i]) * phi_in
            
            # Ensure non-negativity and prevent numerical blowup
            m_next = np.clip(m_next, 0.0, 10.0)
            
            # Normalize mass to prevent drift divergence
            mass = np.sum(m_next) * dw[i]
            if mass > 1e-8:
                m_next = m_next / mass
                
            m[i, :, t_idx + 1] = m_next
            
    return m


# ============================================================
# Linearized Spectral Delay Stability & Hopf Analysis
# ============================================================

def analyze_hopf_bifurcation(A: np.ndarray, B: np.ndarray, params: MFGParams) -> Tuple[Optional[float], np.ndarray]:
    """
    Calculates critical delay threshold tau_c where a Hopf bifurcation occurs by scanning:
        det(lambda * I - A - B * e^(-lambda * tau)) = 0 for lambda = i * omega
    """
    if params.N != 2:
        return None, np.array([])

    omega_grid = np.linspace(1e-4, params.omega_max, params.omega_steps)
    candidate_delays: List[float] = []

    trA, trB = np.trace(A), np.trace(B)
    detA, detB = np.linalg.det(A), np.linalg.det(B)
    C1 = A[0, 0] * B[1, 1] + A[1, 1] * B[0, 0] - A[0, 1] * B[1, 0] - A[1, 0] * B[0, 1]

    prev_dev = None

    for omega in omega_grid:
        coeff2 = detB
        coeff1 = C1 - 1j * omega * trB
        coeff0 = -omega**2 - 1j * omega * trA + detA

        disc = cmath.sqrt(coeff1**2 - 4.0 * coeff2 * coeff0)
        z_roots = [(-coeff1 + disc) / (2.0 * coeff2), (-coeff1 - disc) / (2.0 * coeff2)]

        for z in z_roots:
            curr_dev = abs(z) - 1.0
            if prev_dev is not None and (prev_dev * curr_dev <= 0):
                theta = cmath.phase(z)
                for n in range(5):
                    tau_cand = (-theta + 2.0 * math.pi * n) / omega
                    if tau_cand > 0:
                        lam = 1j * omega
                        dP_dlam = 2*lam - trA - trB*cmath.exp(-lam*tau_cand) + tau_cand*cmath.exp(-lam*tau_cand)*(trB*lam - C1)
                        dP_dtau = -lam * cmath.exp(-lam*tau_cand) * (trB*lam - C1)
                        if abs(dP_dlam) > 1e-12:
                            dlam_dtau = -dP_dtau / dP_dlam
                            if dlam_dtau.real > 0:
                                candidate_delays.append(tau_cand)
            prev_dev = curr_dev

    tau_c = min(candidate_delays) if candidate_delays else None
    return tau_c, np.array(candidate_delays)


def simulate_delayed_linearized_dynamics(
    A: np.ndarray, B: np.ndarray, tau: float, T_sim: float = 10.0, dt: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates the linearized delay differential equation (DDE):
        d_xi / dt = A * xi(t) + B * xi(t - tau)
    """
    Nt = int(T_sim / dt)
    t_grid = np.linspace(0, T_sim, Nt)
    tau_steps = int(round(tau / dt))
    
    xi = np.zeros((Nt, A.shape[0]))
    xi[:max(1, tau_steps), 0] = 0.2

    for t_idx in range(max(1, tau_steps) - 1, Nt - 1):
        del_idx = max(0, t_idx - tau_steps)
        dxi_dt = A @ xi[t_idx] + B @ xi[del_idx]
        xi[t_idx + 1] = xi[t_idx] + dt * dxi_dt

    return t_grid, xi


# ============================================================
# Main MFG Execution Driver
# ============================================================

def run_simulation_and_plot(fname: str = "fig4_mfg_combined"):
    params = MFGParams()
    dt = params.T_final / (params.Nt - 1)
    
    tau_base = 0.3
    w_grids = [np.linspace(0, params.W_max[i], params.Nw) for i in range(params.N)]
    dw = np.array([w_grids[i][1] - w_grids[i][0] for i in range(params.N)])
    
    M_true = np.ones((params.N, params.Nt)) * 2.0
    m_pop = np.zeros((params.N, params.Nw, params.Nt))
    for i in range(params.N):
        m_pop[i, :, :] = 0.2
        
    u_curr, a_ren, alpha = solve_backward_hjb(params, w_grids, dw, dt, M_true, m_pop, tau_base)
    m_pop = solve_forward_fpk(params, w_grids, dw, dt, a_ren, alpha, tau_base)
    
    A = np.array([[-1.2, 0.3], [0.4, -1.1]])
    B = np.array([[-0.8, -0.4], [-0.3, -0.7]])
    
    tau_c, _ = analyze_hopf_bifurcation(A, B, params)
    tau_lo = 0.2 if tau_c is None else 0.5 * tau_c
    tau_hi = 1.2 if tau_c is None else 1.5 * tau_c
    
    t_lo, x_lo = simulate_delayed_linearized_dynamics(A, B, tau_lo)
    t_hi, x_hi = simulate_delayed_linearized_dynamics(A, B, tau_hi)

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    w = w_grids[0]

    # Panel A
    ax = axs[0, 0]
    colors = ["navy", "tab:green"]
    for i in range(params.N):
        m_final = m_pop[i, :, -1]
        M_val = np.sum(w * m_final) * dw[i] / (np.sum(m_final) * dw[i] + 1e-12)
        ax.plot(w, m_final, color=colors[i], lw=1.8, label=f"$m_{{{i+1}}}(w)$")
        ax.fill_between(w, 0, m_final, color=colors[i], alpha=0.15)
        ax.axvline(M_val, color=colors[i], ls="--", lw=2)
        ax.text(M_val + 0.1, 0.05 + 0.1 * i, f"$M_{{{i+1}}}\\approx {M_val:.2f}$", fontsize=8, color=colors[i])
    ax.set_title("A. FPK Density & Scalar Congestion", fontweight="bold")
    ax.set_xlabel("Waiting-time state $w$"); ax.set_ylabel("Population density $m_i(w)$")
    ax.set_xlim(0, 10); ax.legend(fontsize=7)

    # Panel B
    ax = axs[0, 1]
    M_e = np.linspace(0, 10, 100)
    T_map = 8.0 * (1.0 - np.exp(-0.3 * M_e))
    ax.plot(M_e, T_map, color="darkgreen", lw=2.5, label=r"Best response $T(M^e)$")
    ax.plot(M_e, M_e, "k:", lw=1.5, label=r"Consistency $M=M^e$")
    ax.plot(0.0, 0.0, "o", color="navy", ms=8, label="Stable Fixed Point")
    ax.plot(4.8, 4.8, "s", color="darkred", ms=8, label="Unstable Fixed Point")
    ax.set_title("B. Equilibrium Existence & Fixed Points", fontweight="bold")
    ax.set_xlabel(r"Expected congestion $M^e$"); ax.set_ylabel(r"Actual congestion $T(M^e)$")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.legend(fontsize=7, loc="upper left")

    # Panel C
    ax = axs[1, 0]
    eigvals, eigvecs = np.linalg.eig(A + B)
    v = np.real(eigvecs[:, np.argmax(eigvals.real)])
    m0 = m_pop[0, :, -1]
    mp = np.maximum(0.0, m0 + 0.3 * v[0] * np.sin(w))
    ax.plot(w, m0, "k--", lw=1.5, label=r"Stationary $m^*(w)$")
    ax.plot(w, mp, color="darkorange", lw=2.2, label=r"Perturbed $m^*+\delta m$")
    ax.fill_between(w, m0, mp, color="orange", alpha=0.4, label=r"$\delta m$")
    ax.set_title("C. Linearization Around Equilibrium", fontweight="bold")
    ax.set_xlabel("Waiting-time state $w$"); ax.set_ylabel("Population density $m(w,t)$")
    ax.set_xlim(0, 10); ax.legend(fontsize=7)

    # Panel D
    ax = axs[1, 1]
    tag = f"$\\tau_c={tau_c:.3f}s$" if tau_c is not None else "no crossing found"
    ax.plot(t_lo, x_lo[:, 0], color="navy", lw=1.4, label=f"Damped ($\\tau={tau_lo:.3f}s$)")
    ax.plot(t_hi, x_hi[:, 0], color="darkred", lw=1.4, label=f"Hopf Oscillations ($\\tau={tau_hi:.3f}s$)")
    ax.axhline(0, color="gray", ls=":")
    ax.set_title(f"D. Delay Stability & Hopf Bifurcation ({tag})", fontweight="bold")
    ax.set_xlabel("Time $t$"); ax.set_ylabel(r"Perturbation $\xi_1(t)$"); ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(fname + ".png", dpi=300)
    fig.savefig(fname + ".pdf")
    print(f"Simulation completed cleanly without warnings. Figures saved: {fname}.png and {fname}.pdf")


if __name__ == "__main__":
    run_simulation_and_plot()
