"""
Which information model is better?  Markov (Mk) vs Actor--Critic (AC).

Two measures:
  1. Gap to the optimum: expected cost of a request that follows its class
     policy, minus the cost of the HJB best response, in the same population
     environment (stationary equilibrium). Both are computed from the
     stationary versions of the HJB and of the policy-evaluation equation,
         rho*u - b(w) u'(w) = H(w, u),      u(0) = 0  (service),
     with the running cost f = w + eta*M + a^2/(2K_ab) + sum alpha^2/(2K_sw),
     reneging to the outside option Psi_cloud, and switching to the value at
     the back of the destination queue (eq:delayed-destination-value; at a
     steady state the delayed and current values coincide).
  2. Delay tolerance: tau_c, and the long-run average cost per arriving
     request when the telemetry delay exceeds tau_c (population accounting
     along the simulated delay system).

Fair comparison: instead of one hand-picked setting per class, each class's
aggressiveness is swept (Markov: k_mk and r_mk scaled together; AC: inverse
temperature beta), giving a trade-off curve of gap versus tau_c per class.

Uses QueueLoop, critical_delay and T_of from comparative_sensitivity.py
(same folder). All parameter values are illustrative.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.integrate import solve_ivp
from comparative_sensitivity import QueueLoop, analyse

# ---- cost parameters of the running cost (eq:running-cost) and exit payoff ----
COST = dict(eta=1.0, K_ab=1.0, K_sw=1.0, Psi=6.0, rho=0.0)


# ======================================================================
# 1. Stationary drift, HJB benchmark and class-policy evaluation
# ======================================================================
def drift(model, i, W, Q, nq=2001):
    """b_i(w) = -1 - int_0^w D_i, tabulated on [0, W_i] (density mu_i)."""
    w = np.linspace(0.0, W[i], nq)
    D = model._mix(model.ren, i, w, Q) + sum(model._mix(model.jock, i, j, w, Q)
                                              for j in range(model.N) if j != i)
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (D[1:] + D[:-1]) * np.diff(w))))
    return w, -1.0 - cum


def solve_value(model, W, Q, policy, c=COST, iters=200, tol=1e-10):
    """
    policy = "opt" -> HJB best response u_i(w);
    policy = "mk"/"ac" -> cost v_i(w) of a request that always follows that class rule.
    The environment (drift, M) is the equilibrium population (W, Q).
    Returns the values at the back of each queue, v_i(W_i), and the curves.
    """
    N, M_true = model.N, model.M(W)
    grids = [drift(model, i, W, Q) for i in range(N)]
    back = np.zeros(N)                          # values at the back of each queue
    curves = [None] * N
    for _ in range(iters):
        new = np.zeros(N)
        for i in range(N):
            wg, bg = grids[i]
            bfun = lambda x: np.interp(x, wg, bg)

            def rhs(x, u):
                u = u[0]
                run = x + c["eta"] * M_true[i]
                if policy == "opt":
                    a = c["K_ab"] * max(u - c["Psi"], 0.0)
                    al = [c["K_sw"] * max(u - back[j], 0.0) if j != i else 0.0 for j in range(N)]
                else:
                    xa = np.array([x])
                    a = model.ren(policy, i, xa, Q)[0]
                    al = [model.jock(policy, i, j, xa, Q)[0] if j != i else 0.0 for j in range(N)]
                Hval = (run + a ** 2 / (2 * c["K_ab"]) + a * (c["Psi"] - u)
                        + sum(al[j] ** 2 / (2 * c["K_sw"]) + al[j] * (back[j] - u)
                              for j in range(N) if j != i))
                return [(c["rho"] * u - Hval) / bfun(x)]

            sol = solve_ivp(rhs, (0.0, W[i]), [0.0], rtol=1e-9, atol=1e-11, dense_output=True)
            curves[i] = sol
            new[i] = sol.y[0, -1]
        if np.abs(new - back).max() < tol:
            back = new
            break
        back = new
    return back, curves


def gap_to_optimum(model, W, Q, cls):
    """Arrival-weighted cost gap (absolute and relative) of class cls versus the HJB best response."""
    u, _ = solve_value(model, W, Q, "opt")
    v, _ = solve_value(model, W, Q, cls)
    wts = model.Lam / model.Lam.sum()
    return float(wts @ (v - u)), float(wts @ (v - u)) / float(wts @ u), u, v


# ======================================================================
# 2. Population cost along a delayed trajectory
# ======================================================================
def cost_rate(model, W, Q, c=COST, nq=401):
    """Aggregate cost per unit time: running cost of all queued requests + Psi per reneging."""
    total = 0.0
    M_true = model.M(W)
    p = model.p_ac
    for i in range(model.N):
        if W[i] <= 0:
            continue
        w = np.linspace(0.0, W[i], nq)
        act = 0.0
        ren_mix = 0.0
        for cls, pc in (("mk", 1 - p), ("ac", p)):
            if pc == 0:
                continue
            a = model.ren(cls, i, w, Q)
            al = sum(model.jock(cls, i, j, w, Q) for j in range(model.N) if j != i)
            al2 = sum(model.jock(cls, i, j, w, Q) ** 2 for j in range(model.N) if j != i)
            act = act + pc * (a ** 2 / (2 * c["K_ab"]) + al2 / (2 * c["K_sw"]))
            ren_mix = ren_mix + pc * a
        dens = model.mu[i]
        total += dens * np.trapezoid(w + c["eta"] * M_true[i] + act + c["Psi"] * ren_mix, w)
    return total


def run_delay(model, W_star, tau, period, n_per=60, n_avg=15, dt=0.02, eps=0.05, every=5):
    """
    Simulate the delay system for n_per oscillation periods and return the average
    cost per arriving request (and the oscillation amplitude) over the last n_avg periods.
    """
    T_end, t_avg = n_per * period, n_avg * period
    nlag, steps = int(round(tau / dt)), int(T_end / dt)
    W = W_star * (1 + eps * np.array([1.0, -1.0]))
    hist = [W.copy()] * (nlag + 1)
    acc, n_acc, amp = 0.0, 0, 0.0
    for s in range(steps):
        Q0, Q1 = model.M(hist[-(nlag + 1)]), model.M(hist[-nlag])
        Qm = 0.5 * (Q0 + Q1)
        f = lambda x, Q: model.F(np.maximum(x, 1e-9), Q)
        k1 = f(W, Q0); k2 = f(W + 0.5 * dt * k1, Qm); k3 = f(W + 0.5 * dt * k2, Qm); k4 = f(W + dt * k3, Q1)
        W = np.maximum(W + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
        hist.append(W.copy())
        if len(hist) > nlag + 2:
            hist.pop(0)
        if s * dt >= T_end - t_avg and s % every == 0:
            acc += cost_rate(model, W, Q1); n_acc += 1
            amp = max(amp, np.abs(W - W_star).max())
    return acc / n_acc / model.Lam.sum(), amp


# ======================================================================
# 3. Driver
# ======================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # ---- (A) trade-off: gap to optimum vs tau_c, sweeping each class's aggressiveness ----
    scales = np.geomspace(0.03, 3.0, 15)           # Markov: (k_mk, r_mk) * s
    betas = np.geomspace(0.5, 8.0, 12)             # AC: inverse temperature
    rows = {"mk": [], "ac": []}
    for s in scales:
        m = QueueLoop(p_ac=0.0, k_mk=2.0 * s, r_mk=0.15 * s)
        try:
            r = analyse(m)
        except RuntimeError:
            continue
        g_abs, g_rel, u, v = gap_to_optimum(m, r["W"], r["Q"], "mk")
        rows["mk"].append((s, r["cd"]["tau_c"], g_abs, g_rel, r["W"].mean(), float(m.Lam @ v / m.Lam.sum())))
    for be in betas:
        m = QueueLoop(p_ac=1.0, beta=be)
        try:
            r = analyse(m)
        except RuntimeError:
            continue
        g_abs, g_rel, u, v = gap_to_optimum(m, r["W"], r["Q"], "ac")
        rows["ac"].append((be, r["cd"]["tau_c"], g_abs, g_rel, r["W"].mean(), float(m.Lam @ v / m.Lam.sum())))

    print("=== Gap to the HJB optimum and delay tolerance ===")
    for cls, lab in (("mk", "Markov  scale"), ("ac", "AC      beta ")):
        for row in rows[cls]:
            print("%s %6.2f   tau_c %7.2f   class cost %.3f   gap %.3f (%5.1f%%)   mean back of queue %.2f"
                  % (lab, row[0], row[1], row[5], row[2], 100 * row[3], row[4]))

    # ---- (B) value functions at the base settings ----
    base = {"mk": QueueLoop(p_ac=0.0), "ac": QueueLoop(p_ac=1.0)}
    basres = {k: analyse(m) for k, m in base.items()}
    vf = {}
    for k, m in base.items():
        r = basres[k]
        _, ucur = solve_value(m, r["W"], r["Q"], "opt")
        _, vcur = solve_value(m, r["W"], r["Q"], k)
        vf[k] = (r["W"][0], ucur[0], vcur[0])

    # ---- (C) degradation beyond tau_c: cost per request vs delay ----
    deg = {}
    for k, m in base.items():
        r = basres[k]
        tc = r["cd"]["tau_c"]
        period = 2 * np.pi / r["cd"]["omega_c"]
        eq_cost = cost_rate(m, r["W"], r["Q"]) / m.Lam.sum()
        taus = tc * np.array([0.5, 0.9, 1.1, 1.25, 1.5, 2.0])
        fast = QueueLoop(p_ac=m.p_ac, nq=201)              # coarser quadrature for simulation
        vals = [run_delay(fast, r["W"], t, period) for t in taus]
        deg[k] = (tc, eq_cost, taus, np.array([v[0] for v in vals]), np.array([v[1] for v in vals]))
        print("%s: tau_c=%.2f, period at onset %.1f, equilibrium cost per request %.3f"
              % (k, tc, period, eq_cost))
        for t, (cst, amp) in zip(taus, vals):
            print("   tau=%6.2f (%.2f tau_c): cost per request %.3f (%+5.1f%%), amplitude %.3f"
                  % (t, t / tc, cst, 100 * (cst / eq_cost - 1), amp))

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    col = {"mk": "tab:blue", "ac": "tab:red"}
    name = {"mk": "Markov", "ac": "Actor-Critic"}

    a_ = ax[0]
    finite = [r[1] for k in rows for r in rows[k] if np.isfinite(r[1])]
    x_inf = 2.0 * max(finite)                      # where "robust to any delay" is drawn
    for k in ("mk", "ac"):
        R = np.array(rows[k])
        tc = np.where(np.isfinite(R[:, 1]), R[:, 1], x_inf)
        a_.plot(tc, 100 * R[:, 3], "o-", color=col[k], label=name[k])
        for x, y, lab in zip(tc, 100 * R[:, 3], R[:, 0]):
            a_.annotate("%.2g" % lab, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a_.axvline(x_inf, color="gray", ls=":", lw=1)
    a_.text(x_inf, a_.get_ylim()[1] * 0.97, " $τ_c=\\infty$\n (robust to\n any delay)", fontsize=7, va="top")
    a_.set_xscale("log")
    a_.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a_.set_xlabel("critical delay $τ_c$ (higher = more delay-tolerant)")
    a_.set_ylabel("cost gap to HJB optimum (%)  (lower = better)")
    a_.set_title("(a) Trade-off: optimality vs delay tolerance\n(labels: Markov gain scale / AC β)")
    a_.legend(fontsize=8, loc="upper center"); a_.grid(alpha=.3, which="both")

    a_ = ax[1]
    for k in ("mk", "ac"):
        Wb, ucur, vcur = vf[k]
        x = np.linspace(0, Wb, 200)
        a_.plot(x, ucur.sol(x)[0], "--", color=col[k], label="%s environment: HJB optimum" % name[k])
        a_.plot(x, vcur.sol(x)[0], "-", color=col[k], label="%s environment: class policy" % name[k])
    a_.set_xlabel("projected remaining wait w (queue 1)")
    a_.set_ylabel("expected cost to exit")
    a_.set_title("(b) Value of the class policy vs the optimum (base settings)")
    a_.legend(fontsize=8); a_.grid(alpha=.3)

    a_ = ax[2]
    for k in ("mk", "ac"):
        tc, eqc, taus, cst, amp = deg[k]
        a_.plot(taus, 100 * (cst / eqc - 1), "o-", color=col[k], label=name[k])
        a_.axvline(tc, color=col[k], ls=":", lw=1)
    a_.set_xscale("log")
    a_.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a_.set_xlabel("telemetry delay τ (dotted: $τ_c$)")
    a_.set_ylabel("cost per request vs equilibrium (%)")
    a_.set_title("(c) Degradation once oscillations start (base settings)")
    a_.legend(fontsize=8); a_.grid(alpha=.3, which="both")

    plt.tight_layout()
    plt.savefig("information_model_comparison.png", dpi=200)
    plt.savefig("information_model_comparison.pdf")
    print("Figures saved.")
