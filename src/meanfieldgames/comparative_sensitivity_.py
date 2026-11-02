"""
Comparative sensitivity of the delayed queue-telemetry loop:
Markov (Mk) vs Actor--Critic (AC) tenants.

Illustrates Section "Comparative Sensitivity and Instability: Markov vs.
Actor--Critic" with a model that is computed, not assumed:
  * class derivatives (eqn:markov-derivs, eqn:ac-derivs) and the mixture
    G(p) = p_Mk G^Mk + p_AC G^AC (eqn:G-mixture), checked by finite differences;
  * class outflow gains (eqn:class-gains), saturation and the hard-switch limit;
  * the AC corrective conditions (eqn:ac-corrective) and Assumption H5;
  * T(lambda) = H (lambda I - A)^{-1} G, T(0) = D Phi(Q*)   (Lemma reduced);
  * candidate delays, crossing direction and tau_c (eqn:candidates,
    eqn:crossing-direction, eqn:tau-c), validated by simulating the delay system;
  * how p and gamma move tau_c: full recomputation vs the fixed-equilibrium
    mixture formula (eqn:T-mixture) vs the closed-form scalar formula.

Model reduction used here (exact for the deterministic model)
-------------------------------------------------------------
With w the projected remaining wait, drift b_i from eq:drift-derived and
injection at the back of the queue W_i = N_i / mu_i, the density
m_i(w) = mu_i on [0, W_i] is invariant: the departure term -D_i m_i is exactly
cancelled by the compression -d_w(b_i m_i), and requests injected at W_i arrive
with density mu_i. Every equilibrium has this form, so
    M_i = W_i / 2,   H = I / 2,
and the FPK reduces to one equation per queue for the back of the queue:
    dW_i/dt = Lam_i/mu_i + (1/mu_i) sum_j mu_j int_0^{W_j} alpha_{j->i} dw
              - 1 - int_0^{W_i} D_i dw,         D_i = a_i^ren + sum_j alpha_{i->j},
with all rates evaluated at the delayed, discounted report  gamma * W(t - tau) / 2.
A positive equilibrium therefore needs overload (Lam_i > mu_i): a deterministic
fluid queue below capacity is empty.

Markov destination index: a switcher joins queue j at its back, W_j = 2 M_j,
so the Markov rule compares w + Mhat_i with W_j + Mhat_j = 3 Mhat_j
(dest_c = 2). The rule in eq:markov_joc uses I_j(0) = Mhat_j (dest_c = 0).

All parameter values are illustrative.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import root


# ======================================================================
# 1. Model
# ======================================================================
def sig(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))   # logistic, overflow-safe


class QueueLoop:
    def __init__(self, mu=(1.0, 0.8), Lam=(1.3, 1.0), gamma=1.0, p_ac=0.5,
                 r_mk=0.15, k_mk=2.0, dest_c=2.0,
                 beta=2.0, abar=0.8, albar=0.8,
                 theta=(-2.0, 0.4, 0.4, -0.2, 0.4), theta_r=(-3.0, 0.4, 0.4),
                 nq=4001):
        self.mu = np.asarray(mu, float)
        self.Lam = np.asarray(Lam, float)
        self.N = len(self.mu)
        self.gamma = np.full(self.N, float(gamma))
        self.p_ac = p_ac
        self.r_mk, self.k_mk, self.dest_c = r_mk, k_mk, dest_c
        self.beta, self.abar, self.albar = beta, abar, albar
        self.theta = np.asarray(theta, float)
        self.theta_r = np.asarray(theta_r, float)
        self.nq = nq

    # ---- class policies (eq:markov_ren/joc, eq:ac_ren/joc_sigmoid) ----
    def ren(self, cls, i, w, Q):
        Mh = self.gamma * Q
        if cls == "mk":
            return self.r_mk * (w + Mh[i])
        r0, r1, r2 = self.theta_r
        return self.abar * sig(self.beta * (r0 + r1 * w + r2 * Mh[i]))

    def jock(self, cls, i, j, w, Q):
        Mh = self.gamma * Q
        if cls == "mk":
            return self.k_mk * np.maximum(0.0, w + Mh[i] - (1 + self.dest_c) * Mh[j])
        t0, t1, t2, t3, t4 = self.theta
        z = t0 + t1 * w + t2 * Mh[i] + t3 * Mh[j] + t4 * (Mh[i] - Mh[j])
        return self.albar * sig(self.beta * z)

    # ---- class derivatives w.r.t. Q_l (eqn:markov-derivs, eqn:ac-derivs) ----
    def dren(self, cls, i, l, w, Q):
        g = self.gamma
        if l != i:
            return np.zeros_like(w)
        if cls == "mk":
            return np.full_like(w, self.r_mk * g[i])
        r0, r1, r2 = self.theta_r
        s = sig(self.beta * (r0 + r1 * w + r2 * g[i] * Q[i]))
        return self.abar * self.beta * s * (1 - s) * r2 * g[i]

    def djock(self, cls, i, j, l, w, Q):
        g = self.gamma
        Mh = g * Q
        out = np.zeros_like(w)
        if cls == "mk":
            on = (w + Mh[i] - (1 + self.dest_c) * Mh[j] > 0).astype(float)
            if l == i:
                out += self.k_mk * on * g[i]
            if l == j:
                out -= self.k_mk * on * (1 + self.dest_c) * g[j]
            return out
        t0, t1, t2, t3, t4 = self.theta
        s = sig(self.beta * (t0 + t1 * w + t2 * Mh[i] + t3 * Mh[j] + t4 * (Mh[i] - Mh[j])))
        fac = self.albar * self.beta * s * (1 - s)
        if l == i:
            out += fac * (t2 + t4) * g[i]
        if l == j:
            out += fac * (t3 - t4) * g[j]
        return out

    def _mix(self, f, *args):
        return (1 - self.p_ac) * f("mk", *args) + self.p_ac * f("ac", *args)

    def _int(self, fn, W):
        w = np.linspace(0.0, W, self.nq)
        return np.trapezoid(fn(w), w)

    # ---- back-of-queue dynamics dW/dt = F(W, Q) ----
    def F(self, W, Q):
        N, out = self.N, np.zeros(self.N)
        for i in range(N):
            dep = self._int(lambda w: self._mix(self.ren, i, w, Q)
                            + sum(self._mix(self.jock, i, j, w, Q) for j in range(N) if j != i), W[i])
            inflow = sum(self.mu[j] * self._int(lambda w, j=j: self._mix(self.jock, j, i, w, Q), W[j])
                         for j in range(N) if j != i)
            out[i] = self.Lam[i] / self.mu[i] + inflow / self.mu[i] - 1.0 - dep
        return out

    @staticmethod
    def M(W):
        return W / 2.0

    def equilibrium(self, W0=None):
        W0 = np.full(self.N, 3.0) if W0 is None else W0
        r = root(lambda W: self.F(W, self.M(W)), W0, method="hybr", tol=1e-12)
        if not r.success or np.any(r.x <= 0):
            raise RuntimeError("no positive equilibrium (check overload Lam > mu)")
        return r.x

    # ---- Jacobians ----
    def A(self, W, Q):
        """Current-state Jacobian at fixed telemetry (Leibniz rule at the back of the queue)."""
        N, A = self.N, np.zeros((self.N, self.N))
        for i in range(N):
            wi = np.array([W[i]])
            A[i, i] = -(self._mix(self.ren, i, wi, Q)[0]
                        + sum(self._mix(self.jock, i, j, wi, Q)[0] for j in range(N) if j != i))
            for j in range(N):
                if j != i:
                    A[i, j] = self.mu[j] / self.mu[i] * self._mix(self.jock, j, i, np.array([W[j]]), Q)[0]
        return A

    def G_class(self, W, Q, cls):
        """G^c: departures (incl. acceleration of requests behind) and jockeying inflow."""
        N, G = self.N, np.zeros((self.N, self.N))
        for i in range(N):
            for l in range(N):
                dep = self._int(lambda w: self.dren(cls, i, l, w, Q)
                                + sum(self.djock(cls, i, j, l, w, Q) for j in range(N) if j != i), W[i])
                inflow = sum(self.mu[j] * self._int(lambda w, j=j: self.djock(cls, j, i, l, w, Q), W[j])
                             for j in range(N) if j != i)
                G[i, l] = inflow / self.mu[i] - dep
        return G

    def G(self, W, Q):
        return (1 - self.p_ac) * self.G_class(W, Q, "mk") + self.p_ac * self.G_class(W, Q, "ac")

    def H(self):
        return 0.5 * np.eye(self.N)

    def class_outflow_gain(self, W, Q, cls):
        """eqn:class-gains with m_i^* = mu_i on [0, W_i]."""
        N = self.N
        return np.array([self.mu[i] * self._int(
            lambda w: self.dren(cls, i, i, w, Q)
            + sum(self.djock(cls, i, j, i, w, Q) for j in range(N) if j != i), W[i])
            for i in range(N)])

    # ---- finite-difference references ----
    def A_fd(self, W, Q, h=1e-6):
        return np.column_stack([(self.F(W + h * e, Q) - self.F(W - h * e, Q)) / (2 * h) for e in np.eye(self.N)])

    def G_fd(self, W, Q, h=1e-6):
        return np.column_stack([(self.F(W, Q + h * e) - self.F(W, Q - h * e)) / (2 * h) for e in np.eye(self.N)])

    def DPhi_fd(self, W_star, h=1e-5):
        """Jacobian of Phi(Q) = M(steady state for frozen Q), by finite differences."""
        Q0 = self.M(W_star)
        def Phi(Q):
            return self.M(root(lambda W: self.F(W, Q), W_star, method="hybr", tol=1e-12).x)
        return np.column_stack([(Phi(Q0 + h * e) - Phi(Q0 - h * e)) / (2 * h) for e in np.eye(self.N)])


# ======================================================================
# 2. Loop transfer matrix and critical delay
# ======================================================================
def T_of(lam, A, G, H):
    return H @ np.linalg.solve(lam * np.eye(A.shape[0]) - A, G)


def critical_delay(A, G, H, w_max=30.0, n=3000, n_rep=2):
    """Candidate delays from |mu|=1 and the phase condition; tau_c = first rightward crossing."""
    omegas = np.geomspace(1e-4, w_max, n)
    smod = lambda w: np.sort(np.abs(np.linalg.eigvals(T_of(1j * w, A, G, H))))
    mods = np.array([smod(w) for w in omegas])
    cands = []
    for b in range(mods.shape[1]):
        f = mods[:, b] - 1.0
        for k in np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]:
            lo, hi = omegas[k], omegas[k + 1]
            flo = smod(lo)[b] - 1
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                fm = smod(mid)[b] - 1
                if np.sign(fm) == np.sign(flo):
                    lo, flo = mid, fm
                else:
                    hi = mid
            w = 0.5 * (lo + hi)
            ev = np.linalg.eigvals(T_of(1j * w, A, G, H))
            mu = ev[np.argmin(np.abs(np.abs(ev) - 1))]
            th = np.angle(mu) % (2 * np.pi)
            th = 2 * np.pi if th == 0 else th
            near = lambda x: (lambda e: e[np.argmin(np.abs(e - mu))])(np.linalg.eigvals(T_of(1j * x, A, G, H)))
            dw = 1e-6 * w
            dmu_dlam = (near(w + dw) - near(w - dw)) / (2 * dw) / 1j
            for r in range(n_rep):
                tau = (th + 2 * np.pi * r) / w
                dlam = -(1j * w) / (tau - dmu_dlam / mu)          # eqn:crossing-direction
                cands.append((tau, w, dlam.real))
    cands.sort()
    out = dict(candidates=cands, tau1=np.inf, omega1=np.nan, dir1=np.nan, tau_c=np.inf, omega_c=np.nan)
    if cands:
        out.update(tau1=cands[0][0], omega1=cands[0][1], dir1=cands[0][2])
        right = [c for c in cands if c[2] > 0]
        if right:
            out.update(tau_c=right[0][0], omega_c=right[0][1])
    return out


def tau_c_scalar(kappa, a):
    """Closed form of the first-order single-mode illustration (eqn:tau-c-scalar)."""
    return np.inf if kappa <= 1.0 else np.arccos(-1.0 / kappa) / (a * np.sqrt(kappa ** 2 - 1.0))


def kappa_of(T0):
    """Loop gain of the most strongly corrective mode of T(0)."""
    return -np.min(np.linalg.eigvals(T0).real)


def analyse(model, W0=None):
    W = model.equilibrium(W0)
    Q = model.M(W)
    A, G, H = model.A(W, Q), model.G(W, Q), model.H()
    return dict(W=W, Q=Q, A=A, G=G, H=H, cd=critical_delay(A, G, H),
                stab0=np.linalg.eigvals(A + G @ H).real.max())


# ======================================================================
# 3. Nonlinear delayed simulation
# ======================================================================
def simulate(model, W_star, tau, T_end=120.0, dt=0.01, eps=0.05):
    nlag, steps = int(round(tau / dt)), int(T_end / dt)
    W = W_star * (1 + eps * np.array([1.0, -1.0]))
    hist = [W.copy()] * (nlag + 1)
    out = np.zeros((steps, model.N))
    for s in range(steps):
        Q0, Q1 = model.M(hist[-(nlag + 1)]), model.M(hist[-nlag])
        Qm = 0.5 * (Q0 + Q1)
        f = lambda x, Q: model.F(np.maximum(x, 1e-9), Q)
        k1 = f(W, Q0)
        k2 = f(W + 0.5 * dt * k1, Qm)
        k3 = f(W + 0.5 * dt * k2, Qm)
        k4 = f(W + dt * k3, Q1)
        W = np.maximum(W + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
        hist.append(W.copy())
        if len(hist) > nlag + 2:
            hist.pop(0)
        out[s] = W
    return np.arange(steps) * dt, out


# ======================================================================
# 4. Driver
# ======================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    p0 = 0.5
    base = QueueLoop(p_ac=p0)
    b = analyse(base)
    W, Q, A, G, H = b["W"], b["Q"], b["A"], b["G"], b["H"]

    print("=== Checks at the base mixture p_AC = %.1f ===" % p0)
    print("equilibrium back of queue W* =", W, " mean waits Q* =", Q)
    print("H3  eig(A)                  =", np.linalg.eigvals(A))
    print("    stable at tau = 0: max Re eig(A + G H) = %.4f" % b["stab0"])
    print("A   analytic vs FD, max err = %.2e" % np.abs(A - base.A_fd(W, Q)).max())
    print("G   analytic vs FD, max err = %.2e" % np.abs(G - base.G_fd(W, Q)).max())
    T0 = T_of(0, A, G, H)
    print("Lemma: T(0) - DPhi(Q*), max err = %.2e" % np.abs(T0 - base.DPhi_fd(W)).max())
    print("H4  eig(T(0))               =", np.linalg.eigvals(T0), "(1 must not be one of them)")
    print("H5  diag T(0)               =", np.diag(T0), "(negative = corrective own-queue response)")
    for cls in ("mk", "ac"):
        print("class outflow gain %-2s       =" % cls, base.class_outflow_gain(W, Q, cls))
    t0, t1, t2, t3, t4 = base.theta
    print("AC corrective (eqn:ac-corrective): theta_r2>=0 %s, theta2+theta4>=0 %s, theta3-theta4<=0 %s"
          % (base.theta_r[2] >= 0, t2 + t4 >= 0, t3 - t4 <= 0))
    print("candidates (tau, omega, Re dlambda/dtau):")
    for c in b["cd"]["candidates"][:4]:
        print("   tau=%.3f  omega=%.3f  dir=%+.4f" % c)

    # ---- (a) eigenvalue loci of T(i omega) for three mixtures ----
    omegas = np.geomspace(1e-3, 30, 800)
    loci, results = {}, {}
    for p in (0.0, 0.5, 1.0):
        r = analyse(QueueLoop(p_ac=p))
        results[p] = r
        loci[p] = np.array([np.linalg.eigvals(T_of(1j * w, r["A"], r["G"], r["H"])) for w in omegas])

    # ---- (b) tau_c versus p_AC ----
    ps = np.linspace(0, 1, 41)
    tau_full, Wg = [], None
    for p in ps:
        r = analyse(QueueLoop(p_ac=p), Wg)
        Wg = r["W"]
        tau_full.append(r["cd"]["tau_c"])
    Gmk, Gac = base.G_class(W, Q, "mk"), base.G_class(W, Q, "ac")
    tau_fixed = [critical_delay(A, (1 - p) * Gmk + p * Gac, H)["tau_c"] for p in ps]   # eqn:T-mixture
    a0 = -np.linalg.eigvals(A).real.max()
    k_mk, k_ac = kappa_of(T_of(0, A, Gmk, H)), kappa_of(T_of(0, A, Gac, H))
    tau_scalar = [tau_c_scalar((1 - p) * k_mk + p * k_ac, a0) for p in ps]

    # ---- (c) tau_c versus gamma (AC-only population, which has finite tau_c) ----
    gs = np.linspace(0.3, 1.2, 37)
    tau_g_full, Wg = [], None
    for g in gs:
        r = analyse(QueueLoop(p_ac=1.0, gamma=g), Wg)
        Wg = r["W"]
        tau_g_full.append(r["cd"]["tau_c"])
    r1 = results[1.0]
    kap1, a1 = kappa_of(T_of(0, r1["A"], r1["G"], r1["H"])), -np.linalg.eigvals(r1["A"]).real.max()
    tau_g_scalar = [tau_c_scalar(g * kap1, a1) for g in gs]

    # ---- (d) AC outflow gain versus beta, at the base equilibrium density ----
    # Two AC weight sets: decision boundaries inside the occupied range [0, W*]
    # (gain tends to the hard-switch limit) and outside it (policy saturates,
    # gain vanishes). Markov gain for comparison.
    betas = np.geomspace(0.2, 200, 80)
    inside = dict(theta=(-0.6, 0.4, 0.4, -0.2, 0.4), theta_r=(-0.9, 0.4, 0.4))
    gain_in, gain_out = [], []
    for be in betas:
        gain_in.append(QueueLoop(p_ac=p0, beta=be, nq=40001, **inside).class_outflow_gain(W, Q, "ac")[0])
        gain_out.append(QueueLoop(p_ac=p0, beta=be, nq=40001).class_outflow_gain(W, Q, "ac")[0])
    gain_mk = base.class_outflow_gain(W, Q, "mk")[0]
    g0, mu0 = base.gamma[0], base.mu[0]
    # hard-switch limit: gamma*mu*(abar*theta_r2/|theta_r1| + albar*(theta2+theta4)/|theta1|)
    ti, tri = inside["theta"], inside["theta_r"]
    hard = g0 * mu0 * (base.abar * tri[2] / abs(tri[1]) + base.albar * (ti[2] + ti[4]) / abs(ti[1]))
    print("AC gain at beta=200: boundaries inside %.4f (limit %.4f), outside %.2e"
          % (gain_in[-1], hard, gain_out[-1]))

    # ---- (e) corrective vs non-corrective AC weights ----
    nonc = QueueLoop(p_ac=1.0, theta=(-2.0, 0.4, -0.6, -0.2, 0.1), theta_r=(-3.0, 0.4, -0.4))
    rn = analyse(nonc)
    diag_c = np.diag(T_of(0, r1["A"], r1["G"], r1["H"]))
    diag_n = np.diag(T_of(0, rn["A"], rn["G"], rn["H"]))
    print("non-corrective AC: diag T(0) =", diag_n, " max Re eig at tau=0 = %.4f" % rn["stab0"],
          " tau_c = %.3f" % rn["cd"]["tau_c"])

    # ---- (f) simulation below and above tau_c (AC-only) ----
    tc = r1["cd"]["tau_c"]
    sims = {f: simulate(QueueLoop(p_ac=1.0, nq=801), r1["W"], f * tc) for f in (0.8, 1.2)}
    print("AC-only tau_c = %.3f, omega_c = %.3f (period %.2f)"
          % (tc, r1["cd"]["omega_c"], 2 * np.pi / r1["cd"]["omega_c"]))

    # ---- figure ----
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    cols = {0.0: "tab:blue", 0.5: "tab:purple", 1.0: "tab:red"}
    names = {0.0: "Markov only", 0.5: "50/50 mix", 1.0: "Actor-Critic only"}

    a_ = ax[0, 0]
    th = np.linspace(0, 2 * np.pi, 300)
    a_.plot(np.cos(th), np.sin(th), "k:", lw=1, label="|μ| = 1")
    for p, L in loci.items():
        for k in range(L.shape[1]):
            a_.plot(L[:, k].real, L[:, k].imag, color=cols[p], lw=1.6, label=names[p] if k == 0 else None)
    a_.set_title("(a) Eigenvalues of T(iω), ω from 0 to ∞")
    a_.set_xlabel("Re μ"); a_.set_ylabel("Im μ"); a_.set_aspect("equal"); a_.legend(fontsize=8)
    a_.grid(alpha=.3)

    a_ = ax[0, 1]
    a_.plot(ps, tau_full, "k-", lw=2, label="full: equilibrium recomputed")
    a_.plot(ps, tau_fixed, "--", color="tab:green", label="fixed equilibrium, eqn:T-mixture")
    a_.plot(ps, tau_scalar, ":", color="tab:orange", lw=2, label="scalar formula, κ(p) linear")
    a_.set_title("(b) Critical delay vs composition")
    a_.set_xlabel("$p_{AC}$"); a_.set_ylabel("$τ_c$ (∞ = robust to any delay)")
    a_.set_ylim(0, 40); a_.legend(fontsize=8); a_.grid(alpha=.3)

    a_ = ax[0, 2]
    a_.plot(gs, tau_g_full, "k-", lw=2, label="full: equilibrium recomputed")
    a_.plot(gs, tau_g_scalar, ":", color="tab:orange", lw=2, label="scalar formula, κ ∝ γ")
    a_.axvline(1 / kap1, color="tab:orange", ls="--", lw=1, label="γ = 1/κ(γ=1)")
    a_.set_title("(c) Critical delay vs telemetry confidence (AC only)")
    a_.set_xlabel("γ"); a_.set_ylabel("$τ_c$"); a_.set_ylim(0, 25); a_.legend(fontsize=8); a_.grid(alpha=.3)

    a_ = ax[1, 0]
    a_.semilogx(betas, gain_in, "tab:red", lw=2, label="AC, decision boundaries inside queue")
    a_.semilogx(betas, gain_out, "tab:red", lw=2, ls="--", label="AC, boundaries outside queue")
    a_.axhline(hard, color="tab:red", ls=":", lw=1.2, label="hard-switch limit (β→∞)")
    a_.axhline(gain_mk, color="tab:blue", lw=2, label="Markov")
    a_.set_title("(d) Own-queue outflow gain vs AC temperature β")
    a_.set_xlabel("β"); a_.set_ylabel("extra outflow per unit report, queue 1")
    a_.legend(fontsize=8); a_.grid(alpha=.3)

    a_ = ax[1, 1]
    x = np.arange(base.N)
    a_.bar(x - 0.18, diag_c, 0.36, color="tab:green", label="corrective weights")
    a_.bar(x + 0.18, diag_n, 0.36, color="tab:gray", label="non-corrective weights")
    a_.axhline(0, color="k", lw=.8)
    a_.set_xticks(x, ["queue 1", "queue 2"])
    a_.set_title("(e) Own-queue response diag T(0) (H5 needs < 0)")
    a_.legend(fontsize=8); a_.grid(alpha=.3, axis="y")

    a_ = ax[1, 2]
    for f, c in ((0.8, "tab:green"), (1.2, "tab:red")):
        ts, Ws = sims[f]
        a_.plot(ts, Ws[:, 0] - r1["W"][0], color=c, lw=1.2, label="τ = %.1f τ_c" % f)
    a_.set_title("(f) Simulated back-of-queue deviation, queue 1 (AC only)")
    a_.set_xlabel("t"); a_.set_ylabel("$W_1(t) - W_1^*$"); a_.legend(fontsize=8); a_.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig("comparative_sensitivity_analysis.png", dpi=200)
    plt.savefig("comparative_sensitivity_analysis.pdf")
    print("Figures saved.")
