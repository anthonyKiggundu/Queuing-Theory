'''
    The UAV’s and UEs’ animation (for the first altitude).
    Per-UE offloading fraction and per-UE offloading energy (shown on the plot and in the console at each step).
    After all runs, plots of:
        Total energy for all altitudes.
        Per-UE offloading fraction vs time (for each altitude).
        Per-UE offloading energy vs time (for each altitude).

    It includes the UAV battery constraint and is ready for 2 UEs
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from math import atan2, degrees, sqrt, log

# ========== SYSTEM & MODEL PARAMETERS ==========
B = 20e6
fc = 2e9
c = 3e8
N0 = 10**((-174 - 30)/10)
G = 10**(20/10)
D_bits = 28_000
T_slot = 0.01
r = D_bits/(B*T_slot)
area = 1000
f_uav = 2e9
f_bs = 5e9
E_rem = 2000.
W_kg = 5.0
A = 0.3
Cd = 0.4
c1 = 1.0

def pathloss_38_811_a2g(tx, rx):
    d3 = np.linalg.norm(tx - rx)
    d2 = np.linalg.norm(tx[:2] - rx[:2])
    theta = degrees(atan2(tx[2] - rx[2], d2))
    a, b = 9.61, 0.16
    PLoS = 1/(1 + a * np.exp(-b * (theta - a)))
    pl_l = 28 + 22 * np.log10(d3) + 20 * np.log10(fc / 1e9)
    pl_n = 13.54 + 39.08 * np.log10(d3) + 20 * np.log10(fc / 1e9) - 0.6 * (tx[2] - 1.5)
    pl_dB = PLoS * pl_l + (1 - PLoS) * pl_n
    pl_dB += np.random.randn() * 4
    return 10 ** (pl_dB / 10), theta

def rician_doppler_gain(theta, speed_mps):
    if theta >= 70:
        Kdb = 10
    elif theta >= 30:
        Kdb = 5
    else:
        Kdb = 0
    K = 10 ** (Kdb / 10)
    mu = sqrt(K / (K + 1))
    sig = sqrt(1 / (2 * (K + 1)))
    fD = speed_mps * fc / c
    rho = np.sinc(2 * fD * T_slot)
    h = mu + sig * (np.random.randn() + 1j * np.random.randn())
    noise = sig * (np.random.randn() + 1j * np.random.randn())
    h2 = rho * h + sqrt(1 - rho ** 2) * noise
    return abs(h2) ** 2

def air_density(h_m):
    ρ0 = 1.225
    H = 8500
    return ρ0 * np.exp(-h_m / H)

def propulsion_energy_rotary(W_kg, v_mps, h_m, t_sec, A=0.3, Cd=0.4, c1=1.0):
    g = 9.81
    W = W_kg * g
    ρ = air_density(h_m)
    P_hover = c1 * pow(W, 1.5) / np.sqrt(ρ)
    P_drag = 0.5 * Cd * A * ρ * v_mps
    return (P_hover + P_drag) * t_sec

def compute_total_energy_general(alpha_vec, C_list, uav_pos, bs_pos, ue_positions,
                                kappa=1e-27, W_kg=5.0, A=0.3, Cd=0.4, c1=1.0,
                                altitude=100, speed_kmh=60, area=1000, T_slot=0.01):
    M = len(alpha_vec)
    speed_mps = speed_kmh * 1000 / 3600
    t_flight = area / speed_mps
    n_slots = int(np.ceil(t_flight / T_slot))
    total_energy = 0
    for m in range(M):
        C = C_list[m]
        f = C / T_slot
        E_cpu = kappa * C * f ** 2 * (1 - alpha_vec[m])
        E_comm = 0
        for t in range(n_slots):
            ue_pos = np.array([ue_positions[m][0], ue_positions[m][1], 0.0])
            PL_lin, theta = pathloss_38_811_a2g(uav_pos, ue_pos)
            g = rician_doppler_gain(theta, speed_mps)
            Kmt = log(2) * PL_lin * N0 * B / (G * g)
            E_base = Kmt * (2 ** r - 1) * T_slot
            E_comm += alpha_vec[m] * E_base
        total_energy += E_cpu + E_comm
    total_energy += propulsion_energy_rotary(W_kg, speed_mps, altitude, t_flight, A, Cd, c1)
    return total_energy

def compute_per_ue_offload_energy(alpha_vec, C_list, uav_pos, bs_pos, ue_positions,
                                  kappa=1e-27, W_kg=5.0, A=0.3, Cd=0.4, c1=1.0,
                                  altitude=100, speed_kmh=60, area=1000, T_slot=0.01):
    # Returns array of per-UE offload energies (without UAV propulsion)
    M = len(alpha_vec)
    speed_mps = speed_kmh * 1000 / 3600
    t_flight = area / speed_mps
    n_slots = int(np.ceil(t_flight / T_slot))
    ue_energies = np.zeros(M)
    for m in range(M):
        C = C_list[m]
        f = C / T_slot
        E_cpu = kappa * C * f ** 2 * (1 - alpha_vec[m])
        E_comm = 0
        for t in range(n_slots):
            ue_pos = np.array([ue_positions[m][0], ue_positions[m][1], 0.0])
            PL_lin, theta = pathloss_38_811_a2g(uav_pos, ue_pos)
            g = rician_doppler_gain(theta, speed_mps)
            Kmt = log(2) * PL_lin * N0 * B / (G * g)
            E_base = Kmt * (2 ** r - 1) * T_slot
            E_comm += alpha_vec[m] * E_base
        ue_energies[m] = E_cpu + E_comm
    return ue_energies

def optimize_offloading_detailed(C, uav_pos, bs_pos, ue_positions,
                                 altitude=100, speed_kmh=60, area=1000, T_slot=0.01,
                                 f_uav=2e9, f_bs=5e9, E_rem=2000., W_kg=5.0, A=0.3, Cd=0.4, c1=1.0):
    N = len(C)
    def total_energy(alpha_u):
        alpha_vec = np.clip(np.array(alpha_u), 0, 1)
        return compute_total_energy_general(
            alpha_vec, C, uav_pos, bs_pos, ue_positions,
            W_kg=W_kg, A=A, Cd=Cd, c1=c1, altitude=altitude, speed_kmh=speed_kmh, area=area, T_slot=T_slot
        )
    # Constraints
    bounds = Bounds(0, 1)
    cpu_constraint = LinearConstraint(C, lb=None, ub=f_uav * T_slot)
    sum_constraint = LinearConstraint(np.ones(N), lb=None, ub=N)
    # Battery constraint: total UAV energy must not exceed E_rem
    def uav_energy_constraint(alpha_u):
        return E_rem - compute_total_energy_general(
            np.clip(np.array(alpha_u), 0, 1), C, uav_pos, bs_pos, ue_positions,
            W_kg=W_kg, A=A, Cd=Cd, c1=c1, altitude=altitude, speed_kmh=speed_kmh, area=area, T_slot=T_slot
        )
    battery_constraint = NonlinearConstraint(uav_energy_constraint, 0, np.inf)

    x0 = np.full(N, 0.5)
    result = minimize(
        total_energy, x0,
        bounds=bounds,
        constraints=[cpu_constraint, sum_constraint, battery_constraint]
    )
    return result.x, result.fun

def run_user_animation_with_optimization(altitude, user_positions, show_animation=False):
    T, N, _ = user_positions.shape
    C = np.array([1e8, 1.2e8])  # adjust for more UEs as needed
    area_size = area
    uav_pos = np.array([area_size / 2, area_size / 2, altitude])
    bs_pos = np.array([0, 0, 0])
    speed_kmh = 60

    all_alpha_u = []
    all_energy = []
    per_ue_energy = []

    if show_animation:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, area_size)
        ax.set_ylim(0, area_size)
        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Y position (m)")
        ax.scatter(uav_pos[0], uav_pos[1], c='red', marker='^', s=100, label='UAV')
        ax.scatter(bs_pos[0], bs_pos[1], c='blue', marker='s', s=100, label='BS')
        ue_scatter = ax.scatter(user_positions[0, :, 0],
                                user_positions[0, :, 1],
                                c=['orange', 'green'][:N], s=50, label='UEs')
        ax.legend(loc='upper right')
        text_out = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')

    def update(frame):
        coords = user_positions[frame]
        alpha_u_opt, total_e = optimize_offloading_detailed(
            C, uav_pos, bs_pos, coords,
            altitude=altitude,
            speed_kmh=speed_kmh,
            area=area_size,
            T_slot=T_slot,
            f_uav=f_uav,
            f_bs=f_bs,
            E_rem=E_rem,
            W_kg=W_kg,
            A=A,
            Cd=Cd,
            c1=c1
        )
        per_ue_E = compute_per_ue_offload_energy(
            alpha_u_opt, C, uav_pos, bs_pos, coords,
            W_kg=W_kg, A=A, Cd=Cd, c1=c1,
            altitude=altitude, speed_kmh=speed_kmh, area=area_size, T_slot=T_slot
        )
        all_alpha_u.append(alpha_u_opt)
        per_ue_energy.append(per_ue_E)
        all_energy.append(total_e)
        if show_animation:
            ue_scatter.set_offsets(coords)
            ue_text = "\n".join([
                f"UE{i+1}: α_UAV={alpha_u_opt[i]:.3f}, E_offload={per_ue_E[i]/1000:.2f} kJ"
                for i in range(N)
            ])
            txt = (f"Frame {frame}\n{ue_text}\nTotal Energy: {total_e/1000:.2f} kJ")
            text_out.set_text(txt)
            print(txt)
            return ue_scatter, text_out

    if show_animation:
        ani = FuncAnimation(fig, update, frames=T, interval=600, blit=True)
        plt.title(f"UAV Altitude: {altitude} m")
        plt.show()
        # Run through all frames again for data collection
        for frame in range(T):
            update(frame)
    else:
        for frame in range(T):
            update(frame)
    return (np.array(all_energy) / 1000, np.array(all_alpha_u), np.array(per_ue_energy) / 1000)  # energies in kJ

if __name__ == "__main__":
    T = 15
    N = 2
    area_size = area
    step_std = 20

    # Set random seed for reproducibility!
    np.random.seed(42)
    user_positions = np.zeros((T, N, 2))
    user_positions[0] = np.random.rand(N, 2) * area_size
    for t in range(1, T):
        step = np.random.randn(N, 2) * step_std
        user_positions[t] = np.clip(user_positions[t - 1] + step, 0, area_size)

    altitudes = [100, 150, 200]
    energies_by_alt = {}
    alpha_by_alt = {}
    per_ue_energy_by_alt = {}

    for i, alt in enumerate(altitudes):
        print(f"\n=== Simulating for UAV altitude {alt} m ===")
        show_anim = (i == 0)  # Show animation only for first altitude
        energies, alpha_u, per_ue_energy = run_user_animation_with_optimization(
            altitude=alt,
            user_positions=user_positions,
            show_animation=show_anim
        )
        energies_by_alt[alt] = energies
        alpha_by_alt[alt] = alpha_u
        per_ue_energy_by_alt[alt] = per_ue_energy

    # Plot total energy
    plt.figure(figsize=(8, 5))
    for alt in altitudes:
        plt.plot(energies_by_alt[alt], marker='o', label=f'Altitude {alt} m')
    plt.xlabel('Time Step')
    plt.ylabel('Total Energy [kJ]')
    plt.title('Comparison of Optimized Total Energy at Different UAV Altitudes\n(UAV battery constraint enforced)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot per-UE offloading for each altitude
    for alt in altitudes:
        alpha_u = alpha_by_alt[alt]
        plt.figure(figsize=(8, 5))
        for i in range(N):
            plt.plot(alpha_u[:, i], marker='o', label=f'UE{i+1} α_UAV')
        plt.xlabel('Time Step')
        plt.ylabel('Optimal Offloading Fraction to UAV')
        plt.title(f'Per-UE Offloading Fraction vs Time (Altitude {alt} m)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # Plot per-UE offload energy for each altitude
    for alt in altitudes:
        per_ue_energy = per_ue_energy_by_alt[alt]
        plt.figure(figsize=(8, 5))
        for i in range(N):
            plt.plot(per_ue_energy[:, i], marker='o', label=f'UE{i+1} Offload Energy [kJ]')
        plt.xlabel('Time Step')
        plt.ylabel('Per-UE Offload Energy [kJ]')
        plt.title(f'Per-UE Offload Energy vs Time (Altitude {alt} m)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
