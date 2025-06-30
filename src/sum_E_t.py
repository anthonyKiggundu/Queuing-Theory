# multi_uav_offload_with_detailed_channel.py
#
# Parameter-Block (Lines 1–10) defines all physical and traffic parameters.
# Scenario Loop (Lines 22–33) over altitudes and speeds generates scenario_id.
# Channel models:
#   pathloss_38_811_a2g: TR 38.811 A2G with LoS/NLoS mixture and log-normal shadowing.
#   rician_doppler_gain: Doppler-dependent Rician fading via AR(1)/J0 model.
# Base energy computation E_base for alpha=1 in energy_base.csv.
# Offload candidates: list all alpha vectors with sum(alpha)<=1 and compute total energy
#   sum_{m,t} E_{m,t}(alpha_m) into offload_candidates.csv.
# This captures all offload options across UAVs, altitudes, and speeds with full
# channel modeling (3GPP 38.811, shadowing, elevation, Doppler due to UAV speed).
# =============================================================================
# energy_base.csv: Basis-Sendeenergie für α=1 je (Szenario, UAV, Slot)

# offload_candidates.csv: Alle Off-load-Vektoren  mit ∑α≤1 und die zugehörige Gesamtenergie

import numpy as np
import pandas as pd
from itertools import product
from math import atan2, degrees, sqrt, log
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
# 1) System- & Traffic-Parameter
# -----------------------------------------------------------------------------
B        = 20e6               # Bandwidth [Hz]
fc       = 2e9                # Carrier freq [Hz]
c        = 3e8                # Speed of light [m/s]
N0       = 10**((-174 - 30)/10)  # Noise PSD [W/Hz]
G        = 10**(20/10)        # Antenna gain (linear)

D_bits   = 28_000             # bits per slot per UAV
T_slot   = 0.01               # slot duration [s]
r        = D_bits/(B*T_slot)  # spectral load (b/Hz)

# -----------------------------------------------------------------------------
# 2) Scenario grid: altitude & speed
# -----------------------------------------------------------------------------
altitudes  = [50, 100, 150, 200, 300]   # UAV heights [m]
speeds_kmh = [30, 60, 90, 120]          # UAV speeds [km/h]

# -----------------------------------------------------------------------------
# 3) Geometry & offload grid
# -----------------------------------------------------------------------------
area   = 1000                          # square area [m]
bs_pos = np.array([area/2, area/2, 0]) # ground BS at center
M      = 3                             # number of UAVs
alphas = np.round(np.arange(0,1.0001,0.05),2)  # offload fractions

# -----------------------------------------------------------------------------
# 4) Detailed Channel Model: 3GPP 38.811 A2G + Rician Doppler
# -----------------------------------------------------------------------------
def pathloss_38_811_a2g(tx, rx):
    d3    = np.linalg.norm(tx - rx)
    d2    = np.linalg.norm(tx[:2] - rx[:2])
    theta = degrees(atan2(tx[2] - rx[2], d2))
    a, b  = 9.61, 0.16
    PLoS  = 1/(1 + a*np.exp(-b*(theta - a)))
    pl_l  = 28 + 22*np.log10(d3) + 20*np.log10(fc/1e9)
    pl_n  = 13.54 + 39.08*np.log10(d3) + 20*np.log10(fc/1e9) - 0.6*(tx[2] - 1.5)
    pl_dB = PLoS*pl_l + (1 - PLoS)*pl_n
    # shadowing sigma=4 dB
    pl_dB += np.random.randn() * 4
    return 10**(pl_dB/10), theta

def rician_doppler_gain(theta, speed_mps):
    if   theta >= 70: Kdb = 10
    elif theta >= 30: Kdb = 5
    else:              Kdb = 0
    K    = 10**(Kdb/10)
    mu   = sqrt(K/(K+1))
    sig  = sqrt(1/(2*(K+1)))
    # Doppler coefficient approx with sinc
    fD   = speed_mps * fc / c
    rho  = np.sinc(2 * fD * T_slot)
    h    = mu + sig*(np.random.randn() + 1j*np.random.randn())
    noise= sig*(np.random.randn() + 1j*np.random.randn())
    h2   = rho*h + sqrt(1 - rho**2)*noise
    return abs(h2)**2

# -----------------------------------------------------------------------------
# 5) Compute base energy for each (scenario, UAV, slot)
# -----------------------------------------------------------------------------
rows = []
scenario_id = 0

for h in altitudes:
    for v_kmh in speeds_kmh:
        speed_mps = v_kmh * 1000/3600
        t_flight  = area/speed_mps
        n_slots   = int(np.ceil(t_flight/T_slot))

        for m in range(M):
            for t in range(n_slots):
                x_t   = max(area - speed_mps*t*T_slot, 0.0)
                uav_p = np.array([x_t, area/2, h])

                PL_lin, theta = pathloss_38_811_a2g(uav_p, bs_pos)
                g_val         = rician_doppler_gain(theta, speed_mps)

                Kmt   = log(2) * PL_lin * N0 * B / (G * g_val)
                Ebase = Kmt * (2**r - 1) * T_slot

                rows.append({
                    "scenario_id": scenario_id,
                    "altitude_m":  h,
                    "speed_kmh":   v_kmh,
                    "uav_id":      m,
                    "slot":        t,
                    "E_base_J":    Ebase
                })
        scenario_id += 1

df_base = pd.DataFrame(rows)
df_base.to_csv("energy_base.csv", index=False)
print("energy_base.csv erstellt mit", df_base.shape, "Zeilen")

# -----------------------------------------------------------------------------
# 6) Build offload candidates (Σα ≤ 1) and compute sum energy
# -----------------------------------------------------------------------------
alpha_combos = [vec for vec in product(alphas, repeat=M) if sum(vec) <= 1.0]
cand_rows    = []

for (sid, h, v_kmh), grp in df_base.groupby(["scenario_id","altitude_m","speed_kmh"]):
    T    = grp.slot.max()+1
    Emat = grp.pivot(index="uav_id", columns="slot", values="E_base_J").values
    speed_mps = v_kmh * 1000/3600

    for vec in alpha_combos:
        fac   = (2**(np.array(vec)*r) - 1)/(2**r - 1)
        E_sum = np.sum(Emat * fac[:,None])
        entry = {
            "scenario_id": sid,
            "altitude_m":  h,
            "speed_kmh":   v_kmh,
            "E_sum_J":     E_sum
        }
        for m in range(M):
            entry[f"alpha_{m}"] = vec[m]
        cand_rows.append(entry)

df_cand = pd.DataFrame(cand_rows)
df_cand.to_csv("offload_candidates.csv", index=False)
print("offload_candidates.csv erstellt mit", df_cand.shape, "Zeilen")

 
print(df_cand.head())





# === Extension 1: Plot Base-Energy vs. Speed for each UAV ===

# Wir nutzen wieder df_base aus energy_base.csv
# und aggregieren über altitude & slot je speed.

for uav_id in range(M):
    df_u = df_base[df_base["uav_id"] == uav_id]
    avg_E_speed = (
        df_u
        .groupby("speed_kmh")["E_base_J"]
        .mean()
        .reset_index()
        .sort_values("speed_kmh")
    )

    plt.figure(figsize=(6,4))
    plt.plot(avg_E_speed["speed_kmh"], avg_E_speed["E_base_J"], marker="s")
    plt.xlabel("Speed [km/h]")
    plt.ylabel("Mean Base-Energy $E_{base}$ [J]")
    plt.title(f"UAV {uav_id}: Mean Base-Energy vs. Speed")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Lese gerade geschriebene energy_base.csv
df_base = pd.read_csv("energy_base.csv")

# Für jeden UAV einzeln plotten
for uav_id in range(M):
    df_u = df_base[df_base["uav_id"] == uav_id]
    # Mittel über speed & slot je altitude
    avg_E = (
        df_u
        .groupby("altitude_m")["E_base_J"]
        .mean()
        .reset_index()
        .sort_values("altitude_m")
    )

    plt.figure(figsize=(6,4))
    plt.plot(avg_E["altitude_m"], avg_E["E_base_J"], marker="o")
    plt.xlabel("Altitude [m]")
    plt.ylabel("Mean Base-Energy $E_{base}$ [J]")
    plt.title(f"UAV {uav_id}: Mean Base-Energy vs. Altitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
'''
   -  Computes total offloading energy for all possible combinations of processing fractions αₘ per device, where αₘ ∈ [0, 1] and ∑αₘ ≤ 1.
   -  Returns the energy values for each valid α-combination.
   -  Plots total energy vs. offloading fractions.
   
      * M devices offload tasks to a UAV. Each device contributes a fraction αₘ of its computation.
      * The UAV’s energy depends on the offload level αₘ using the dynamic power model:
      Eₘ = κ · Cₘ³ / Tₘ², where Tₘ = Cₘ / (fₘ) and fₘ = αₘ-adjusted frequency.
'''   

def total_offload_energy_vs_fraction(
    M=3,
    kappa=1e-27,
    C_list=None,
    total_time=0.05,       # total time constraint per task [s]
    include_flight_energy=False,
    flight_energy_per_uav=5.0  # e.g. in Joules
):
    """
    Computes and plots total energy for all valid α-vectors (∑α ≤ 1)
    across M devices offloading computation to a UAV.

    Parameters:
    - M: number of devices
    - kappa: effective switched capacitance [J/Hz²]
    - C_list: list of CPU cycles per task per device (length M)
    - total_time: time constraint per task [s]
    - include_flight_energy: add UAV propulsion energy if True
    - flight_energy_per_uav: propulsion cost [J] per UAV

    Returns:
    - energy_results: list of (alpha_vec, total_energy)
    - plot of total energy vs α₁ (for fixed M and varying α₁)
    """
    if C_list is None:
        C_list = [1e9] * M  # default: 1 Gcycle per task

    alphas = np.round(np.arange(0, 1.01, 0.05), 2)
    alpha_combos = [
        vec for vec in product(alphas, repeat=M) if sum(vec) <= 1.0
    ]

    energy_results = []

    for alpha_vec in alpha_combos:
        E_total = 0
        for m in range(M):
            α = alpha_vec[m]
            C = C_list[m]
            if α > 0:
                T = total_time
                f = C / T
                E = kappa * C * f**2
            else:
                E = 0
            E_total += E

        if include_flight_energy:
            E_total += flight_energy_per_uav

        energy_results.append((alpha_vec, E_total))

    # Plot: Total energy vs α₁ (fix α₂ = α₃ = 0 if M = 3)
    if M == 3:
        filtered = [res for res in energy_results if res[0][1] == 0 and res[0][2] == 0]
        x_vals = [vec[0][0] for vec in filtered]
        y_vals = [vec[1] for vec in filtered]

        plt.figure(figsize=(6, 4))
        plt.plot(x_vals, y_vals, marker='o')
        plt.xlabel("Offload Fraction α₁")
        plt.ylabel("Total Energy (Joules)")
        plt.title("Total Energy vs Offload Fraction α₁ (others fixed at 0)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return energy_results
    

#results = total_offload_energy_vs_fraction(M=3, kappa=1e-27, C_list=[1e9, 1.2e9, 0.8e9], 
#                           total_time=0.05, include_flight_energy=True, flight_energy_per_uav=4.0
#                        )

# Print top 5 configurations
#for vec, E in results[:5]:
#    print(f"α = {vec}, Energy = {E:.2f} J")
    
    

#######################################################################################
#					includes total energy consumed by the UAV                         #
#######################################################################################

def air_density(h_m):
    ρ0 = 1.225 # sea level
    H = 8500 # scale height
    return ρ0 * np.exp(-h_m / H)

def propulsion_energy_rotary(W_kg, v_mps, h_m, t_sec, A=0.3, Cd=0.4, c1=1.0):
    g = 9.81
    W = W_kg * g
    ρ = air_density(h_m)
    P_hover = c1 *  pow(W,1.5) / np.sqrt(ρ) # (W1.5) / np.sqrt(ρ)
    P_drag = 0.5 * Cd * A * ρ * v_mps #3
    return (P_hover + P_drag) * t_sec
    
    
def generate_alpha_vectors(M, step=0.1, max_samples=1000):
    """Generate valid α-vectors with sum(α) ≤ 1 using coarse grid + sampling."""
    
    grid = np.round(np.arange(0, 1.01, step), 2)
    all_vecs = [vec for vec in product(grid, repeat=M) if sum(vec) <= 1.0]
    
    if len(all_vecs) > max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(all_vecs), max_samples, replace=False)
        return [all_vecs[i] for i in indices]
        
    return all_vecs
    
def generate_alpha_fixed_sum(M, alpha_sum, num_samples=20, seed=42):
    """Generate random α-vectors with sum(α) ≈ alpha_sum and αᵢ ∈ [0,1]."""
    np.random.seed(seed)
    vecs = []
    for _ in range(num_samples):
        base = np.random.rand(M)
        scaled = base / np.sum(base) * alpha_sum
        scaled = np.clip(scaled, 0, 1) # Ensure within [0,1]
        if np.sum(scaled) <= 1.0:
            vecs.append(np.round(scaled, 4))
            
    return vecs  

'''
   We want to include the propulsion energy of the UAV
'''

def compute_total_energy_general(alpha_vec, C_list, kappa=1e-27, W_kg=5.0, A=0.3,
                                   Cd=0.4, c1=1.0, altitude=150, speed_kmh=60, area=1000, T_slot=0.01):
    M = len(alpha_vec)
    speed_mps = speed_kmh * 1000 / 3600
    t_flight = area / speed_mps
    n_slots = int(np.ceil(t_flight / T_slot))

    total_energy = 0

    for m in range(M):
        C = C_list[m]
        f = C / T_slot
        E_cpu = kappa * C * f**2 * (1 - alpha_vec[m])
        E_comm = 0

        for t in range(n_slots):
            x_t = max(area - speed_mps * t * T_slot, 0.0)
            uav_p = np.array([x_t, area / 2, altitude])
            PL_lin, theta = pathloss_38_811_a2g(uav_p, bs_pos)
            g = rician_doppler_gain(theta, speed_mps)
            Kmt = log(2) * PL_lin * N0 * B / (G * g)
            E_base = Kmt * (2**r - 1) * T_slot
            E_comm += alpha_vec[m] * E_base

        total_energy += E_cpu + E_comm

    # Add propulsion energy
    total_energy += propulsion_energy_rotary(W_kg, speed_mps, altitude, t_flight, A, Cd, c1)
    
    return total_energy
    
  
def plot_energy_surface_with_propulsion(M=2, kappa=1e-27, C_list=None,
                                         W_kg=5.0, A=0.3, Cd=0.4, c1=1.0,
                                         altitude=150, speed_kmh=60
                                      ):   
    M = len(alpha_vec)
    speed_mps = speed_kmh * 1000 / 3600
    t_flight = area / speed_mps
    n_slots = int(np.ceil(t_flight / T_slot))                                  
    
    B = 20e6
    fc = 2e9
    c = 3e8
    N0 = 10**((-174 - 30)/10)
    G = 10**(20/10)
    D_bits = 28000
    r = D_bits / (B * T_slot)
    bs_pos = np.array([area / 2, area / 2, 0])

    total_energy = 0

    for m in range(M):
        C = C_list[m]
        f = C / T_slot
        E_cpu = kappa * C * f**2 * (1 - alpha_vec[m])
        E_comm = 0

        for t in range(n_slots):
            x_t = max(area - speed_mps * t * T_slot, 0.0)
            uav_p = np.array([x_t, area / 2, altitude])
            PL_lin, theta = pathloss_38_811_a2g(uav_p, bs_pos)
            g = rician_doppler_gain(theta, speed_mps)
            Kmt = log(2) * PL_lin * N0 * B / (G * g)
            E_base = Kmt * (2**r - 1) * T_slot
            E_comm += alpha_vec[m] * E_base

        total_energy += E_cpu + E_comm

    # Add propulsion energy
    total_energy += propulsion_energy_rotary(W_kg, speed_mps, altitude, t_flight, A, Cd, c1)
    print("\n -> ", total_energy)
    return total_energy
    
 
def visualize_energy_vs_total_offload_old(M=10, step=0.1, num_alpha_samples=25, altitude=200,
                                      speed_kmh=60, W_kg=6.0, A=0.3, Cd=0.4, c1=1.0,
                                      return_data=False, verbose=True):
                                      
    alpha_totals = np.round(np.arange(0.2, 1.01, step), 2)
    C_list = [1e9 + 1e8 * np.random.rand() for _ in range(M)]
    avg_energy = []

    it = tqdm(alpha_totals, desc="Evaluating α-total") if verbose and len(alpha_totals) > 10 else alpha_totals
    for α_sum in it:
        α_vecs = generate_alpha_fixed_sum(M, α_sum, num_samples=num_alpha_samples)
        energies = []
        for α in α_vecs:
            E = compute_total_energy_general(
                α,
                C_list,
                altitude=altitude,
                speed_kmh=speed_kmh,
                W_kg=W_kg,
                A=A,
                Cd=Cd,
                c1=c1
            )
            energies.append(E)
        avg_E = np.mean(energies) if energies else float('nan')
        avg_energy.append(avg_E)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(alpha_totals, avg_energy, marker='o', color='royalblue')
    plt.xlabel("Total Offloading Fraction ∑α (across M devices)")
    plt.ylabel("Average Total Energy [J]")
    plt.title(f"Energy vs Offloaded Load (M={M}, Alt={altitude} m, Speed={speed_kmh} km/h)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if return_data:
        return alpha_totals, avg_energy
    else:
        return None


def visualize_energy_vs_total_offload(M=10, step=0.1, num_alpha_samples=25, altitudes=[200], speeds_kmh=[60], W_kg=6.0, A=0.3, Cd=0.4, c1=1.0):
    plt.figure(figsize=(8, 6))
    alpha_totals = np.round(np.arange(0.2, 1.01, step), 2)

    for altitude in altitudes:
        for speed_kmh in speeds_kmh:
            C_list = [1e9 + 1e8 * np.random.rand() for _ in range(M)]
            avg_energy = []
            for α_sum in alpha_totals:
                α_vecs = generate_alpha_fixed_sum(M, α_sum, num_samples=num_alpha_samples)
                energies = []
                for α in α_vecs:
                    E = compute_total_energy_general(
                        α,
                        C_list,
                        altitude=altitude,
                        speed_kmh=speed_kmh,
                        W_kg=W_kg,
                        A=A,
                        Cd=Cd,
                        c1=c1
                    )
                    if E is not None:
                        energies.append(E)
                avg_E = np.mean(energies) if energies else None
                avg_energy.append(avg_E)
            label = f"Altitude={altitude}m, Speed={speed_kmh}km/h"
            plt.plot(alpha_totals, np.array(avg_energy)/1000, marker='o', label=label)

    plt.xlabel("Total Offloading Fraction ∑α (across M devices)")
    plt.ylabel("Average Total Energy [kJ]")
    plt.title(f"Energy vs Offloaded Load (M={M})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


from tqdm import tqdm  # pip install tqdm, or comment this out if not wanted

def find_best_alpha_M_devices(M=12, step=0.1, max_samples=2000, altitudes=[150], speeds_kmh=[60]):
    results = []
    C_list = [1e9 + 1e8 * np.random.rand() for _ in range(M)]  # heterogeneous tasks
    alpha_candidates = generate_alpha_vectors(M, step=step, max_samples=max_samples)

    for altitude in altitudes:
        for speed_kmh in speeds_kmh:
            best_E = float('inf')
            best_vec = None
            for α in alpha_candidates:
                E = compute_total_energy_general(α, C_list, altitude=altitude, speed_kmh=speed_kmh)
                if E is not None and E < best_E:
                    best_E = E
                    best_vec = α
            print(f"[Alt {altitude} m, Speed {speed_kmh} km/h] ✅ Optimal α (sum={sum(best_vec):.2f}):", np.round(best_vec, 2))
            print(f"🔋 Minimum total energy: {best_E/1000:.3f} kJ")
            results.append({
                "altitude": altitude,
                "speed_kmh": speed_kmh,
                "best_alpha": best_vec,
                "min_energy_J": best_E
            })
            
    return results
    

def find_best_alpha_M_devices_old(M=12, step=0.1, max_samples=1000, altitude=150, speed_kmh=60, verbose=True):
    C_list = [1e9 + 1e8 * np.random.rand() for _ in range(M)]  # heterogeneous tasks
    alpha_candidates = generate_alpha_vectors(M, step=step, max_samples=max_samples)

    best_E = float('inf')
    best_vec = None

    it = tqdm(alpha_candidates, desc="Searching α-vectors") if verbose and len(alpha_candidates) > 100 else alpha_candidates
    for α in it:
        E = compute_total_energy_general(α, C_list, altitude=altitude, speed_kmh=speed_kmh)
        
        if E < best_E: # if E is not None and E < best_E:        
            best_E = E
            best_vec = α

    if verbose:
        print(f"✅ Optimal α (sum={sum(best_vec):.2f}):", np.round(best_vec, 2))
        print(f"🔋 Minimum total energy: {best_E/1000:.3f} kJ")
    return best_vec, best_E
    

# ToDo:: Below is for a single altitude, we need to try this out for different altitudes and speeds

# find_best_alpha_M_devices(M=3, step=0.1, max_samples=100, altitude=200, speed_kmh=90)

# visualize_energy_vs_total_offload(M=3, step=0.1, num_alpha_samples=20, altitude=200, speed_kmh=90 )

# Find best alphas for all combinations
results = find_best_alpha_M_devices(M=4, step=0.2, max_samples=50, altitudes=altitudes, speeds_kmh=speeds_kmh)

# Visualize energy surfaces for these scenarios
visualize_energy_vs_total_offload(M=4, step=0.2, num_alpha_samples=10, altitudes=altitudes, speeds_kmh=speeds_kmh)

#######################################################################################
#					End - includes total energy consumed by the UAV                         #
#######################################################################################

'''
   Let's generate a heatmap that shows—for each UAV configuration (altitude & speed)—the optimal α-combination (α₁, α₂) that minimizes total energy consumption. 
   Each cell in the heatmap will represent a unique (altitude, speed) pair, and the color will reflect either:
     - the minimum energy value (for a gradient heatmap), or
     - the optimal α₁ and α₂ values themselves (with annotations inside the cells).
     
     That is: Loop over each (altitude, speed) configuration. For each, search all valid α₁ + α₂ ≤ 1 combinations.
              then Compute total energy (local CPU + communication + optional flight). Identify the α-combo with minimum total energy.
              and Plot a heatmap showing optimal offloading policy (α₁, α₂) per scenario.
'''

def heatmap_optimal_alpha_per_config( M=2, C_list=None, altitudes=[100, 200, 300], 
                                 speeds_kmh=[30, 60, 90], kappa=1e-27,include_flight_energy=True, power_flight=120
                              ):
                              
    assert M == 2, "Heatmap supports M=2 for α₁, α₂."
    
    if C_list is None:
        C_list = [1e9] * M

    alphas = np.round(np.arange(0, 1.01, 0.1), 2)

    heatmap_data = np.zeros((len(altitudes), len(speeds_kmh)))
    alpha_opt = [["" for _ in speeds_kmh] for _ in altitudes]

    for i, alt in enumerate(altitudes):
        for j, v_kmh in enumerate(speeds_kmh):
            speed_mps = v_kmh * 1000 / 3600
            t_flight = area / speed_mps
            n_slots = int(np.ceil(t_flight / T_slot))
            best_energy = float('inf')
            best_alpha = None

            for α1, α2 in product(alphas, repeat=2):
                if α1 + α2 > 1: continue
                α_vec = [α1, α2]
                total_energy = 0

                for m in range(M):
                    C = C_list[m]
                    f = C / T_slot
                    E_cpu = kappa * C * f**2 * (1 - α_vec[m])

                    E_comm = 0
                    for t in range(n_slots):
                        x_t = max(area - speed_mps * t * T_slot, 0.0)
                        uav_p = np.array([x_t, area / 2, alt])
                        PL_lin, theta = pathloss_38_811_a2g(uav_p, bs_pos)
                        g = rician_doppler_gain(theta, speed_mps)
                        Kmt = log(2) * PL_lin * N0 * B / (G * g)
                        E_base = Kmt * (2**r - 1) * T_slot
                        E_comm += α_vec[m] * E_base

                    total_energy += E_cpu + E_comm

                if include_flight_energy:
                    total_energy += M * power_flight * t_flight

                if total_energy < best_energy:
                    best_energy = total_energy
                    best_alpha = α_vec

            heatmap_data[i, j] = best_energy
            alpha_opt[i][j] = f"({best_alpha[0]:.1f}, {best_alpha[1]:.1f})"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap_data, cmap="YlGnBu")

    # Annotate optimal α in each cell
    for i in range(len(altitudes)):
        for j in range(len(speeds_kmh)):
            text = ax.text(j, i, alpha_opt[i][j],
                       ha="center", va="center", color="black", fontsize=9)

    ax.set_xticks(np.arange(len(speeds_kmh)))
    ax.set_yticks(np.arange(len(altitudes)))
    ax.set_xticklabels([f"{s} km/h" for s in speeds_kmh])
    ax.set_yticklabels([f"{a} m" for a in altitudes])
    ax.set_xlabel("UAV Speed")
    ax.set_ylabel("UAV Altitude")
    ax.set_title("Optimal (α₁, α₂) for Each UAV Configuration")
    fig.colorbar(im, ax=ax, label="Minimum Total Energy [J]")
    plt.tight_layout()
    plt.show()
 
 
 
# heatmap_optimal_alpha_per_config(M=2, C_list=[1e9, 1.2e9],altitudes=[100, 200, 300], speeds_kmh=[30, 60, 90], power_flight=120 )  
'''
   - Compute total energy per α = (α₁, α₂, ..., αₘ) where each αₘ ∈ [0,1], ∑αₘ ≤ 1
   - Include:
       Local computation energy (CMOS CPU model)
       Communication (offloading) energy using real channel/pathloss & Doppler models
       Optional flight (propulsion) energy
   - Generate a 3D surface plot: Total Energy vs. (α₁, α₂)
      
'''

'''
def energy_surface_from_alpha ( M=3, kappa=1e-27, C_list= None, altitude=0,
                                speed_kmh=0, include_flight_energy=False, power_flight=0 
                                ): 

    if C_list is None:
        C_list = [1e9] * M  # 1 Gcycle per device

    alphas = np.round(np.arange(0, 1.01, 0.1), 2)
    speed_mps = speed_kmh * 1000 / 3600
    t_flight = area / speed_mps
    n_slots = int(np.ceil(t_flight / T_slot))

    energy_vals = []
    alpha_grid = []

    for α1 in alphas:
        for α2 in alphas:
            if α1 + α2 > 1: continue
            alpha_vec = [α1, α2]
            total_energy = 0

            for m in range(M):
                # Local CPU energy
                C = C_list[m]
                f = C / T_slot
                E_cpu = kappa * C * f**2 * (1 - alpha_vec[m])

                # Communication (offload) energy over flight
                E_comm = 0
                for t in range(n_slots):
                    x_t = max(area - speed_mps * t * T_slot, 0.0)
                    uav_p = np.array([x_t, area / 2, altitude])
                    PL_lin, theta = pathloss_38_811_a2g(uav_p, bs_pos)
                    g_val = rician_doppler_gain(theta, speed_mps)
                    Kmt = log(2) * PL_lin * N0 * B / (G * g_val)
                    E_base = Kmt * (2**r - 1) * T_slot
                    E_comm += alpha_vec[m] * E_base

                total_energy += E_cpu + E_comm

            if include_flight_energy:
                total_energy += M * power_flight * t_flight

            energy_vals.append(total_energy)
            alpha_grid.append((α1, α2))

    # Reshape for plotting
    A1 = sorted(set(a[0] for a in alpha_grid))
    A2 = sorted(set(a[1] for a in alpha_grid))
    Z = np.zeros((len(A1), len(A2)))

    for (a1, a2), E in zip(alpha_grid, energy_vals):
        i = A1.index(a1)
        j = A2.index(a2)
        Z[i, j] = E

    A1, A2 = np.meshgrid(A1, A2)
    Z = Z.T  # match axes

    # Plot 3D surface
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A1, A2, Z, cmap='viridis', edgecolor='k', alpha=0.85)
    ax.set_xlabel("α₁")
    ax.set_ylabel("α₂")
    ax.set_zlabel("Total Energy [J]")
    ax.set_title("Energy vs Offloading Fractions (α₁, α₂)")
    plt.tight_layout()
    plt.show()

    return alpha_grid, energy_vals


#alpha_grid, energy_vals = energy_surface_from_alpha(M=2,C_list=[1e9, 1.2e9], altitude=150,
#                                          speed_kmh=60,include_flight_energy=True,
#                                          power_flight=120
#                                     )
                                     
#print("\n Showing the values of Alpha Grid :",alpha_grid,"\n Energy Values: " ,energy_vals )



#   Improvements: Compute total energy for each α-combination (α₁, α₂) as before
#                 - Repeat this for various (altitude, speed) pairs
#                 - Create one surface plot per configuration using:
#                 - 3GPP TR 38.811 air-to-ground pathloss model
#                 - Rician Doppler fading
#                 - Show how flight profile affects optimal offloading
#


def plot_energy_surface_multiple_configs(M=2, kappa=1e-27, C_list=None,
                                      altitudes=[100, 200], speeds_kmh=[30, 90],
                                    include_flight_energy=True, power_flight=120
                                ):
                                
    assert M == 2, "3D plot only supports M=2."
    if C_list is None:
        C_list = [1e9] * M
    
    alphas = np.round(np.arange(0, 1.01, 0.1), 2)

    fig = plt.figure(figsize=(14, 6))
    plot_id = 1

    for alt in altitudes:
        for v_kmh in speeds_kmh:
            speed_mps = v_kmh * 1000 / 3600
            t_flight = area / speed_mps
            n_slots = int(np.ceil(t_flight / T_slot))
            alpha_grid = []
            energy_vals = []

            for α1 in alphas:
                for α2 in alphas:
                    if α1 + α2 > 1: continue
                    α_vec = [α1, α2]
                    total_energy = 0

                    for m in range(M):
                        C = C_list[m]
                        f = C / T_slot
                        E_cpu = kappa * C * f**2 * (1 - α_vec[m])

                        E_comm = 0
                        for t in range(n_slots):
                            x_t = max(area - speed_mps * t * T_slot, 0.0)
                            uav_p = np.array([x_t, area/2, alt])
                            PL_lin, theta = pathloss_38_811_a2g(uav_p, bs_pos)
                            g = rician_doppler_gain(theta, speed_mps)
                            Kmt = log(2) * PL_lin * N0 * B / (G * g)
                            E_base = Kmt * (2**r - 1) * T_slot
                            E_comm += α_vec[m] * E_base

                        total_energy += E_cpu + E_comm

                    if include_flight_energy:
                        total_energy += M * power_flight * t_flight

                    energy_vals.append(total_energy)
                    alpha_grid.append((α1, α2))

            A1 = sorted(set(a[0] for a in alpha_grid))
            A2 = sorted(set(a[1] for a in alpha_grid))
            Z = np.zeros((len(A1), len(A2)))

            for (a1, a2), E in zip(alpha_grid, energy_vals):
                i = A1.index(a1)
                j = A2.index(a2)
                Z[i, j] = E

            A1, A2 = np.meshgrid(A1, A2)
            Z = Z.T

            ax = fig.add_subplot(len(altitudes), len(speeds_kmh), plot_id, projection='3d')
            ax.plot_surface(A1, A2, Z, cmap='viridis', edgecolor='k', alpha=0.85)
            ax.set_xlabel("α₁")
            ax.set_ylabel("α₂")
            ax.set_zlabel("Energy [J]")
            ax.set_title(f"Alt={alt}m, Speed={v_kmh}km/h")
            plot_id += 1

    plt.tight_layout()
    plt.show()

#plot_energy_surface_multiple_configs(M=2, C_list=[1e9, 1.2e9], altitudes=altitudes, speeds_kmh=[30, 60, 90], include_flight_energy=True, power_flight=120)
'''

