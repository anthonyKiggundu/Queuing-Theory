"""Run the baseline simulation for each information model and export the
quantities needed by the cost-comparison analysis.

Place next to stabanaly_ver_ren_jock_latest_mfg_integrated.py and run:
    python export_trained_params.py [--duration 10 --episodes 15 --seeds 2 --out trained_params]

Outputs (in --out):
    actor_critic_weights.pt   trained A2C state_dict (reload with agent.model.load_state_dict)
    actor_critic_weights.npz  same weights as numpy arrays (framework-free)
    trained_params.json       eta, K_abandon, K_switch, rho, Psi_cloud, mu_bar, lam_bar,
                              wait_proxy, calibrated K per information model, beta_hat
"""
import argparse
import importlib
import json
import os

import numpy as np
import torch

MODULE = "stabanaly_ver_ren_jock_latest_mfg_integrated"


def to_jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="trained_params")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sim = importlib.import_module(MODULE)

    state_dim, action_dim = 11, 2
    result = {"psi_cloud": "1.5*w + 0.5 (cloud_outside_option)", "models": {}}

    for uses_nn in (False, True):
        label = "Actor-Critic" if uses_nn else "Markov"
        actor_critic = sim.ActorCritic(state_dim, action_dim)
        agent = sim.A2CAgent(state_dim, action_dim)
        rq = sim.RequestQueue(state_dim, action_dim, 1.0, 0.1, actor_critic, agent,
                              uses_nn=uses_nn)
        env = sim.ImpatientTenantEnv()
        env.requestObj = rq

        rq.run(args.duration, env, adjust_service_rate=False,
               num_episodes=args.episodes, progress_bar=False, progress_log=False,
               save_to_file=os.path.join(args.out, f"baseline_{label}.csv"),
               num_seeds=args.seeds)

        base = rq.export_baseline_mfg_state()
        total_t = max(len(rq.global_queue_length_history) * rq.time_res, 1e-6)
        calib = rq.calibrate_mfg_from_subscribers(
            mfg_time_scale=max(8.0 / total_t, 1.0), use_hazard=True, per_server=False)
        params = sim.FullDensityMFG(uses_nn=uses_nn).build_params(base, calib)

        entry = {
            "eta": params.eta, "K_abandon": params.K_abandon,
            "K_switch": params.K_switch, "rho": params.rho,
            "mu_bar": base["mu_bar"], "lam_bar": base["lam_bar"],
            "lam_per_queue": params.lam, "wait_proxy": base["wait_proxy"],
            "m0_mean": params.m0_mean,
            "psi_cloud_at_m0_mean": sim.cloud_outside_option(np.asarray(params.m0_mean)),
            "calibration": calib,
        }

        if uses_nn:
            try:
                entry["beta"] = rq.estimate_empirical_beta()["beta_hat"]
            except Exception as exc:
                entry["beta_error"] = repr(exc)
            sd = agent.model.state_dict()
            torch.save(sd, os.path.join(args.out, "actor_critic_weights.pt"))
            np.savez(os.path.join(args.out, "actor_critic_weights.npz"),
                     **{k: v.detach().cpu().numpy() for k, v in sd.items()})
            entry["weights_shapes"] = {k: list(v.shape) for k, v in sd.items()}

        result["models"][label] = entry
        print(f"[{label}] mu_bar={base['mu_bar']} lam_bar={base['lam_bar']:.4g} "
              f"K_ab={params.K_abandon:.4g} K_sw={params.K_switch:.4g}")

    path = os.path.join(args.out, "trained_params.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=to_jsonable)
    print("Saved to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
