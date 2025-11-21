#!/usr/bin/env python3
"""
ablation/run_ablation.py

Run ablation experiments:
  python ablation/run_ablation.py --variant no_ca --exp ablation_no_ca --loads 0.3 0.7 1.1 --seeds 0 1 2 --episodes 100 --k 3

variant: no_ca | no_mbe

Outputs (per run):
  results/<exp>/load_<lf>/seed_<s>/logs/epoch_*.json
  results/<exp>/load_<lf>/seed_<s>/logs/rewards.csv
  results/<exp>/load_<lf>/seed_<s>/agent_checkpoints/
  results/<exp>/load_<lf>/seed_<s>/meta.json

Notes:
- This script intentionally keeps things simple and sequential so it's easy to debug.
- It uses the provided rl/agent_no_ca.py and rl/agent_no_mbe.py variants when requested.
- Provenance: the uploaded paper path is stored in meta.json.
"""
import argparse
import os
import time
import json
import random
from collections import defaultdict

# Project imports (assumes you run from project root)
from topology.graph import build_sample_graph, k_shortest_paths
from sim.sim_core import Simulator

# provenance (uploaded pdf path)
PAPER_PDF_PATH = "/mnt/data/Trustworthy_and_Congestion_Aware_Multipath_Routing_in_SDN__A_Hybrid_Approach_using_Max_Boltzmann_Q_Learning__ILP_Verification__and_Blockchain_Auditing.pdf"

def load_agent_class(variant):
    if variant == "no_ca":
        from rl.agent_no_ca import CAMBEAgent as Agent
    elif variant == "no_mbe":
        from rl.agent_no_mbe import CAMBEAgent as Agent
    else:
        raise ValueError("Unknown variant: " + str(variant))
    return Agent

def default_flows():
    return [
        {'id':'f1','src':'h1','dst':'h2','demand':3.0},
        {'id':'f2','src':'h1','dst':'h2','demand':2.0},
        {'id':'f3','src':'h1','dst':'h2','demand':1.5},
    ]

def compute_reward(sim):
    """Negative RMS latency across active flows."""
    latencies = []
    for fid, meta in sim.flows.items():
        p = meta.get('path')
        if p is None:
            continue
        latencies.append(sim.probe_latency(p))
    if not latencies:
        return 0.0
    import math
    sq = sum([l*l for l in latencies]) / len(latencies)
    return -math.sqrt(sq)

def ensure_dirs(base_path):
    os.makedirs(os.path.join(base_path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "agent_checkpoints"), exist_ok=True)

def save_meta(base_path, config, seed):
    meta = {
        "config": config,
        "seed": seed,
        "paper_pdf": PAPER_PDF_PATH,
        "timestamp": time.asctime(),
        "cwd": os.getcwd()
    }
    with open(os.path.join(base_path, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

def run_one(exp_name, variant, load_factor, seed, episodes, k_paths):
    Agent = load_agent_class(variant)

    base = os.path.join("results", exp_name, f"load_{float(load_factor):.2f}", f"seed_{seed}")
    ensure_dirs(base)
    save_meta(base, {'variant': variant, 'load_factor': load_factor, 'episodes': episodes, 'k_paths': k_paths}, seed)

    print(f"[RUN] {exp_name} | variant={variant} | load={load_factor} | seed={seed} | episodes={episodes}")

    G = build_sample_graph()
    sim = Simulator(G)

    flows = default_flows()
    # scale flows by load_factor
    flows = [{**f, 'demand': float(f['demand']) * float(load_factor)} for f in flows]

    agent = Agent(seed=seed)
    monitored_edges = list(G.edges())

    rewards_log = []

    for ep in range(1, episodes + 1):
        # clear existing flows
        for fid in list(sim.flows.keys()):
            sim.remove_flow(fid)

        epoch_snapshot = {
            'epoch': ep,
            'flows': [],
            'candidates': {},
            'Avalid': {},
            'assignments': {},
            'link_loads': {}
        }

        for f in flows:
            fid = f['id']; src = f['src']; dst = f['dst']; demand = float(f['demand'])
            # NOTE: use positional k_paths argument
            paths = k_shortest_paths(G, src, dst, k_paths)

            epoch_snapshot['candidates'][fid] = paths

            # build Avalid (same logic as main trainer)
            Avalid = []
            for idx, p in enumerate(paths):
                ok = True
                for u, v in zip(p[:-1], p[1:]):
                    cap = float(G[u][v].get('capacity', 1.0))
                    load = sim.get_link_load(u, v)
                    if (cap - load) < (demand - 1e-9):
                        ok = False
                        break
                if ok:
                    Avalid.append(idx)

            fallback = False
            if not Avalid:
                Avalid = list(range(len(paths)))
                fallback = True

            # quantized state (simple)
            state = tuple([int(min(1.0, sim.get_link_load(u, v) / float(G[u][v].get('capacity', 1.0))) * (6 - 1)) for (u, v) in monitored_edges] + [int(demand)])

            # agent chooses action; agent may ignore Avalid if it's the no_ca variant
            action = agent.select_action(state, Avalid)
            try:
                chosen_idx = int(action)
            except Exception:
                # fallback to first valid
                chosen_idx = Avalid[0] if Avalid else 0

            chosen_path = paths[chosen_idx]
            sim.install_flow(fid, chosen_path, demand)

            reward = compute_reward(sim)
            next_state = state  # simple; we do not simulate intermediate transitions here

            # update agent
            try:
                agent.update(state, chosen_idx, reward, next_state, Avalid)
            except Exception as e:
                # keep going but log
                print("[WARN] agent.update failed:", e)

            # record
            epoch_snapshot['flows'].append({'id': fid, 'demand': demand, 'chosen': chosen_idx, 'fallback': fallback})
            epoch_snapshot['Avalid'][fid] = Avalid
            epoch_snapshot['assignments'][fid] = chosen_path

        # record link loads
        for u, v, d in G.edges(data=True):
            epoch_snapshot['link_loads'][f"{u}->{v}"] = sim.get_link_load(u, v)

        # compute epoch reward
        epoch_reward = compute_reward(sim)
        rewards_log.append({'epoch': ep, 'reward': epoch_reward})

        # write snapshot (minimal)
        snap_path = os.path.join(base, "logs", f"epoch_{ep}.json")
        with open(snap_path, 'w') as fh:
            json.dump(epoch_snapshot, fh, indent=2)

        # print progress occasionally
        if ep % max(1, episodes // 10) == 0 or ep == episodes:
            print(f"[INFO] ep={ep} reward={epoch_reward:.4f} saved {snap_path}")
            # checkpoint
            ckpt_path = os.path.join(base, "agent_checkpoints", f"agent_ep_{ep}.pkl")
            try:
                agent.save(ckpt_path)
            except Exception as e:
                print("[WARN] agent.save failed:", e)

    # write rewards.csv
    rewards_csv = os.path.join(base, "logs", "rewards.csv")
    with open(rewards_csv, 'w') as fh:
        fh.write("epoch,reward\n")
        for r in rewards_log:
            fh.write(f"{r['epoch']},{r['reward']}\n")

    # final checkpoint
    try:
        agent.save(os.path.join(base, "agent_checkpoints", "agent_final.pkl"))
    except Exception as e:
        print("[WARN] final agent.save failed:", e)

    print("[DONE] run finished:", base)
    return base

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['no_ca', 'no_mbe'], required=True)
    p.add_argument('--exp', default='ablation_run')
    p.add_argument('--loads', nargs='+', type=float, default=[0.3, 0.7, 1.1])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--k', type=int, default=3)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    for lf in args.loads:
        for s in args.seeds:
            run_one(args.exp, args.variant, lf, s, args.episodes, args.k)
