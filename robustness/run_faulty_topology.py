#!/usr/bin/env python3
"""
robustness/run_faulty_topology.py

Simulate a link failure by setting its capacity to zero.
Runs standard RL training for 50 episodes.

This version guards against zero-capacity edges when computing state.
"""
import os, json
from topology.graph import build_sample_graph, k_shortest_paths
from sim.sim_core import Simulator
from rl.agent import CAMBEAgent

def compute_reward(sim):
    lat=[]
    for fid,meta in sim.flows.items():
        lat.append(sim.probe_latency(meta['path']))
    if not lat: return 0.0
    import math
    return -math.sqrt(sum([l*l for l in lat])/len(lat))

def _safe_link_ratio(sim, G, u, v):
    """Return load/capacity ratio but avoid division by zero."""
    cap = float(G[u][v].get('capacity', 1.0))
    if cap <= 0.0:
        # If capacity is zero, treat ratio as 1.0 (fully congested)
        return 1.0
    load = sim.get_link_load(u, v)
    return min(1.0, load / cap)

def run_fault(seed=0, fail_edge=("s1","s2"), exp="robustness_faulty", episodes=50, k_paths=3):
    base = os.path.join("results", exp, f"seed_{seed}")
    os.makedirs(os.path.join(base,"logs"), exist_ok=True)
    os.makedirs(os.path.join(base,"agent_checkpoints"), exist_ok=True)

    G = build_sample_graph()

    # Inject fault
    if G.has_edge(*fail_edge):
        G[fail_edge[0]][fail_edge[1]]['capacity'] = 0.0
        print(f"[FAULT] Edge {fail_edge} capacity set to 0")

    sim = Simulator(G)

    flows = [
        {'id':'f1','src':'h1','dst':'h2','demand':3.0},
        {'id':'f2','src':'h1','dst':'h2','demand':2.0},
        {'id':'f3','src':'h1','dst':'h2','demand':1.5}
    ]

    agent = CAMBEAgent(seed=seed)
    monitored_edges = list(G.edges())
    rewards = []

    for ep in range(1, episodes+1):
        # clear flows
        for fid in list(sim.flows.keys()):
            sim.remove_flow(fid)

        for f in flows:
            fid=f['id']; src=f['src']; dst=f['dst']; demand=f['demand']
            paths = k_shortest_paths(G, src, dst, k_paths)

            # compute Avalid
            Avalid=[]
            for idx,p in enumerate(paths):
                ok=True
                for u,v in zip(p[:-1], p[1:]):
                    cap=float(G[u][v].get('capacity',1.0))
                    load=sim.get_link_load(u,v)
                    if (cap-load) < demand-1e-9:
                        ok=False; break
                if ok:
                    Avalid.append(idx)
            if not Avalid:
                Avalid=list(range(len(paths)))

            # safe state computation: avoid division by zero on capacity
            ratios = []
            for (u,v) in monitored_edges:
                r = _safe_link_ratio(sim, G, u, v)
                # quantize into 0..5 buckets (same scale as before)
                ratios.append(int(r * 5))
            # append demand bucket (coarse)
            state = tuple(ratios + [int(demand)])

            act = agent.select_action(state, Avalid)
            idx=int(act)
            path = paths[idx]
            sim.install_flow(fid, path, demand)

        R = compute_reward(sim)
        rewards.append({'epoch':ep, 'reward':R})

        with open(os.path.join(base,"logs",f"epoch_{ep}.json"),"w") as fh:
            json.dump({"epoch":ep,"flows":flows}, fh)

        if ep % 10 == 0:
            print(f"[INFO] epoch={ep} reward={R}")

    # save rewards.csv
    with open(os.path.join(base,"logs","rewards.csv"),"w") as fh:
        fh.write("epoch,reward\n")
        for r in rewards: fh.write(f"{r['epoch']},{r['reward']}\n")

    agent.save(os.path.join(base,"agent_checkpoints","agent_final.pkl"))

    print("[DONE] Link-failure robustness experiment completed:", base)

if __name__ == "__main__":
    run_fault()
