#!/usr/bin/env python3
"""
robustness/run_overload.py

Stress test: traffic demands exceed link capacities.
Agent must degrade gracefully.
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

def run_overload(seed=0, exp="robustness_overload", episodes=50, k_paths=3):
    base = os.path.join("results", exp, f"seed_{seed}")
    os.makedirs(os.path.join(base,"logs"), exist_ok=True)
    os.makedirs(os.path.join(base,"agent_checkpoints"), exist_ok=True)

    G = build_sample_graph()
    sim = Simulator(G)

    flows = [
        {'id':'f1','src':'h1','dst':'h2','demand':10.0},
        {'id':'f2','src':'h1','dst':'h2','demand':8.0},
        {'id':'f3','src':'h1','dst':'h2','demand':6.0}
    ]

    agent = CAMBEAgent(seed=seed)
    monitored_edges = list(G.edges())
    rewards = []

    for ep in range(1, episodes+1):
        for fid in list(sim.flows.keys()):
            sim.remove_flow(fid)

        for f in flows:
            fid=f['id']; src=f['src']; dst=f['dst']; demand=f['demand']
            paths = k_shortest_paths(G, src, dst, k_paths)

            Avalid=[]
            for idx,p in enumerate(paths):
                ok=True
                for u,v in zip(p[:-1], p[1:]):
                    cap=float(G[u][v].get('capacity',1.0))
                    load=sim.get_link_load(u,v)
                    if (cap-load) < demand-1e-9:
                        ok=False; break
                if ok: Avalid.append(idx)
            if not Avalid:
                Avalid=list(range(len(paths)))

            state = tuple([int(min(1.0, sim.get_link_load(u,v)/float(G[u][v].get('capacity',1.0))) * 5) for (u,v) in monitored_edges] + [int(demand)])

            act = agent.select_action(state, Avalid)
            idx=int(act)
            sim.install_flow(fid, paths[idx], demand)

        R = compute_reward(sim)
        rewards.append({'epoch':ep, 'reward':R})

        if ep % 10 == 0:
            print(f"[INFO] epoch={ep} reward={R}")

        with open(os.path.join(base,"logs",f"epoch_{ep}.json"),"w") as fh:
            json.dump({"epoch":ep}, fh)

    with open(os.path.join(base,"logs","rewards.csv"),"w") as fh:
        fh.write("epoch,reward\n")
        for r in rewards:
            fh.write(f"{r['epoch']},{r['reward']}\n")

    agent.save(os.path.join(base,"agent_checkpoints","agent_final.pkl"))
    print("[DONE] Overload robustness experiment completed:", base)

if __name__ == "__main__":
    run_overload()
