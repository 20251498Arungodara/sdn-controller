# sim/rl_multi_flow.py
"""
Multi-flow CA-MBE RL simulation.

- Builds sample graph.
- Accepts a pre-defined set of flows (id, src, dst, demand).
- For each epoch:
    - clears previous flow installs
    - for each flow (in order), computes candidate paths, builds Avalid
    - agent selects action (from Avalid)
    - installs chosen path
    - computes reward (negative RMS latency across active flows)
    - updates Q-table for that flow decision
- Saves an epoch JSON snapshot (flows, assignments, link loads, Avalids)
- Saves a Q-table checkpoint at the end.
"""

import json
import math
import os
import time
from collections import defaultdict

from sim.sim_core import Simulator
from topology.graph import build_sample_graph, k_shortest_paths
from rl.agent import MBEQAgent

LOGDIR = "logs"
os.makedirs(LOGDIR, exist_ok=True)

def compute_reward(sim: Simulator):
    """Negative RMS latency across active flows (same as Step 3)."""
    latencies = []
    for fid, meta in sim.flows.items():
        p = meta['path']
        lat = sim.probe_latency(p)
        latencies.append(lat)
    if not latencies:
        return 0.0
    sq = sum([l*l for l in latencies]) / len(latencies)
    return -math.sqrt(sq)

def quantize_state(sim: Simulator, edges=None, buckets=4):
    """Quantize link-load ratios into small tuple state (like Step 3)."""
    if edges is None:
        edges = list(sim.G.edges())
    state = []
    for u,v in edges:
        cap = float(sim.G[u][v].get('capacity', 1.0))
        load = sim.get_link_load(u,v)
        ratio = 0.0 if cap <= 0 else min(1.0, load/cap)
        b = int(ratio*(buckets-1) + 1e-9)
        state.append(b)
    return tuple(state)

def run_multi_flow(episodes=5, k_paths=3, flows=None):
    """
    episodes: number of epochs to simulate (each epoch reassigns all flows)
    flows: list of dicts: {'id','src','dst','demand'}
    """
    if flows is None:
        flows = [
            {'id':'f1','src':'h1','dst':'h2','demand':3.0},
            {'id':'f2','src':'h1','dst':'h2','demand':2.0},
            {'id':'f3','src':'h1','dst':'h2','demand':1.5},
        ]

    G = build_sample_graph()
    sim = Simulator(G)
    agent = MBEQAgent(alpha=0.6, gamma=0.9, eps=0.15, tau=0.8)

    monitored_edges = list(G.edges())

    for ep in range(1, episodes+1):
        # remove all flows at start of epoch
        for fid in list(sim.flows.keys()):
            sim.remove_flow(fid)

        epoch_snapshot = {
            'epoch': ep,
            'flows': [],
            'link_loads': {},
            'assignments': {},
            'candidates': {},
            'Avalid': {},
        }

        # Decide paths sequentially for each flow
        for f in flows:
            fid = f['id']
            src, dst = f['src'], f['dst']
            demand = float(f['demand'])

            # Candidate paths
            paths = k_shortest_paths(G, src, dst, K=k_paths)
            epoch_snapshot['candidates'][fid] = paths

            # Build Avalid (congestion-aware)
            Avalid = []
            for idx, p in enumerate(paths):
                ok = True
                for u,v in zip(p[:-1], p[1:]):
                    cap = float(G[u][v].get('capacity', 1.0))
                    load = sim.get_link_load(u, v)
                    if (cap - load) < demand - 1e-9:
                        ok = False
                        break
                if ok:
                    Avalid.append(idx)
            if not Avalid:
                # fallback to full set but mark fallback
                Avalid = list(range(len(paths)))
                fallback = True
            else:
                fallback = False

            # Build state and choose action
            state = quantize_state(sim, edges=monitored_edges, buckets=6)
            # agent returns a chosen action from Avalid (it expects actions list)
            chosen_action = agent.select_action(state, Avalid)
            # guard if chosen_action not integer index (agent returns element from Avalid)
            if chosen_action not in range(len(paths)):
                # if agent returns action object equal to index, coerce
                try:
                    chosen_idx = int(chosen_action)
                except Exception:
                    chosen_idx = Avalid[0] if Avalid else 0
            else:
                chosen_idx = chosen_action

            chosen_path = paths[chosen_idx] if paths else None

            # Install
            if chosen_path is None:
                raise RuntimeError(f"No path chosen for flow {fid}")
            sim.install_flow(fid, chosen_path, demand)

            # compute reward (after install)
            reward = compute_reward(sim)
            next_state = quantize_state(sim, edges=monitored_edges, buckets=6)
            next_actions = Avalid.copy()

            # Update Q-table
            agent.update(state, chosen_idx, reward, next_state, next_actions)

            # record snapshot info
            epoch_snapshot['flows'].append({'id': fid, 'demand': demand, 'chosen': chosen_idx, 'fallback': fallback})
            epoch_snapshot['Avalid'][fid] = Avalid
            epoch_snapshot['assignments'][fid] = chosen_path

        # After assigning all flows, record link loads and save snapshot
        for u,v,d in G.edges(data=True):
            epoch_snapshot['link_loads'][f"{u}->{v}"] = sim.get_link_load(u,v)

        # Save snapshot to logs
        fname = os.path.join(LOGDIR, f"epoch_{ep}.json")
        with open(fname, 'w') as fh:
            json.dump(epoch_snapshot, fh, indent=2)
        print(f"[Epoch {ep}] saved snapshot -> {fname}")

        # print summary metrics
        print(f"Epoch {ep} summary:")
        for fid in epoch_snapshot['flows']:
            print(f"  Flow {fid['id']} -> path_idx {fid['chosen']} (fallback={fid['fallback']})")
        print(" Link loads:")
        for k,v in epoch_snapshot['link_loads'].items():
            print(f"   {k} : {v:.2f} Mbps")
        # quick reward print
        total_reward = compute_reward(sim)
        print(f"  epoch reward (neg RMS latency) = {total_reward:.3f}")
        print("-"*60)
        time.sleep(0.1)

    # Save Q-table
    try:
        import pickle
        qdict = dict(agent.Q)
        with open(os.path.join(LOGDIR, "agent_Q_final.pkl"), 'wb') as f:
            pickle.dump(qdict, f)
        print("Saved agent Q to logs/agent_Q_final.pkl")
    except Exception as e:
        print("Warning: could not save Q:", e)

    print("Multi-flow demo complete.")

if __name__ == "__main__":
    # default: run 6 epochs
    run_multi_flow(episodes=6, k_paths=3)
