# sim/rl_loop.py
"""
Simple RL loop integrating MBEQAgent with the Simulator.
- chooses a path for an incoming flow using congestion-aware action masking (Avalid)
- installs chosen path, computes reward (negative RMS latency), and updates Q
This is a minimal, illustrative single-flow RL episode loop (repeatable).
"""

from sim.sim_core import Simulator
from topology.graph import build_sample_graph, k_shortest_paths
from rl.agent import MBEQAgent
import math
import time

# --- helper: quantize state into a small tuple for tabular Q ---
def quantize_state(sim: Simulator, monitored_edges=None, buckets=4, demand_mbps=1.0):
    """
    Return a small state tuple derived from link load ratios.
    monitored_edges: list of edges (u,v) to include; by default include all edges.
    buckets: number of bins for load ratio [0..1].
    demand_mbps: used to add a simple demand bucket feature.
    """
    if monitored_edges is None:
        monitored_edges = list(sim.G.edges())
    ratios = []
    for u, v in monitored_edges:
        cap = float(sim.G[u][v].get('capacity', 1.0))
        load = sim.get_link_load(u, v)
        ratio = 0.0 if cap <= 0 else min(1.0, load / cap)
        # bucketize: 0..buckets-1
        b = int(ratio * (buckets - 1) + 0.000001)
        ratios.append(b)
    # include demand bucket (coarse)
    d_bucket = int(min(4, max(0, demand_mbps // 2)))
    return tuple(ratios + [d_bucket])

# --- reward: negative RMS latency across active flows ---
def compute_reward(sim: Simulator):
    # compute per-flow latencies then negative RMS
    latencies = []
    for fid, meta in sim.flows.items():
        p = meta['path']
        lat = sim.probe_latency(p)
        latencies.append(lat)
    if not latencies:
        return 0.0
    sq = sum([l*l for l in latencies]) / len(latencies)
    rms = math.sqrt(sq)
    return -rms

def run_episodes(num_episodes=10, k_paths=3):
    G = build_sample_graph()
    sim = Simulator(G)
    agent = MBEQAgent(alpha=0.6, gamma=0.9, eps=0.2, tau=0.8)

    # We'll repeatedly request routing for the same flow (src,dst,demand)
    flow_id = "flowA"
    src, dst = 'h1', 'h2'
    demand = 3.0  # Mbps

    monitored_edges = list(G.edges())

    print("=== RL loop demo: CA-MBE selection + Q-update ===")
    for ep in range(1, num_episodes + 1):
        # clear flow if present
        if flow_id in sim.flows:
            sim.remove_flow(flow_id)

        # gather candidate paths
        paths = k_shortest_paths(G, src, dst, K=k_paths)
        # convert paths to action labels (we'll use indices as actions)
        actions = list(range(len(paths)))  # e.g., [0,1,2]
        # compute Avalid: paths where every edge has spare capacity >= demand
        Avalid = []
        for idx, p in enumerate(paths):
            ok = True
            for u, v in zip(p[:-1], p[1:]):
                cap = float(G[u][v].get('capacity', 1.0))
                load = sim.get_link_load(u, v)
                if (cap - load) < demand - 1e-9:
                    ok = False
                    break
            if ok:
                Avalid.append(idx)

        # fallback: if no valid actions, allow full action set but log
        if not Avalid:
            Avalid = actions.copy()
            flagged = True
        else:
            flagged = False

        # build a compact state
        state = quantize_state(sim, monitored_edges=monitored_edges, demand_mbps=demand)

        # map action indices to action labels expected by agent
        # here agent uses any hashable action, we'll pass indices
        chosen_idx = agent.select_action(state, Avalid)
        chosen_path = paths[chosen_idx] if (0 <= chosen_idx < len(paths)) else None

        # apply action: install flow on chosen_path
        sim.install_flow(flow_id, chosen_path, demand)

        # compute reward and update Q
        reward = compute_reward(sim)
        # prepare next state (after installing)
        next_state = quantize_state(sim, monitored_edges=monitored_edges, demand_mbps=demand)
        # available next actions (for q_next computation) same as current Avalid for simplicity
        next_actions = Avalid.copy()

        # perform Q update
        agent.update(state, chosen_idx, reward, next_state, next_actions)

        # print episode summary
        qval = agent.Q[(tuple(state), chosen_idx)]
        print(f"[Ep {ep}] chosen={chosen_idx}, flagged_fallback={flagged}, reward={reward:.3f}, q={qval:.4f}")
        print("  Path chosen:", chosen_path)
        # show link loads briefly
        sim.show_link_stats()
        print("-" * 60)
        # small delay for readability
        time.sleep(0.1)

    print("Demo complete. Example Q-entries (non-zero):")
    for (s,a), v in list(agent.Q.items())[:20]:
        if abs(v) > 1e-9:
            print(" state", s, " action", a, " Q", v)
    # save agent Q to a small file for later (optional)
    try:
        import pickle
        with open('logs/agent_Q.pkl', 'wb') as f:
            pickle.dump(dict(agent.Q), f)
        print("Saved Q snapshot to logs/agent_Q.pkl")
    except Exception:
        pass

if __name__ == "__main__":
    run_episodes(num_episodes=8, k_paths=3)
