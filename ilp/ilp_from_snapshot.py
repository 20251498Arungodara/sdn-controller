# ilp/ilp_from_snapshot.py
"""
Load an epoch snapshot produced by sim/rl_multi_flow.py (logs/epoch_<n>.json),
build an ILP to compute optimal routing, solve it, compare ILP vs RL assignments.

Usage:
    source .venv/bin/activate
    python ilp/ilp_from_snapshot.py logs/epoch_1.json
"""
import json
import os
import sys
from collections import defaultdict, deque

import pulp

# reuse simulator and topology utilities
from topology.graph import build_sample_graph
from sim.sim_core import Simulator

def load_snapshot(path):
    with open(path, 'r') as fh:
        return json.load(fh)

def build_ilp(G, flows):
    """
    G: networkx DiGraph with edges having 'capacity' and 'base_delay'
    flows: list of dicts with keys: id, src, dst, demand
    Returns: pulp.Problem, variables dict x[(fid,u,v)]
    """
    prob = pulp.LpProblem("routing_ilp", pulp.LpMinimize)
    # Create variables
    x = {}  # (fid,u,v) -> binary var
    for f in flows:
        fid = f['id']
        for u, v, d in G.edges(data=True):
            var = pulp.LpVariable(f"x_{fid}_{u}_{v}", lowBound=0, upBound=1, cat='Binary')
            x[(fid, u, v)] = var

    # Objective: minimize sum_{e} delay_e * sum_f x_{f,e}
    objective = []
    for u, v, d in G.edges(data=True):
        delay = float(d.get('base_delay', 1.0))
        objective.append(delay * pulp.lpSum([x[(f['id'], u, v)] for f in flows]))
    prob += pulp.lpSum(objective)

    # Flow conservation constraints per flow
    nodes = list(G.nodes())
    for f in flows:
        fid = f['id']
        src = f['src']
        dst = f['dst']
        for n in nodes:
            out_edges = [x[(fid,u,v)] for (u,v,_) in G.edges(data=True) if u == n]
            in_edges  = [x[(fid,u,v)] for (u,v,_) in G.edges(data=True) if v == n]
            if n == src:
                prob += (pulp.lpSum(out_edges) - pulp.lpSum(in_edges) == 1), f"flow_{fid}_cons_src_{n}"
            elif n == dst:
                prob += (pulp.lpSum(out_edges) - pulp.lpSum(in_edges) == -1), f"flow_{fid}_cons_dst_{n}"
            else:
                prob += (pulp.lpSum(out_edges) - pulp.lpSum(in_edges) == 0), f"flow_{fid}_cons_mid_{n}"

    # Capacity constraints per edge: sum_f demand_f * x_{f,e} <= capacity_e
    for u, v, d in G.edges(data=True):
        cap = float(d.get('capacity', 1.0))
        prob += (pulp.lpSum([f['demand'] * x[(f['id'], u, v)] for f in flows]) <= cap), f"cap_{u}_{v}"

    return prob, x

def extract_paths_from_x(G, flows, x_vars):
    """
    For each flow, extract a path by following edges where x_var == 1.
    This assumes ILP solution is integral and forms a simple path per flow.
    """
    paths = {}
    for f in flows:
        fid = f['id']
        src = f['src']
        dst = f['dst']
        # build adjacency selected edges for this flow
        sel_adj = defaultdict(list)
        for u, v, _ in G.edges(data=True):
            val = pulp.value(x_vars[(fid, u, v)])
            if val is not None and val > 0.5:
                sel_adj[u].append(v)
        # walk from src to dst greedily (avoid cycles)
        path = [src]
        visited = set([src])
        cur = src
        max_hops = len(G.nodes()) + 5
        hops = 0
        while cur != dst and hops < max_hops:
            nxts = [n for n in sel_adj.get(cur, []) if n not in visited]
            if not nxts:
                # if no unvisited nexts, but there are selected edges, take the first to avoid deadlock
                nxts = sel_adj.get(cur, [])
            if not nxts:
                # no selected edge forward — break
                break
            nxt = nxts[0]
            path.append(nxt)
            visited.add(nxt)
            cur = nxt
            hops += 1
        if cur != dst:
            # fallback: no valid path reconstructed
            paths[fid] = None
        else:
            paths[fid] = path
    return paths

def compute_rms_latency_for_assignments(G, assignments):
    """
    assignments: dict fid -> path (list of nodes)
    Build a simulator, install all flows with given demands (flows must be supplied)
    Here we need demands; assume flows list provided externally.
    """
    sim = Simulator(G)
    # We can't proceed without demands — caller must provide a mapping fid->demand too.
    return None

def main(snapshot_path):
    # load snapshot
    snapshot = load_snapshot(snapshot_path)
    # build graph (must be consistent with snapshot generation)
    G = build_sample_graph()

    # Build flows list (with demands) from snapshot: snapshot['flows'] contains id,demand,chosen
    flows = []
    for f in snapshot.get('flows', []):
        # snapshot entry: {'id':..., 'demand':..., 'chosen':...}
        flows.append({'id': f['id'], 'src': 'h1', 'dst': 'h2', 'demand': float(f['demand'])})

    if not flows:
        print("No flows found in snapshot.")
        return

    # Build ILP
    prob, xvars = build_ilp(G, flows)
    print("Solving ILP (this may take a moment)...")
    # Solve (CBC default)
    solver = pulp.PULP_CBC_CMD(msg=False)  # set msg=True for solver logs
    res = prob.solve(solver)
    status = pulp.LpStatus.get(prob.status, 'Unknown')
    print("ILP status:", pulp.LpStatus[prob.status])

    # Extract ILP paths
    ilp_paths = extract_paths_from_x(G, flows, xvars)

    # compute RL latency: use simulator and the RL assignments from snapshot
    sim_rl = Simulator(G)
    for f in snapshot.get('flows', []):
        fid = f['id']
        chosen_idx = int(f['chosen'])
        candidates = snapshot.get('candidates', {}).get(fid, [])
        if not candidates:
            raise RuntimeError("No candidates stored in snapshot")
        # chosen path
        chosen_path = candidates[chosen_idx]
        sim_rl.install_flow(fid, chosen_path, float(f['demand']))
    # compute RL RMS latency
    rl_latencies = []
    for fid, meta in sim_rl.flows.items():
        rl_latencies.append(sim_rl.probe_latency(meta['path']))
    rl_rms = (sum([l*l for l in rl_latencies]) / len(rl_latencies)) ** 0.5 if rl_latencies else 0.0

    # compute ILP RMS latency
    sim_ilp = Simulator(G)
    for f in flows:
        fid = f['id']
        p = ilp_paths.get(fid)
        if p is None:
            # if we cannot reconstruct path, skip (or fall back to snapshot assignment)
            p = snapshot.get('assignments', {}).get(fid)
            if p is None:
                print(f"Warning: cannot determine ILP path for {fid}, skipping")
                continue
        sim_ilp.install_flow(fid, p, float(f['demand']))
    ilp_latencies = []
    for fid, meta in sim_ilp.flows.items():
        ilp_latencies.append(sim_ilp.probe_latency(meta['path']))
    ilp_rms = (sum([l*l for l in ilp_latencies]) / len(ilp_latencies)) ** 0.5 if ilp_latencies else 0.0

    # objective value from ILP
    ilp_obj = pulp.value(prob.objective)

    # compute optimality gap: (RL - ILP) / ILP  (latencies are positive, but rewards were negative RMS)
    gap = None
    if ilp_rms != 0:
        gap = (rl_rms - ilp_rms) / ilp_rms

    # Save results
    out = {
        'snapshot': os.path.basename(snapshot_path),
        'ilp_status': pulp.LpStatus[prob.status],
        'ilp_objective': ilp_obj,
        'rl_rms_latency': rl_rms,
        'ilp_rms_latency': ilp_rms,
        'optimality_gap': gap,
        'ilp_paths': ilp_paths,
    }
    out_path = os.path.join("logs", f"ilp_result_{os.path.basename(snapshot_path)}")
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=2)

    # Print summary
    print("=== ILP Result Summary ===")
    print("Snapshot:", snapshot_path)
    print("ILP status:", out['ilp_status'])
    print(f"ILP objective (sum delay * flow-edge): {ilp_obj}")
    print(f"RL RMS latency: {rl_rms:.3f} ms")
    print(f"ILP RMS latency: {ilp_rms:.3f} ms")
    if gap is not None:
        print(f"Optimality gap (RL vs ILP): {gap*100:.2f}%  (positive => RL worse)")
    else:
        print("Optimality gap: NA")
    print("ILP per-flow paths:")
    for fid, p in ilp_paths.items():
        print(f"  {fid}: {p}")
    print("Saved detailed result to", out_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ilp/ilp_from_snapshot.py logs/epoch_1.json")
        sys.exit(1)
    main(sys.argv[1])
