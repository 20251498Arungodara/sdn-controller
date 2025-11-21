#!/usr/bin/env python3
"""
train/trainer.py

Trainer with audit hooks:
 - writes epoch snapshots
 - emits audit events after epoch snapshot and after ILP verification (if ILP run)
 - compatible with run_experiment.py (exposes run_training)

Usage (CLI):
  source .venv/bin/activate
  python train/trainer.py --exp-name audit_test --seed 0 --episodes 5 --k 3 --load 1.00 --ilp-interval 0

Public function:
  run_training(exp_name, load_factor, seed, episodes, k_paths, ilp_interval, base_results_dir="results")
"""
import os
import json
import time
import math
import argparse
import subprocess
from pathlib import Path

# project imports (assumes project root is in PYTHONPATH)
from topology.graph import build_sample_graph, k_shortest_paths
from sim.sim_core import Simulator
try:
    from rl.agent import CAMBEAgent
except Exception:
    # Fallback: if rl/agent.py defines CAMBEAgent differently, the import will fail here.
    raise

# audit emitter (utils/audit_emitter.py)
from utils.audit_emitter import emit_audit_event, sha256_file, sha256_str

# Provenance: uploaded paper path
PAPER_PDF_PATH = "/mnt/data/Trustworthy_and_Congestion_Aware_Multipath_Routing_in_SDN__A_Hybrid_Approach_using_Max_Boltzmann_Q_Learning__ILP_Verification__and_Blockchain_Auditing.pdf"

def compute_rms_latency_from_sim(sim):
    """Compute RMS latency across active flows in the simulator."""
    lat = []
    for fid, meta in sim.flows.items():
        p = meta.get("path")
        if not p:
            continue
        try:
            l = sim.probe_latency(p)
        except Exception:
            l = float("inf")
        lat.append(l)
    if not lat:
        return 0.0
    s = sum([x*x for x in lat]) / len(lat)
    if math.isinf(s) or math.isnan(s):
        return float("-inf") if s < 0 else float("inf")
    return math.sqrt(s)

def compute_optimality_gap(rl_rms, ilp_json):
    """Return (RL_RMS - ILP_RMS)/ILP_RMS if ilp_rms exists, else None."""
    try:
        ilp_rms = ilp_json.get("ilp_rms_latency") or ilp_json.get("ilp_rms") or ilp_json.get("ilp_total_latency")
        if ilp_rms is None:
            return None
        ilp_rms = float(ilp_rms)
        if ilp_rms == 0:
            return None
        return (rl_rms - ilp_rms) / ilp_rms
    except Exception:
        return None

def try_run_ilp(snapshot_path, out_dir):
    """
    Try to run the ILP solver on a snapshot. This function attempts:
      1) If ilp/ilp_from_snapshot.py exists and is executable, call it with snapshot_path.
      2) If there is a Python API function ilp.ilp_from_snapshot.run(snapshot_path, out_path), we try subprocess as fallback.
    Returns: path to generated ilp_result JSON or None.
    """
    # ensure out_dir exists
    os.makedirs(out_dir, exist_ok=True)
    # 1) try via script call (most robust)
    script = os.path.join("ilp", "ilp_from_snapshot.py")
    if os.path.exists(script):
        try:
            # call script, let it write ilp_result_*.json in cwd
            proc = subprocess.run(["python", script, snapshot_path], check=True, capture_output=True, text=True)
            # try to detect generated JSON in cwd
            # common pattern: ilp_result_epoch_XXX.json
            # search for recent files in out_dir
        except subprocess.CalledProcessError as e:
            print("[ILP] subprocess failed:", e, e.stdout, e.stderr)
    # 2) attempt to find any ilp_result_*.json in current working directory or out_dir
    candidates = list(Path(".").glob("ilp_result_*.json")) + list(Path(out_dir).glob("ilp_result_*.json"))
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    # move the best candidate into out_dir (if not already there)
    chosen = candidates[0]
    dest = Path(out_dir) / chosen.name
    try:
        if chosen.resolve() != dest.resolve():
            chosen.rename(dest)
    except Exception:
        # fallback to copy
        import shutil
        shutil.copy2(chosen, dest)
    return str(dest)

def run_training(exp_name: str,
                 load_factor: float = 1.0,
                 seed: int = 0,
                 episodes: int = 100,
                 k_paths: int = 3,
                 ilp_interval: int = 0,
                 base_results_dir: str = "results"):
    """
    Core training loop.

    - exp_name: name of experiment (subfolder under results/)
    - load_factor: scaling for flow demands
    - seed: random seed
    - episodes: number of episodes
    - k_paths: candidate path count
    - ilp_interval: run ILP every N epochs if > 0 (offline ILP recommended)
    """
    # prepare result path
    load_label = f"load_{float(load_factor):.2f}"
    base_path = os.path.join(base_results_dir, exp_name, load_label, f"seed_{seed}")
    logs_dir = os.path.join(base_path, "logs")
    ckpt_dir = os.path.join(base_path, "agent_checkpoints")
    ilp_dir = os.path.join(base_path, "ilp_results")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(ilp_dir, exist_ok=True)

    # save meta
    meta = {
        "exp_name": exp_name,
        "load_factor": load_factor,
        "seed": seed,
        "episodes": episodes,
        "k_paths": k_paths,
        "ilp_interval": ilp_interval,
        "paper_pdf": PAPER_PDF_PATH,
        "timestamp": time.asctime()
    }
    with open(os.path.join(base_path, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # build topology & simulator
    G = build_sample_graph()
    sim = Simulator(G)

    # sample flows used by the paper (you can adapt if your project uses a flows file)
    flows = [
        {'id':'f1','src':'h1','dst':'h2','demand':1.5},
        {'id':'f2','src':'h1','dst':'h2','demand':1.0},
        {'id':'f3','src':'h1','dst':'h2','demand':0.75},
    ]
    # scale flows
    for f in flows:
        f['demand'] = float(f['demand']) * float(load_factor)

    # instantiate agent
    agent = CAMBEAgent(seed=seed)

    monitored_edges = list(G.edges())

    rewards_log = []

    print("====================================")
    print(" STARTING TRAINING SESSION")
    print(" Experiment :", exp_name)
    print(" Seed       :", seed)
    print(" Episodes   :", episodes)
    print(" K-paths    :", k_paths)
    print(" Load       :", load_factor)
    print(" ILP every  :", ilp_interval, "epochs" if ilp_interval>0 else "disabled")
    print("====================================")

    for ep in range(1, episodes+1):
        # remove any existing flows in simulator
        for fid in list(sim.flows.keys()):
            sim.remove_flow(fid)

        epoch_assignments = {}
        epoch_candidates = {}
        epoch_avails = {}
        # for each flow, compute k-shortest, Avalid, select action, install flow
        for f in flows:
            fid = f['id']; src = f['src']; dst = f['dst']; demand = float(f['demand'])
            paths = k_shortest_paths(G, src, dst, k_paths)  # positional arg
            epoch_candidates[fid] = paths

            # compute Avalid mask (feasible paths under residual capacity)
            Avalid = []
            for idx, p in enumerate(paths):
                feasible = True
                for u, v in zip(p[:-1], p[1:]):
                    cap = float(G[u][v].get('capacity', 1.0))
                    load = sim.get_link_load(u, v)
                    if (cap - load) < (demand - 1e-9):
                        feasible = False
                        break
                if feasible:
                    Avalid.append(idx)
            fallback = False
            if not Avalid:
                Avalid = list(range(len(paths)))
                fallback = True
            epoch_avails[fid] = {'Avalid': Avalid, 'fallback': fallback}

            # build quantized state (simple vector)
            ratios = []
            for (u, v) in monitored_edges:
                cap = float(G[u][v].get('capacity', 1.0))
                if cap <= 0.0:
                    ratio = 1.0
                else:
                    ratio = min(1.0, sim.get_link_load(u, v) / cap)
                ratios.append(int(ratio * (6 - 1)))  # 0..5
            state = tuple(ratios + [int(demand)])

            # agent selects an action (agent may ignore Avalid depending on variant)
            action = agent.select_action(state, Avalid)
            try:
                chosen_idx = int(action)
            except Exception:
                chosen_idx = int(Avalid[0]) if Avalid else 0

            # install flow
            chosen_path = paths[chosen_idx]
            sim.install_flow(fid, chosen_path, demand)
            epoch_assignments[fid] = chosen_idx

            # compute reward and update agent after each flow (simplified)
            r = -compute_rms_latency_from_sim(sim) if compute_rms_latency_from_sim(sim) not in [float("inf"), float("-inf")] else -1e6
            agent.update(state, chosen_idx, r, state, Avalid)

        # end of epoch: compute epoch reward from simulator
        epoch_rms = compute_rms_latency_from_sim(sim)
        # store negative RMS as reward (consistent with earlier code)
        epoch_reward = -epoch_rms if (not math.isinf(epoch_rms) and not math.isnan(epoch_rms)) else float("-inf")
        rewards_log.append({'epoch': ep, 'reward': epoch_reward})

        # snapshot for this epoch
        snapshot = {
            'epoch': ep,
            'flows': flows,
            'candidates': epoch_candidates,
            'assignments': epoch_assignments,
            'Avalid': epoch_avails,
            'link_loads': {f"{u}->{v}": sim.get_link_load(u, v) for (u, v, d) in G.edges(data=True)},
            'epoch_reward': epoch_reward
        }
        snapshot_path = os.path.join(logs_dir, f"epoch_{ep}.json")
        with open(snapshot_path, 'w') as fh:
            json.dump(snapshot, fh, indent=2)

        # --- AUDIT: emit epoch commit event ---
        try:
            snapshot_hash = sha256_file(snapshot_path)
            # Q-table hash
            try:
                qdict = {str(k): float(v) for k, v in getattr(agent, "Q", {}).items()}
            except Exception:
                qdict = {}
            agent_q_str = json.dumps(qdict, sort_keys=True)
            agent_hash = sha256_str(agent_q_str)

            epoch_event = {
                "event_type": "epoch_commit",
                "experiment": exp_name,
                "seed": seed,
                "load_factor": load_factor,
                "epoch": ep,
                "rl_assignment": epoch_assignments,
                "rl_rms_latency": epoch_rms,
                "agent_hash": agent_hash,
                "snapshot_hash": snapshot_hash,
                "paper_pdf_path": PAPER_PDF_PATH
            }
            emit_audit_event(base_path, ep, epoch_event)
        except Exception as e:
            print("[AUDIT] Failed to emit epoch audit:", e)

        # optionally run ILP on this snapshot
        ilp_result_path = None
        if ilp_interval and (ilp_interval > 0) and (ep % ilp_interval == 0):
            try:
                # run ILP and get result JSON path
                ilp_result_path = try_run_ilp(snapshot_path, ilp_dir)
                if ilp_result_path:
                    # read ILP JSON
                    with open(ilp_result_path, 'r') as fh:
                        ilp_json = json.load(fh)
                    # compute gap and emit ILP audit event
                    try:
                        ilp_hash = sha256_file(ilp_result_path)
                    except Exception:
                        ilp_hash = None
                    gap = compute_optimality_gap(epoch_rms, ilp_json)
                    ilp_event = {
                        "event_type": "ilp_verification",
                        "experiment": exp_name,
                        "seed": seed,
                        "load_factor": load_factor,
                        "epoch": ep,
                        "ilp_assignment": ilp_json.get("ilp_assignments") or ilp_json.get("assignments") or {},
                        "ilp_rms_latency": ilp_json.get("ilp_rms_latency") or ilp_json.get("ilp_rms"),
                        "optimality_gap": gap,
                        "ilp_hash": ilp_hash,
                        "paper_pdf_path": PAPER_PDF_PATH
                    }
                    emit_audit_event(base_path, ep, ilp_event)
            except Exception as e:
                print("[ILP] ILP execution failed:", e)

        # checkpoint agent occasionally
        if ep % max(1, episodes // 5) == 0 or ep == episodes:
            ckpt_path = os.path.join(ckpt_dir, f"agent_ep_{ep}.pkl")
            try:
                agent.save(ckpt_path)
            except Exception as e:
                print("[WARN] agent.save failed:", e)

        # progress print
        if ep % max(1, episodes // 10) == 0 or ep == episodes:
            print(f"[INFO] Episode {ep}/{episodes} | Reward: {epoch_reward:.4f} | Path: {len(epoch_assignments)}")

    # write rewards.csv
    rewards_csv = os.path.join(logs_dir, "rewards.csv")
    with open(rewards_csv, 'w') as fh:
        fh.write("epoch,reward\n")
        for r in rewards_log:
            fh.write(f"{r['epoch']},{r['reward']}\n")

    # final checkpoint
    try:
        agent.save(os.path.join(ckpt_dir, "agent_final.pkl"))
    except Exception as e:
        print("[WARN] final agent.save failed:", e)

    print("[DONE] Results saved at:", base_path)
    return base_path

# ----------------- CLI -----------------
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", required=True)
    p.add_argument("--load", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--ilp-interval", type=int, default=0, help="Run ILP every N epochs (0=disabled)")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    run_training(exp_name=args.exp_name,
                 load_factor=args.load,
                 seed=args.seed,
                 episodes=args.episodes,
                 k_paths=args.k,
                 ilp_interval=args.ilp_interval,
                 base_results_dir=args.results_dir)

if __name__ == "__main__":
    cli()
