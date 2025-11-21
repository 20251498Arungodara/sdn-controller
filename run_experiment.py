#!/usr/bin/env python3
"""
run_experiment.py

Experiment runner that sweeps load factors and seeds using Option A (scale demands).
"""
import argparse
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from train.trainer import run_training   # trainer you already implemented

PAPER_PDF_PATH = "/mnt/data/Trustworthy_and_Congestion_Aware_Multipath_Routing_in_SDN__A_Hybrid_Approach_using_Max_Boltzmann_Q_Learning__ILP_Verification__and_Blockchain_Auditing.pdf"

BASE_FLOWS = [
    {'id':'f1','src':'h1','dst':'h2','demand':3.0},
    {'id':'f2','src':'h1','dst':'h2','demand':2.0},
    {'id':'f3','src':'h1','dst':'h2','demand':1.5},
]

def scaled_flows(load_factor):
    """Scale demands by load factor."""
    return [
        {
            'id': f['id'],
            'src': f['src'],
            'dst': f['dst'],
            'demand': float(f['demand']) * float(load_factor)
        }
        for f in BASE_FLOWS
    ]

def ensure_exp_dirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def _run_single(exp_name, load_factor, seed, episodes, k_paths, ilp_interval):
    """Worker job executed inside ProcessPoolExecutor."""
    lf_label = f"load_{float(load_factor):.2f}"
    sub_exp = os.path.join(exp_name, lf_label)

    base_path = os.path.join("results", sub_exp, f"seed_{seed}")
    os.makedirs(os.path.join(base_path, "logs"), exist_ok=True)

    # Write metadata
    meta = {
        'exp_name': exp_name,
        'sub_experiment': sub_exp,
        'load_factor': load_factor,
        'seed': seed,
        'episodes': episodes,
        'k_paths': k_paths,
        'ilp_interval': ilp_interval,
        'paper_pdf': PAPER_PDF_PATH,
        'time': time.asctime(),
    }
    with open(os.path.join(base_path, "meta.json"), 'w') as fh:
        json.dump(meta, fh, indent=2)

    flows = scaled_flows(load_factor)

    run_training(sub_exp, seed, episodes, k_paths, flows=flows, ilp_interval=ilp_interval)

    return {'load_factor': load_factor, 'seed': seed, 'status': "done"}

def aggregate_summary(exp_name, loads, seeds, out_file):
    """Extract last reward from each run."""
    rows = []

    for lf in loads:
        lf_label = f"load_{lf:.2f}"
        for s in seeds:
            rewards_csv = os.path.join(
                "results", exp_name, lf_label, f"seed_{s}", "logs", "rewards.csv"
            )
            last_reward = None

            if os.path.exists(rewards_csv):
                with open(rewards_csv, 'r') as fh:
                    lines = fh.read().strip().splitlines()
                    if len(lines) >= 2:
                        last_reward = lines[-1].split(',')[1]

            rows.append({
                'load': lf,
                'seed': s,
                'last_reward': last_reward
            })

    with open(out_file, 'w') as fh:
        fh.write("load,seed,last_reward\n")
        for r in rows:
            fh.write(f"{r['load']},{r['seed']},{r['last_reward']}\n")

    print("Summary file created:", out_file)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", default="ca_mbe_grid")
    p.add_argument("--loads", nargs="+", type=float, default=[0.5, 0.8, 1.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--ilp-interval", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1)
    return p.parse_args()

def main():
    args = parse_args()
    exp_name = args.exp_name
    loads = args.loads
    seeds = args.seeds

    ensure_exp_dirs(os.path.join("results", exp_name))

    tasks = []
    for lf in loads:
        for seed in seeds:
            tasks.append((exp_name, lf, seed, args.episodes, args.k, args.ilp_interval))

    print(f"Running {len(tasks)} tasks with {args.jobs} workers...")

    futures = []
    with ProcessPoolExecutor(max_workers=args.jobs) as exe:
        for t in tasks:
            futures.append(exe.submit(_run_single, *t))

        for f in as_completed(futures):
            try:
                print("Task result:", f.result())
            except Exception as e:
                print("Task failed:", e)

    summary_path = os.path.join("results", exp_name, "summary_last_rewards.csv")
    aggregate_summary(exp_name, loads, seeds, summary_path)

    print("Experiment completed.")
    print("Summary written to:", summary_path)

if __name__ == "__main__":
    main()
