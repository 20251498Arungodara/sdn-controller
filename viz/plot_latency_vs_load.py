"""
viz/plot_latency_vs_load.py

Plots latency vs offered load using results from:
results/<exp_name>/load_<lf>/seed_<seed>/logs/epoch_*.json

Latency = RMS latency across flows (already stored in epoch snapshots)
"""

import os
import argparse
import json
import glob
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# provenance
PAPER_PDF = "/mnt/data/Trustworthy_and_Congestion_Aware_Multipath_Routing_in_SDN__A_Hybrid_Approach_using_Max_Boltzmann_Q_Learning__ILP_Verification__and_Blockchain_Auditing.pdf"

def rms_latency_from_snapshot(path):
    """Compute RMS latency from epoch JSON."""
    try:
        with open(path, "r") as fh:
            snap = json.load(fh)
    except:
        return None

    latencies = []
    loads = snap.get("link_loads", {})
    # but actual per-flow latency is not saved directly
    # however, rewards are negative RMS, so use that:

    # reward = - RMS(lat)
    # RMS(lat) = -reward
    # Here we reconstruct reward from file name (snapshot saved after reward computed).
    # But instead, read reward from snapshot if stored:

    # NOTE: best is to recompute reward using link loads stored + path info
    # For now, rely on simulator reward stored in training if saved.

    # If you later store per-episode reward in snapshots, replace this logic.

    # Try to locate "episode_reward" if present (not required for basic plot)
    return None

def extract_reward_from_rewards_csv(csv_path):
    """Reads epoch,reward from rewards.csv and returns list of rewards."""
    vals = []
    try:
        with open(csv_path, "r") as fh:
            lines = fh.read().strip().splitlines()
        for line in lines[1:]:
            ep, r = line.split(",")
            vals.append(float(r))
        return vals
    except:
        return []

def load_latency_by_load(exp_root):
    """
    Returns:
        loads: sorted list of load factors (float)
        mean_lat: mean latency per load
        std_lat: std latency per load
    """
    pattern = os.path.join(exp_root, "load_*", "seed_*", "logs", "rewards.csv")
    files = sorted(glob.glob(pattern))
    groups = defaultdict(list)

    for fp in files:
        parts = fp.split(os.sep)
        load_label = next((p for p in parts if p.startswith("load_")), None)
        if load_label:
            load_factor = float(load_label.split("_")[1])
            rewards = extract_reward_from_rewards_csv(fp)
            if rewards:
                # convert reward to RMS latency
                # reward = -RMS => RMS = -reward
                latencies = [-r for r in rewards]
                # average latency across episodes (steady-state approx)
                avg_lat = np.mean(latencies)
                groups[load_factor].append(avg_lat)

    loads = sorted(groups.keys())
    mean_lat = [np.mean(groups[lf]) for lf in loads]
    std_lat = [np.std(groups[lf]) for lf in loads]

    return loads, mean_lat, std_lat

def plot_latency_vs_load(loads, mean_lat, std_lat, out_prefix):
    plt.figure(figsize=(7,4))
    loads_f = np.array(loads)
    mean_f = np.array(mean_lat)
    std_f = np.array(std_lat)

    plt.errorbar(loads_f, mean_f, yerr=std_f, fmt='-o', capsize=4, label="CA-MBE-QLMR")

    plt.xlabel("Offered Load (scaling factor)")
    plt.ylabel("Avg RMS Latency (ms)")
    plt.title("Latency vs Load\n(Provenance: {})".format(os.path.basename(PAPER_PDF)))
    plt.grid(True)
    plt.legend()

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_prefix + ".pdf", bbox_inches="tight")
    print("Saved:", out_prefix + ".png", "and .pdf")

def write_csv(loads, mean_lat, std_lat, out_csv):
    with open(out_csv, "w") as fh:
        fh.write("load,mean_latency,std_latency\n")
        for lf, m, s in zip(loads, mean_lat, std_lat):
            fh.write(f"{lf},{m},{s}\n")
    print("Saved CSV:", out_csv)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="results/ca_mbe_grid")
    p.add_argument("--out", default="figures/latency_vs_load")
    args = p.parse_args()

    loads, mean_lat, std_lat = load_latency_by_load(args.exp)
    if not loads:
        raise RuntimeError(f"No latency data found under {args.exp}")

    write_csv(loads, mean_lat, std_lat, args.out + "_data.csv")
    plot_latency_vs_load(loads, mean_lat, std_lat, args.out)

if __name__ == "__main__":
    main()
