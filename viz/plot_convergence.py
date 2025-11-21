"""
viz/plot_convergence.py

Plot convergence: reward vs episodes (mean ± std across seeds) for each load factor.

Usage:
    source .venv/bin/activate
    python viz/plot_convergence.py --exp results/ca_mbe_grid --out figures/convergence_reward --max-episodes 200

Notes:
- Expects folder layout: results/<exp_name>/load_<lf>/seed_<s>/logs/rewards.csv
- rewards.csv format: epoch,reward
- Saves:
    - figures/convergence_reward.png
    - figures/convergence_reward.pdf
    - figures/convergence_aggregated.csv
"""
import os
import argparse
import glob
import csv
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import json

# Provenance: uploaded paper path
PAPER_PDF = "/mnt/data/Trustworthy_and_Congestion_Aware_Multipath_Routing_in_SDN__A_Hybrid_Approach_using_Max_Boltzmann_Q_Learning__ILP_Verification__and_Blockchain_Auditing.pdf"

def read_rewards_csv(path):
    """Read epoch,reward lines -> dict epoch->float reward"""
    out = {}
    try:
        with open(path, 'r') as fh:
            rdr = csv.reader(fh)
            header = next(rdr, None)
            for row in rdr:
                if not row: 
                    continue
                ep = int(row[0])
                r = float(row[1])
                out[ep] = r
    except Exception:
        return {}
    return out

def aggregate_rewards(exp_root, max_episodes=None):
    """
    Walk results/<exp_root>/load_*/seed_*/logs/rewards.csv
    Return: dict load_label -> 2D array (n_seeds x n_episodes) with NaNs padded
    """
    pattern = os.path.join(exp_root, "load_*", "seed_*", "logs", "rewards.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No rewards.csv found with pattern {pattern}")
    # group by load_label
    groups = defaultdict(list)
    for fp in files:
        parts = fp.split(os.sep)
        # expect .../results/<exp_root>/load_0.50/seed_0/logs/rewards.csv
        # find index of exp_root in path
        if exp_root in parts:
            idx = parts.index(exp_root)
            # load_label is next element
            try:
                load_label = parts[idx+1]
            except Exception:
                load_label = "load_unknown"
        else:
            # fallback: find "load_" pattern
            load_label = next((p for p in parts if p.startswith("load_")), "load_unknown")
        groups[load_label].append(fp)

    aggregated = {}
    for load_label, fps in groups.items():
        rewards_list = []
        for f in fps:
            d = read_rewards_csv(f)
            if not d:
                continue
            # build array sorted by epoch index
            max_ep = max(d.keys())
            if max_episodes:
                max_ep = min(max_ep, max_episodes)
            arr = [math.nan] * max_ep
            for ep in range(1, max_ep+1):
                arr[ep-1] = d.get(ep, math.nan)
            rewards_list.append(arr)
        if not rewards_list:
            continue
        # pad to equal length
        maxlen = max(len(a) for a in rewards_list)
        if max_episodes:
            maxlen = min(maxlen, max_episodes)
        A = np.full((len(rewards_list), maxlen), np.nan)
        for i,a in enumerate(rewards_list):
            L = min(len(a), maxlen)
            A[i,:L] = a[:L]
        aggregated[load_label] = A
    return aggregated

def plot_convergence(aggregated, out_prefix):
    """
    aggregated: dict load_label -> array (n_seeds x n_epochs)
    """
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    # plot each load on same axis
    plt.figure(figsize=(7,4))
    for load_label, A in sorted(aggregated.items()):
        mean = np.nanmean(A, axis=0)
        std = np.nanstd(A, axis=0)
        x = np.arange(1, len(mean)+1)
        plt.plot(x, mean, label=load_label)
        plt.fill_between(x, mean-std, mean+std, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Reward (negative RMS latency)")
    plt.title("Convergence: Reward vs Episode\n(provenance: {})".format(os.path.basename(PAPER_PDF)))
    plt.legend()
    plt.grid(True)
    png = out_prefix + ".png"
    pdf = out_prefix + ".pdf"
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print("Saved figures:", png, pdf)

def write_aggregated_csv(aggregated, out_csv):
    """Write mean and std per episode per load into CSV for reproducibility"""
    rows = []
    # collect maximum epoch length
    maxlen = 0
    for A in aggregated.values():
        maxlen = max(maxlen, A.shape[1])
    header = ["load","episode","mean_reward","std_reward","n_seeds"]
    with open(out_csv, 'w') as fh:
        fh.write(",".join(header) + "\n")
        for load_label, A in sorted(aggregated.items()):
            n_seeds = A.shape[0]
            for ep in range(A.shape[1]):
                mean = float(np.nanmean(A[:,ep]))
                std = float(np.nanstd(A[:,ep]))
                fh.write(f"{load_label},{ep+1},{mean},{std},{n_seeds}\n")
    print("Wrote aggregated CSV:", out_csv)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="results/ca_mbe_grid", help="Path to experiment results root")
    p.add_argument("--out", default="figures/convergence_reward", help="Output prefix (png/pdf)")
    p.add_argument("--max-episodes", type=int, default=None, help="Limit episodes (optional)")
    args = p.parse_args()

    exp_root = args.exp
    if not os.path.exists(exp_root):
        raise FileNotFoundError(f"Experiment root not found: {exp_root}")
    aggregated = aggregate_rewards(exp_root, max_episodes=args.max_episodes)
    if not aggregated:
        raise RuntimeError("No aggregated rewards found")
    write_aggregated_csv(aggregated, args.out + "_aggregated.csv")
    plot_convergence(aggregated, args.out)

if __name__ == "__main__":
    import argparse
    main()
