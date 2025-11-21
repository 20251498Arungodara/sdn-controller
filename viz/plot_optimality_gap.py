"""
viz/plot_optimality_gap.py

Compute and plot Optimality Gap = (RL_RMS - ILP_RMS) / ILP_RMS
for runs under results/<exp_name>/load_<lf>/seed_<s>/

Requirements:
 - Each run should have rewards.csv at .../logs/rewards.csv
 - ILP outputs should be under .../ilp_results/ilp_result_epoch_*.json
   (trainer moves ILP outputs into ilp_results when ILP is enabled).
 - If multiple ILP files exist for a seed, this script uses the ILP result
   matching the last epoch (or the first found).

Outputs:
 - figures/optimality_gap.png / .pdf
 - figures/optimality_gap_data.csv
"""
import os
import argparse
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

# Provenance (local file path used as URL)
PAPER_PDF_URL = "/mnt/data/Trustworthy_and_Congestion_Aware_Multipath_Routing_in_SDN__A_Hybrid_Approach_using_Max_Boltzmann_Q_Learning__ILP_Verification__and_Blockchain_Auditing.pdf"

def read_last_reward(rewards_csv):
    """Return last epoch reward (float) from rewards.csv"""
    try:
        with open(rewards_csv, 'r') as fh:
            lines = fh.read().strip().splitlines()
            if len(lines) < 2:
                return None
            last = lines[-1].split(',')
            return float(last[1])
    except Exception:
        return None

def find_ilp_rms(ilp_dir):
    """Find an ILP JSON and extract 'ilp_rms_latency' or 'ilp_rms' or 'ilp_rms_latency' keys."""
    pattern = os.path.join(ilp_dir, "ilp_result_*json")
    files = sorted(glob.glob(pattern))
    if not files:
        # also try top-level logs folder pattern
        files = sorted(glob.glob(os.path.join("logs", "ilp_result_*json")))
    if not files:
        return None
    # try to find one with numeric ilp_rms_latency
    for f in files[::-1]:  # prefer latest
        try:
            with open(f, 'r') as fh:
                j = json.load(fh)
            # common keys used by our ilp script: 'ilp_rms_latency' or 'ilp_rms'
            if 'ilp_rms_latency' in j and j['ilp_rms_latency'] is not None:
                return float(j['ilp_rms_latency'])
            if 'ilp_rms' in j and j['ilp_rms'] is not None:
                return float(j['ilp_rms'])
            # fallback: ilp_objective not useful for RMS; check for 'ilp_rms_latency' in nested
        except Exception:
            continue
    return None

def collect_gaps(exp_root):
    """
    Walk results/<exp_root>/load_*/seed_*/ and compute gap per run.
    Returns: dict load_factor -> list of gaps (fractions, e.g., 0.12 => 12%)
    """
    runs = glob.glob(os.path.join(exp_root, "load_*", "seed_*"))
    loads_map = {}
    for r in runs:
        parts = r.split(os.sep)
        # load label is the component that starts with load_
        load_label = next((p for p in parts if p.startswith("load_")), None)
        if not load_label:
            continue
        try:
            load_factor = float(load_label.split("_")[1])
        except Exception:
            # fallback: parse with replace
            load_factor = float(load_label.replace("load_", ""))
        rewards_csv = os.path.join(r, "logs", "rewards.csv")
        ilp_dir = os.path.join(r, "ilp_results")
        last_reward = read_last_reward(rewards_csv) if os.path.exists(rewards_csv) else None
        ilp_rms = find_ilp_rms(ilp_dir) if os.path.exists(ilp_dir) else None
        # if ILP not present in per-run ilp_results, try top-level
        if ilp_rms is None:
            ilp_rms = find_ilp_rms(os.path.join("logs"))  # fallback to top-level logs
        if last_reward is None or ilp_rms is None:
            # skip runs without both data items
            continue
        rl_rms = -last_reward  # reward = -RMS
        if ilp_rms <= 0:
            continue
        gap = (rl_rms - ilp_rms) / ilp_rms
        loads_map.setdefault(load_factor, []).append(gap)
    return loads_map

def aggregate_and_plot(loads_map, out_prefix):
    loads = sorted(loads_map.keys())
    means = [np.mean(loads_map[l]) for l in loads]
    stds = [np.std(loads_map[l]) for l in loads]
    ns = [len(loads_map[l]) for l in loads]

    # write CSV
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    csv_path = out_prefix + "_data.csv"
    with open(csv_path, 'w') as fh:
        fh.write("load,mean_gap,std_gap,n\n")
        for l, m, s, n in zip(loads, means, stds, ns):
            fh.write(f"{l},{m},{s},{n}\n")
    print("Wrote:", csv_path)

    # plot percent gap
    plt.figure(figsize=(7,4))
    x = np.array(loads)
    y = np.array(means) * 100.0
    yerr = np.array(stds) * 100.0
    plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=4, label='Optimality gap (RL vs ILP)')
    plt.xlabel("Offered load (scale factor)")
    plt.ylabel("Optimality gap (%)")
    plt.title("Optimality gap (RL vs ILP) — provenance: {}".format(os.path.basename(PAPER_PDF_URL)))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_prefix + ".pdf", bbox_inches="tight")
    print("Saved:", out_prefix + ".png", out_prefix + ".pdf")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="results/ca_mbe_grid", help="experiment results folder")
    p.add_argument("--out", default="figures/optimality_gap", help="output prefix")
    args = p.parse_args()

    loads_map = collect_gaps(args.exp)
    if not loads_map:
        raise RuntimeError("No runs with both rewards.csv and ILP results found under " + args.exp)
    aggregate_and_plot(loads_map, args.out)

if __name__ == "__main__":
    main()
