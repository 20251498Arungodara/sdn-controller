import os, glob, numpy as np
import matplotlib.pyplot as plt
import csv

def mean_latency_by_load(root):
    pattern = os.path.join(root, "load_*", "seed_*", "logs", "rewards.csv")
    files = sorted(glob.glob(pattern))
    groups = {}
    for f in files:
        parts = f.split(os.sep)
        load_label = next((p for p in parts if p.startswith("load_")), None)
        if load_label is None: continue
        lf = float(load_label.split("_")[1])
        # read rewards -> avg RL RMS = -mean(rewards)
        vals = []
        with open(f) as fh:
            rdr = csv.reader(fh); next(rdr, None)
            for row in rdr:
                if not row: continue
                vals.append(float(row[1]))
        if not vals: continue
        avg_rms = -float(sum(vals)/len(vals))
        groups.setdefault(lf, []).append(avg_rms)
    loads = sorted(groups.keys())
    mean_lat = [np.mean(groups[l]) for l in loads]
    std_lat = [np.std(groups[l]) for l in loads]
    return loads, mean_lat, std_lat

def plot_overlay(exp_ref, exp_ablate, out_prefix):
    l1,m1,s1 = mean_latency_by_load(exp_ref)
    l2,m2,s2 = mean_latency_by_load(exp_ablate)
    plt.figure(figsize=(7,4))
    plt.errorbar(l1, m1, yerr=s1, fmt='-o', capsize=4, label='CA-MBE')
    plt.errorbar(l2, m2, yerr=s2, fmt='-s', capsize=4, label='No-MBE')
    plt.xlabel("Load factor"); plt.ylabel("Avg RMS latency")
    plt.title("Latency vs Load: CA-MBE vs No-MBE")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=300); plt.savefig(out_prefix + ".pdf")
    print("Saved:", out_prefix + ".png", out_prefix + ".pdf")

if __name__ == "__main__":
    plot_overlay("results/final_runs", "results/ablation_no_mbe", "figures/overlay_latency_ca_vs_nombe")
