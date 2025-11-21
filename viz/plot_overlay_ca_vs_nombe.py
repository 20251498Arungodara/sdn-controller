import os, glob, csv, numpy as np
import matplotlib.pyplot as plt

def read_rewards(root):
    pattern = os.path.join(root, "load_*", "seed_*", "logs", "rewards.csv")
    files = sorted(glob.glob(pattern))
    all_runs = []
    for f in files:
        vals = []
        with open(f) as fh:
            rdr = csv.reader(fh)
            next(rdr, None)
            for row in rdr:
                if not row: continue
                vals.append(float(row[1]))
        all_runs.append(vals)
    if not all_runs:
        return None
    maxlen = max(len(x) for x in all_runs)
    arr = np.full((len(all_runs), maxlen), np.nan)
    for i, x in enumerate(all_runs):
        arr[i, :len(x)] = x
    return arr

def plot_overlay(exp_ref, exp_ablate, out_prefix, max_ep=None):
    A = read_rewards(exp_ref)
    B = read_rewards(exp_ablate)
    if A is None or B is None:
        print("Missing runs for one of the experiments.")
        return
    if max_ep:
        A = A[:, :max_ep]
        B = B[:, :max_ep]
    meanA = np.nanmean(A, axis=0); stdA = np.nanstd(A, axis=0)
    meanB = np.nanmean(B, axis=0); stdB = np.nanstd(B, axis=0)
    x = np.arange(1, meanA.shape[0]+1)
    plt.figure(figsize=(8,4))
    plt.plot(x, meanA, label="Original (CA-MBE)", linewidth=2)
    plt.fill_between(x, meanA-stdA, meanA+stdA, alpha=0.2)
    plt.plot(x, meanB, label="Ablation: No-MBE (ε-greedy)", linewidth=2)
    plt.fill_between(x, meanB-stdB, alpha=0.2)
    plt.xlabel("Episode"); plt.ylabel("Reward (negative RMS latency)")
    plt.title("Convergence: CA-MBE vs No-MBE")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=300); plt.savefig(out_prefix + ".pdf")
    plt.close()
    print("Saved:", out_prefix + ".png", out_prefix + ".pdf")

if __name__ == "__main__":
    plot_overlay("results/final_runs", "results/ablation_no_mbe", "figures/overlay_convergence_ca_vs_nombe", max_ep=30)
