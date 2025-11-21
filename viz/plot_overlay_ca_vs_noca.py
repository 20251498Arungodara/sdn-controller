import os, glob, csv, numpy as np
import matplotlib.pyplot as plt

def read_rewards(root):
    pattern = os.path.join(root, "load_*", "seed_*", "logs", "rewards.csv")
    files = sorted(glob.glob(pattern))
    all_runs = []
    for f in files:
        epochs = []
        vals = []
        with open(f) as fh:
            rdr = csv.reader(fh)
            next(rdr)
            for row in rdr:
                epochs.append(int(row[0]))
                vals.append(float(row[1]))
        all_runs.append(vals)
    if not all_runs:
        return None
    maxlen = max(len(x) for x in all_runs)
    arr = np.full((len(all_runs), maxlen), np.nan)
    for i, x in enumerate(all_runs):
        arr[i, :len(x)] = x
    return arr

def plot_overlay(exp1, exp2, out):
    A = read_rewards(exp1)
    B = read_rewards(exp2)
    if A is None or B is None:
        print("Missing runs")
        return
    
    meanA = np.nanmean(A, axis=0)
    meanB = np.nanmean(B, axis=0)
    
    plt.figure(figsize=(7,4))
    plt.plot(meanA, label="Original (CA-MBE)", linewidth=2)
    plt.plot(meanB, label="Ablation: No CA", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Convergence Comparison: CA-MBE vs No-CA")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out + ".png", dpi=300)
    plt.savefig(out + ".pdf")
    plt.close()
    print("Saved:", out + ".png", out + ".pdf")

if __name__ == "__main__":
    plot_overlay("results/final_runs", "results/ablation_no_ca", "figures/overlay_convergence_ca_vs_noca")
