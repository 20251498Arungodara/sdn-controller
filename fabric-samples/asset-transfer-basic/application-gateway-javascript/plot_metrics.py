import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv("metrics.csv")

# ===============================
# 1. Latency vs Epoch
# ===============================
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["latency_ms"], marker='o')
plt.xlabel("Epoch")
plt.ylabel("Latency (ms)")
plt.title("Epoch Commit Latency")
plt.grid(True)
plt.tight_layout()
plt.savefig("latency_vs_epoch.png")
plt.close()

# ===============================
# 2. Throughput
# ===============================
total_time_sec = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 1000
throughput = len(df) / total_time_sec

plt.figure(figsize=(5, 4))
plt.bar(["Throughput"], [throughput])
plt.ylabel("Epochs / second")
plt.title("System Throughput")
plt.tight_layout()
plt.savefig("throughput.png")
plt.close()

print("✅ Plots generated:")
print(" - latency_vs_epoch.png")
print(" - throughput.png")
print(f"📈 Throughput = {throughput:.3f} epochs/sec")



# ===============================
# 3. Latency CDF
# ===============================
latencies = df["latency_ms"].sort_values()
cdf = latencies.rank(method="average", pct=True)

plt.figure(figsize=(8, 5))
plt.plot(latencies, cdf)
plt.xlabel("Latency (ms)")
plt.ylabel("CDF")
plt.title("Latency CDF of Epoch Commit")
plt.grid(True)
plt.tight_layout()
plt.savefig("latency_cdf.png")
plt.close()

# Percentiles (for paper text)
p50 = latencies.quantile(0.50)
p95 = latencies.quantile(0.95)
p99 = latencies.quantile(0.99)

print("📊 Latency percentiles:")
print(f"  P50 (median): {p50:.2f} ms")
print(f"  P95:          {p95:.2f} ms")
print(f"  P99:          {p99:.2f} ms")
