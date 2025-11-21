# CA-MBE-QLMR: Trustworthy & Congestion-Aware Multipath Routing in SDN

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research_Prototype-orange.svg)]()
[![Blockchain](https://img.shields.io/badge/Audit-Blockchain_Ready-purple.svg)]()

> **A Hybrid Approach using Max-Boltzmann Q-Learning, ILP Verification, and Blockchain Auditing.**

---

## 📌 Project Overview

This repository implements **CA-MBE-QLMR**, a Congestion-Aware Max-Boltzmann Exploration Q-Learning Multipath Routing algorithm for Software-Defined Networks (SDN).

The system ensures:
* ✅ **Efficient multipath routing** under dynamic loads
* ✅ **Congestion-aware action masking (CA)**
* ✅ **Max-Boltzmann exploration (MBE)** for stable RL convergence
* ✅ **ILP verification** for optimality checking
* ✅ **Blockchain-ready audit trails** for trust and tamper-proof routing decisions

This repository reproduces the full experimental pipeline described in:
📄 **Trustworthy and Congestion-Aware Multipath Routing in SDN** (Provenance file included inside project environment).

---

## 🎯 Key Features

| Feature | Description |
| :--- | :--- |
| **Max-Boltzmann Q-Learning** | Stabilizes RL with a hybrid softmax-epsilon exploration policy. |
| **Congestion-Aware Masking** | Masks path choices based on instantaneous link residual capacity. |
| **ILP Verification** | Validates RL outputs using ground-truth Integer Linear Programming. |
| **Blockchain-Audit Layer** | Every RL/ILP decision is checkpointed and hashed for immutability. |
| **Ablation Suite** | Includes **No-CA** and **No-MBE** experiments for contribution validation. |
| **Robustness Suite** | Link failure + overload stress tests. |
| **Fully Reproducible** | Automated scripts, environment files, and artifact packaging. |

---

## 🧠 System Architecture

```text
                    +---------------------------+
                    |        RL Agent           |
                    |  (CA + MBE Q-Learning)    |
                    +------------+--------------+
                                 |
                                 v
                   +-------------+-------------+
                   |       Path Selection       |
                   |  k-shortest path + masking |
                   +-------------+--------------+
                                 |
                                 v
+--------+     +-----------------+-----------------+     +-----------+
| Topo   | --> |   SDN Simulator (Python-based)    | --> |  Metrics  |
| Graph  |     |   (latency, load, residual cap)   |     |  Logs     |
+--------+     +-----------------+-----------------+     +-----------+
                                 |
                                 | Snapshots (epoch_x.json)
                                 v
                      +----------+----------+
                      |  ILP Verification   |
                      | (CBC / OR-Tools)    |
                      +----------+----------+
                                 |
                                 | ILP JSON
                                 v
                     +-----------+------------+
                     | Blockchain Audit Layer |
                     | (hashes + meta events) |
                     +------------------------+
```

---

## 📦 Project Structure

```text
sdn_rl_project/
│
├── train/                 # RL trainer with Audit Layer
├── topology/              # Network topology + k-shortest paths
├── sim/                   # SDN simulator (latency, loads, link caps)
├── rl/                    # CA-MBE Q-learning agent(s)
├── ilp/                   # ILP solver + snapshot verifier
├── viz/                   # Plotting scripts for paper figures
├── ablation/              # No-CA, No-MBE experiments
├── robustness/            # Link-failure + overload tests
├── utils/                 # Audit emitter + hashing utilities
│
├── results/               # Experimental outputs (RL, ILP, Audit)
├── figures/               # Generated plots
│
├── run_experiment.py      # Main experiment runner
├── run_ablation.sh        # Ablation script wrapper
├── run_robustness.sh      # Robustness script wrapper
├── run_ilp_batch.sh       # Batch ILP solver
│
├── requirements.txt       # Python dependencies
├── REPRODUCE.md           # Full reproducibility instructions
├── AUDIT_SCHEMA.md        # Blockchain audit event schema
├── create_artifact.sh     # Creates reproducibility artifact
└── README.md              # Project documentation
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)<username>/SDN-CA-MBE-Routing.git
cd sdn_rl_project

# 2. Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt
```

---

## 🚀 Running Experiments

### 1. Baseline: Main Results
This runs the full training loop across multiple seeds and load factors.

```bash
python run_experiment.py \
  --exp-name final_runs \
  --loads 0.3 0.5 0.7 0.9 1.1 \
  --seeds 0 1 2 3 4 \
  --episodes 200 \
  --k 3 \
  --jobs 4
```

### 2. Ablation Studies
Verify the contribution of specific components.

* **No Congestion-Aware Mask (No-CA):**
    ```bash
    python ablation/run_ablation.py --variant no_ca --exp ablation_no_ca \
      --loads 0.3 0.7 1.1 --seeds 0 1 2 --episodes 100 --k 3
    ```

* **No Max-Boltzmann Exploration (No-MBE):**
    ```bash
    python ablation/run_ablation.py --variant no_mbe --exp ablation_no_mbe \
      --loads 0.3 0.7 1.1 --seeds 0 1 2 --episodes 100 --k 3
    ```

### 3. Robustness Tests
* **Link Failure:**
    ```bash
    python robustness/run_faulty_topology.py
    ```
* **Overload Stress:**
    ```bash
    python robustness/run_overload.py
    ```

---

## 🧮 ILP Verification & Visualization

### Run ILP on snapshots
Calculates the optimality gap between RL and the optimal solution.
```bash
bash run_ilp_batch.sh
```

### Generating Figures
```bash
# Convergence Plot
python viz/plot_convergence.py --exp results/final_runs --out figures/convergence

# Latency vs Load
python viz/plot_latency_vs_load.py --exp results/final_runs --out figures/latency

# Optimality Gap
python viz/plot_optimality_gap.py --exp results/final_runs --out figures/opt_gap
```

---

## 🔐 Blockchain Audit Layer (Tamper-Proof Logging)

Every epoch generates immutable audit logs under:
`results/<exp>/load_<lf>/seed_<s>/audit_events/<epoch>.json`

Each audit event includes:
* 🔑 **Snapshot Hash**
* 📊 **RL Assignment & Latency**
* 🧠 **Q-Table Hash**
* ⚖️ **ILP Hash** (when available)
* 🔗 **Merged Integrity Hash**

> A blockchain client can directly ingest these JSON files as immutable logs. See `AUDIT_SCHEMA.md` for the schema definition.

---

## 📦 Reproducibility Artifact

Create a research-grade artifact (`.tar.gz`) with integrity hashes:

```bash
bash create_artifact.sh
sha256sum artifact/ca_mbe_rl_artifact.tar.gz > artifact/integrity_hash.txt
```

---

## 🤝 Contributing

Pull requests are welcome! If you want to help add:
* Additional topologies
* New RL agents
* Real OpenFlow controller integration

Please open an issue to discuss.
