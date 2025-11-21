# AUDIT_SCHEMA.md
Audit Event Schema for CA-MBE-QLMR Routing System
Provenance Document:
  /mnt/data/Trustworthy_and_Congestion_Aware_Multipath_Routing_in_SDN__A_Hybrid_Approach_using_Max_Boltzmann_Q_Learning__ILP_Verification__and_Blockchain_Auditing.pdf

This schema defines the JSON fields that are emitted by the RL controller during training/evaluation and are later submitted to the blockchain by a separate subsystem.

---

## EVENT STRUCTURE

Each audit event must be a JSON object with the following fields:

### 1. Metadata
- **event_type** (string)  
  "epoch_commit", "ilp_verification", or "final_commit"

- **timestamp** (string; ISO 8601)

- **experiment** (string)  
  The experiment name (e.g., "final_runs", "ablation_no_ca")

- **seed** (integer)

- **load_factor** (float)

- **epoch** (integer)

### 2. RL Agent State
- **rl_assignment** (dict)  
  Mapping flow → chosen path index

- **rl_rms_latency** (float)  
  RMS latency computed by RL simulator for this epoch

- **agent_hash** (string)  
  SHA256 hash of the agent Q-table at this epoch

- **snapshot_hash** (string)  
  SHA256 hash of the epoch_*.json snapshot file

### 3. ILP Verification (included only if ILP was run)
- **ilp_assignment** (dict)  
- **ilp_rms_latency** (float)  
- **optimality_gap** (float)

### 4. Integrity
- **merged_hash** (string)  
  SHA256 hash of (agent_hash + snapshot_hash + ilp_hash)

### 5. Provenance
- **paper_pdf_path** (string)  

---

## STORAGE LOCATION

Audit events are stored under:

results/<experiment>/load_<lf>/seed_<s>/audit_events/<epoch>.json

