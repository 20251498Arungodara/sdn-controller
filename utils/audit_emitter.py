import json, os, hashlib, time
from pathlib import Path

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()

def sha256_str(s: str):
    return hashlib.sha256(s.encode()).hexdigest()

def emit_audit_event(base_dir, epoch, event_dict):
    """
    base_dir: path like results/final_runs/load_0.50/seed_0
    epoch: integer
    event_dict: dictionary of event fields (rl, ilp, metadata)
    
    Output file path:
        results/.../audit_events/<epoch>.json
    """
    audit_dir = os.path.join(base_dir, "audit_events")
    os.makedirs(audit_dir, exist_ok=True)
    
    # fill timestamp
    event_dict["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    
    # compute merged hash if fields exist
    merged = ""
    if "agent_hash" in event_dict: merged += event_dict["agent_hash"]
    if "snapshot_hash" in event_dict: merged += event_dict["snapshot_hash"]
    if "ilp_hash" in event_dict: merged += event_dict["ilp_hash"]
    event_dict["merged_hash"] = sha256_str(merged)
    
    out_path = os.path.join(audit_dir, f"{epoch}.json")
    with open(out_path, "w") as fh:
        json.dump(event_dict, fh, indent=2)
    
    print(f"[AUDIT] Wrote audit event: {out_path}")
    return out_path
