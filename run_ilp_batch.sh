#!/bin/bash
source .venv/bin/activate

for lf in 0.30 0.50 0.70 0.90 1.10; do
  for s in 0 1 2 3 4; do
    SNAP="results/final_runs/load_${lf}/seed_${s}/logs/epoch_200.json"
    if [ -f "$SNAP" ]; then
      echo "Running ILP on $SNAP"
      python ilp/ilp_from_snapshot.py "$SNAP"
      mv ilp_result_epoch* "results/final_runs/load_${lf}/seed_${s}/ilp_results/" 2>/dev/null
    fi
  done
done
