#!/bin/bash
source .venv/bin/activate

python run_experiment.py --exp-name final_runs \
  --loads 0.3 0.5 0.7 0.9 1.1 \
  --seeds 0 1 2 3 4 \
  --episodes 200 \
  --k 3 \
  --jobs 4
