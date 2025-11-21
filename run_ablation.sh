#!/bin/bash
source .venv/bin/activate

python ablation/run_ablation.py --variant no_ca  --exp ablation_no_ca  --loads 0.3 0.7 1.1 --seeds 0 1 2 --episodes 100 --k 3
python ablation/run_ablation.py --variant no_mbe --exp ablation_no_mbe --loads 0.3 0.7 1.1 --seeds 0 1 2 --episodes 100 --k 3
