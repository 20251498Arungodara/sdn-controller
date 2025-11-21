#!/bin/bash
source .venv/bin/activate

python robustness/run_faulty_topology.py
python robustness/run_overload.py
