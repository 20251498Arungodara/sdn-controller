#!/bin/bash
echo "[PACKAGING] Creating reproducibility artifact..."

mkdir -p artifact
tar -czf artifact/ca_mbe_rl_artifact.tar.gz \
  train/ \
  topology/ \
  sim/ \
  rl/ \
  viz/ \
  ablation/ \
  robustness/ \
  utils/ \
  results/ \
  REPRODUCE.md \
  requirements.txt \
  environment.txt \
  AUDIT_SCHEMA.md

echo "[DONE] Artifact stored at artifact/ca_mbe_rl_artifact.tar.gz"
