#!/usr/bin/env bash
# activate project venv quickly
if [ -f "$(pwd)/.venv/bin/activate" ]; then
  source "$(pwd)/.venv/bin/activate"
  echo "Activated venv: $(pwd)/.venv"
else
  echo "No venv found at .venv. Create one with: python3 -m venv .venv"
fi
