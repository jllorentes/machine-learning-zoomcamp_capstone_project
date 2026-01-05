#!/bin/bash

# 'set -e' hace que el script se detenga inmediatamente si algún comando falla
set -e

echo "========================================"
echo "STARTING TRAINING PIPELINE"
echo "========================================"

# STEP 1: Prepare Data
echo ""
echo "[1/2] Running: prepare_dataset.py"
python scripts/prepare_dataset.py

# STEP 2: Train
echo ""
echo "[2/2] Running: train.py"
python scripts/train.py

echo ""
echo "========================================"
echo "PIPELINE SUCCESSFULLY COMPLETED"
echo "========================================"