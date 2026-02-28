#!/usr/bin/env bash
# Downloads PTB-XL v1.0.3 from PhysioNet
# Requires wget. Run: bash download_data.sh

set -e

DATA_DIR="./data/ptbxl"
mkdir -p "$DATA_DIR"

echo "Downloading PTB-XL v1.0.3 ..."
wget -r -N -c -np \
  --directory-prefix="$DATA_DIR" \
  --no-host-directories \
  --cut-dirs=3 \
  https://physionet.org/files/ptb-xl/1.0.3/

echo "Done. Data stored in $DATA_DIR"
