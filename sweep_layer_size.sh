#!/bin/bash
# Layer-size sweep: 6 sizes x 3 seeds, 200 epochs max (early stop by patience).
# Re-running skips any (size, seed) that already finished, so it is safe to restart.
# Log: sweep_log.txt
cd "$(dirname "$0")"
SIZES="128:64 256:128 512:256 1024:512 2048:1024 4096:2048"
for seed in 1 2 3; do
  for size in $SIZES; do
    n1=${size%:*}; n2=${size#*:}
    if grep -ls '"finished": true' runs/48_3922_${n1}-${n2}_s${seed}_*/config_preview.json >/dev/null 2>&1; then
      echo "skip ${n1}-${n2} seed ${seed} (done)"; continue
    fi
    echo "===== $(date '+%F %T') start ${n1}-${n2} seed ${seed}"
    python3 NN_Classifier.py --n1 "$n1" --n2 "$n2" --seed "$seed" --epochs 200
    echo "===== $(date '+%F %T') end ${n1}-${n2} seed ${seed} (exit $?)"
  done
done
echo "ALL DONE $(date '+%F %T')"
