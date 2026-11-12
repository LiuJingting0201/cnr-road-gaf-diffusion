set -euo pipefail
root=$PWD
base=/workspace/cnr-experiments/beijing-geolife-difftraj
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
export PYTHONPATH=/workspace/cnr-experiments/difftraj-nov11 OMP_NUM_THREADS=1
pids=()
gpu=0
for arm in C VC; do
 for seed in 42 43; do
 (
 export CUDA_VISIBLE_DEVICES=$gpu DIFFTRAJ_TRAINING_SEED=$seed
 for mode in bike bus car taxi; do
  dir="$root/$arm-$seed-$mode"; mkdir "$dir"; cd "$dir"
  ln -s "$base/pilot.npz" "$base/seed-$seed/$arm-8000.pt" "$base/seed-$seed/$arm-8000-class.pt" .
  "$py" ../pilot.py "$arm" 8000 --evaluate --all-validation --save-draws --target-class "$mode" > replay.log 2>&1
 done
 ) & pids+=($!)
 gpu=$((gpu+1))
 done
done
for p in "${pids[@]}"; do wait "$p"; done
touch COMPLETE
