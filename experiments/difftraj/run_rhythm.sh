set -euo pipefail
root=$PWD
base=/workspace/cnr-experiments/beijing-expanded60-data
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
export PYTHONPATH=/workspace/cnr-experiments/difftraj-nov11 OMP_NUM_THREADS=1
pids=()
gpu=0
for arm in C CR; do
 for seed in 42 43; do
 (
 export CUDA_VISIBLE_DEVICES=$gpu DIFFTRAJ_TRAINING_SEED=$seed GEOLIFE_HEADER=full
 mkdir "$root/$arm-$seed"; cd "$root/$arm-$seed"
 ln -s "$base/beijing60.npz" pilot.npz
 "$py" -u ../pilot.py "$arm" 8000 --all-validation > train.log 2>&1
 for mode in bike bus; do
   "$py" ../pilot.py "$arm" 8000 --evaluate --all-validation --save-draws --target-class "$mode" > "$mode.log" 2>&1
   mv "private-$arm-8000.npz" "$mode.npz"
 done
 touch COMPLETE
 ) & pids+=($!)
 gpu=$((gpu+1))
 done
done
for p in "${pids[@]}"; do wait "$p"; done
touch COMPLETE
