set -euo pipefail
root=$PWD
base=/workspace/cnr-experiments/beijing-geolife-difftraj
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
export PYTHONPATH=/workspace/cnr-experiments/difftraj-nov11 OMP_NUM_THREADS=1
pids=()
gpu=0
for header in no_mean no_distance_cues; do
 for seed in 42 43; do
 (
 export CUDA_VISIBLE_DEVICES=$gpu DIFFTRAJ_TRAINING_SEED=$seed GEOLIFE_HEADER=$header
 mkdir "$root/$header-$seed"; cd "$root/$header-$seed"
 ln -s "$base/pilot.npz" .
 "$py" -u ../pilot.py C 8000 --all-validation > train.log 2>&1
 for mode in bike bus; do
   "$py" ../pilot.py C 8000 --evaluate --all-validation --save-draws --target-class "$mode" > "$mode.log" 2>&1
   mv private-C-8000.npz "$mode.npz"
 done
 touch COMPLETE
 ) & pids+=($!)
 gpu=$((gpu+1))
 done
done
for p in "${pids[@]}"; do wait "$p"; done
touch COMPLETE
