set -euo pipefail
trap 'touch FAILED' ERR
old=/workspace/cnr-experiments/turin-bus-only-v1
export PYTHONPATH=$PWD:/workspace/cnr-experiments/difftraj-nov11 OMP_NUM_THREADS=1 DIFFTRAJ_GLOBAL_BATCH=56
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
pids=()
for spec in 42:0,1 43:2,3; do
(
 seed=${spec%:*}; export PORTO_TRAIN_SEED=$seed CUDA_VISIBLE_DEVICES=${spec#*:}
 mkdir "$seed" "old-$seed"; cd "old-$seed"
 ln -s "$old"/{observations,prepared}.npz "$old/$seed"/{best,checkpoint}.pt .
 "$py" ../porto.py score --best --development >dev.log 2>&1
 cd "../$seed"; ln -s "$old"/{observations,prepared}.npz .
 "$py" -m torch.distributed.run --standalone --nproc_per_node=2 ../porto.py train --large-microbatch --stabilized >train.log 2>&1
 "$py" ../porto.py score --best --development >dev.log 2>&1
 "$py" ../porto.py score --best --all-holdout >score.log 2>&1
 mv geolife-final.json turin-final.json
) & pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
touch COMPLETE
