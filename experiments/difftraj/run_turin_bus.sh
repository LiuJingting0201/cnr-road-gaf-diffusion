set -euo pipefail
trap 'touch FAILED' ERR
export PYTHONPATH=$PWD:/workspace/cnr-experiments/difftraj-nov11 OMP_NUM_THREADS=1
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
"$py" porto.py prepare > split.json
pids=()
for spec in 42:0,1 43:2,3; do
(
 seed=${spec%:*}; export PORTO_TRAIN_SEED=$seed CUDA_VISIBLE_DEVICES=${spec#*:}
 mkdir "$seed"; cd "$seed"; ln -s ../observations.npz ../prepared.npz .
 "$py" -m torch.distributed.run --standalone --nproc_per_node=2 ../porto.py train --stabilized --large-microbatch > train.log 2>&1
 "$py" ../porto.py score --all-holdout --best --save-draws > score.log 2>&1
 mv geolife-final.json turin-final.json
) & pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
touch COMPLETE
