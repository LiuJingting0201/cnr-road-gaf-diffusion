set -euo pipefail
trap 'touch FAILED' ERR
export DIFFTRAJ_GLOBAL_BATCH=220
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
"$py" taxi10k_pack.py
bash run_sumo_original.sh
