# DiffTraj experiments

Clone the [upstream repository](https://github.com/Yasoz/DiffTraj) and check out revision 27d25a9eb4cb94bcbe76068d5c8825f713a7ac2c. The scripts import its utils package. Use a recent PyTorch environment with NumPy, pandas, SciPy, Matplotlib and tqdm. Run from a private experiment directory for data, weights and generated samples.

    export PYTHONPATH=/absolute/path/to/upstream
    python /path/to/experiments/difftraj/checkpoint.py /absolute/path/to/upstream released
    python /path/to/experiments/difftraj/prepare.py /path/to/midimost_with_class2.csv
    CUDA_VISIBLE_DEVICES=0 python /path/to/experiments/difftraj/pilot.py B 2000

Repeat the last command for V (time-varying speed supervision), C (class) and VC (both), one GPU per process. The training script saves motion and class weights locally then evaluates matched draws. Pass --evaluate after the update count to evaluate the saved weights. B 8000 repeats baseline training from the same initialization with an increased budget. The diagnosis and reference scripts use the current run directory.

The notebook replay fixes an upstream module-name typo, uses the released PyTorch state dictionary and retains notebook model mode. The CPU smoke check uses 16 fixed public request headers and evaluation mode. Small-sample local diagnostics use natural-log Jensen–Shannon divergence and Wasserstein distance. Decision records and measured results are in docs/DIFFTRAJ_STUDY.md.

For the second training seed set DIFFTRAJ_TRAINING_SEED=43 in a separate output directory. The 8,000-update evaluation uses the same 128 sampled requests for both training seeds. The optional --evaluate --save-draws combination saves a private local file used by figure.py and duration.py.

For the Porto study, obtain UCI dataset 339 and run porto_audit.py downloaded.zip observations.npz. Run porto.py prepare, then torchrun --standalone --nproc-per-node=4 porto.py train. Run porto.py score to evaluate the terminal checkpoint. Inputs, model weights and raw/generated coordinate arrays stay on the experiment machine.
