"""Rebuild the SUMO supervisor report's metrics and training curves from fitted runs.

Use --run-root /workspace/cnr-experiments/taxi-repro-debug-v1 for the original
10k/20k training logs. With no arguments, redraw from the committed epoch CSV.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / 'docs/results'
OUTPUT = RESULTS / 'sumo-taxi-supervisor-report'
DATASETS = {
    'supplied': ('Supplied vehicle traces', 1478, 32, 2),
    '10k': ('Simulated taxis · 10,000', 10000, 220, 1),
    '20k': ('Simulated taxis · 20,000', 20000, 440, 1),
}
SEEDS = (42, 43)
SCORE_KEYS = ('density_js', 'od_js', 'length_js', 'top20_cell_f1')


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_original(score):
    return {k: score['trip_js' if k == 'od_js' else k] for k in SCORE_KEYS}


def score_audit(audit, grid):
    source = audit[grid]
    return {k: source['length_js'] if k == 'length_js' else source['full'][k] for k in SCORE_KEYS}


def rebuild(run_root, output):
    metrics = {
        'method': 'Released conditional DiffTraj coordinate network; epoch-200 ordinary parameters.',
        'published_experiments': {
            'metric_order': ['density', 'trip', 'length', 'pattern'],
            'pattern_definition': 'F1 between the top-n visited cells. Appendix B.2 leaves n unspecified.',
            'Chengdu': {'table': 1, 'scores': [.0055, .0154, .0169, .823],
                        'source': 'https://arxiv.org/html/2304.11582#S5.T1'},
            'Xian': {'table': 1, 'scores': [.0126, .0165, .0203, .764],
                     'source': 'https://arxiv.org/html/2304.11582#S5.T1'},
            'Porto': {'table': 5, 'scores': [.0087, .0132, .0242, .847],
                      'source': 'https://arxiv.org/html/2304.11582#A3.SS1'},
        },
        'primary_scoring': 'Supplied set uses its training bounds and length bins. Generated-taxi populations use the bounds and length bins of the original 10,000-trip training subset. Every taxi row uses the same ordered 1,000 reference tracks.',
        'training': {'position_channels': 2, 'model_positions': 200, 'diffusion_levels': 500,
                     'beta_start': .0001, 'beta_end': .05, 'network': 'Guide_UNet',
                     'noise_prediction': 'conditional + 3 * (conditional - zero-header branch)',
                     'objective': 'mean squared error over sampled Gaussian noise and both coordinate channels at 200 positions',
                     'optimizer': 'AdamW', 'constant_learning_rate': .0002, 'epochs': 200,
                     'checkpoint': 'final ordinary parameters', 'sampling': '100 reverse steps, eta=0, eval mode, common noise seed 42'},
        'runs': [],
    }
    curve = []
    reference_hashes = set()
    for arm, (_, n, batch, world) in DATASETS.items():
        steps = math.ceil(n / batch)
        for seed in SEEDS:
            folder = ((RESULTS/'sumo-batch32') if arm == 'supplied' else (run_root/arm)) / str(seed)
            log = folder/'epochs.jsonl'
            records = [json.loads(t) for t in log.read_text().splitlines()]
            assert [r['epoch'] for r in records] == list(range(1,201))
            assert all(r['global_batch'] == batch and r['train_seed'] == seed for r in records)
            losses = [r['rank0_mean_noise_loss'] for r in records]
            assert all(math.isfinite(x) and x > 0 for x in losses)
            for r in records:
                curve.append(dict(dataset=arm,seed=seed,epoch=r['epoch'],updates=r['epoch']*steps,
                                  training_noise_mse_rank0=r['rank0_mean_noise_loss']))
            score = read_json(folder / ('sumo-final.json' if arm == 'supplied' else 'development.json'))
            assert score['requests'] == (177 if arm == 'supplied' else 1000) and score['epoch'] == 200
            native = score_original(score)
            primary = native
            if arm != 'supplied':
                aud = read_json(RESULTS/'taxi-reproduction-debug/diagnostics'/arm/str(seed)/'od-diagnostic.json')
                assert all(math.isclose(value,score_audit(aud,'native')[k],abs_tol=1.e-10) for k,value in native.items())
                reference_hashes.add(aud['reference_sha256'])
                primary = score_audit(aud,'common')
            else:
                metrics['supplied_real_development_vs_reserve_177_each'] = dict(zip(SCORE_KEYS,score['real_real']))
            metrics['runs'].append(dict(dataset=arm,seed=seed,train=n,batch=batch,gpus_per_fit=world,
                optimizer_updates=steps*200,eval_set='reserved' if arm == 'supplied' else 'development',
                n_eval=score['requests'],metrics=primary,native_metrics=native,
                observed_median_km=score['reference_median_m']/1000,generated_median_km=score['generated_median_m']/1000,
                length_w1_km=score['length_w1_m']/1000,training_loss_epoch1=losses[0],
                training_loss_epoch200=losses[-1],original_epoch_log_sha256=digest(log)))
    assert len(reference_hashes) == 1
    metrics['shared_taxi_observations_sha256'] = reference_hashes.pop()
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    with (output/'training-loss.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=curve[0],lineterminator='\n')
        writer.writeheader()
        writer.writerows(curve)


def plot(output):
    with (output/'training-loss.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    colors = {42:'#176785',43:'#be4c29'}
    fig, axes = plt.subplots(1,3,figsize=(10.8,3.1),sharex=True,sharey=True)
    for ax,(name,(label,n,batch,_)) in zip(axes,DATASETS.items()):
        for seed in SEEDS:
            group = [r for r in rows if r['dataset']==name and int(r['seed'])==seed]
            assert len(group)==200
            ax.plot([int(r['epoch']) for r in group],[float(r['training_noise_mse_rank0']) for r in group],
                    color=colors[seed],ls='-' if seed==42 else '--',lw=1.4)
        ax.set_yscale('log')
        ax.set_ylim(.007,1.63)
        ax.set_xlim(0,200)
        ax.set_xticks([1,50,100,150,200])
        ax.set_yticks([.01,.03,.1,.3,1],['.01','.03','.1','.3','1'])
        ax.set_yticks([],minor=True)
        ax.grid(axis='y',alpha=.23,lw=.65)
        ax.set_axisbelow(True)
        ax.set_title(label if name!='supplied' else 'Supplied vehicles · 1,478',fontsize=11,pad=8,loc='left',weight='semibold')
        ax.set_xlabel('Epoch',fontsize=9)
        ax.tick_params(labelsize=9,length=3)
        for side in ('top','right'):
            ax.spines[side].set_visible(False)
        for side in ('left','bottom'):
            ax.spines[side].set_color('#73818a')
    axes[0].set_ylabel('Training noise MSE (log scale)',fontsize=9)
    axes[2].annotate('Epoch 134: 1.06',xy=(134,1.063),xytext=(72,.43),fontsize=8,
                     arrowprops={'arrowstyle':'->','lw':.8,'color':'#934226'},color='#934226')
    fig.legend([Line2D([0],[0],lw=1.6,color=colors[42]),Line2D([0],[0],lw=1.6,color=colors[43],ls='--')],
               ['Seed 42','Seed 43'],loc='lower center',bbox_to_anchor=(.5,-.005),ncol=2,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.075,right=.987,top=.9,bottom=.27,wspace=.20)
    fig.savefig(output/'figures/training-loss.png',dpi=220,bbox_inches='tight',pad_inches=.05)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,help='Root containing the completed 10k and 20k runs')
    p.add_argument('--output',type=Path,default=OUTPUT)
    args = p.parse_args()
    (args.output/'figures').mkdir(exist_ok=True,parents=True)
    if args.run_root is not None:
        rebuild(args.run_root,args.output)
    plot(args.output)
