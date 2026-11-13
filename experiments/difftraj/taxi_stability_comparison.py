"""Compare saved 20k-taxi draws before and after the paired optimizer intervention."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_supervisor_sumo_maps import BLUE, INK, ORANGE, SEEDS, map_extent, map_shapes
from taxi_od_diagnostic import audit

MEASURES = ('density_js', 'od_js', 'length_js', 'hotspot_f1', 'length_w1_km',
            'generated_median_km', 'origin_m', 'destination_m', 'joint_hit')
REQUESTS = [0, 500, 999]


def read_json(path):
    return json.loads(path.read_text())


def extract_score(run, common, diagnostic, common_basis):
    saved = read_json(run / 'development.json')
    if diagnostic.exists():
        diag = read_json(diagnostic)
    else:
        diag = audit(run, common, run / 'development.json')
    report = diag['common']
    full = report['full']
    assert saved['requests'] == diag['n'] == 1000
    assert report['bin_edges_m'] == common_basis['length_bins']
    for side in ['lo', 'hi']:
        np.testing.assert_equal(full[side], common_basis[side])
    values = dict(density_js=full['density_js'], od_js=full['od_js'],
                  length_js=report['length_js'], hotspot_f1=full['top20_cell_f1'],
                  length_w1_km=diag['length_w1_km'],
                  observed_median_km=diag['q10_q50_q90_real_length_km'][1],
                  generated_median_km=diag['q10_q50_q90_generated_length_km'][1],
                  origin_m=diag['q10_q50_q90_origin_error_m'][1],
                  destination_m=diag['q10_q50_q90_destination_error_m'][1],
                  joint_hit=full['joint_hit'])
    return dict(epoch=saved['epoch'], requests=diag['n'], **values), diag['reference_sha256'], saved['request_index_sha256']


def map_pair(paths, observed, net, seed, output):
    time_labels = ['Original training', 'Clipping + cosine']
    fig, axs = plt.subplots(2,3,figsize=(11,7.5))
    fig.subplots_adjust(left=.085,right=.98,bottom=.13,top=.83,wspace=.22,hspace=.55)
    fig.text(.055,.965, f'20,000 simulated taxis  ·  training seed {seed}', color=INK,
             weight='semibold', fontsize=17, va='top')
    fig.legend(handles=[Line2D([],[],color=BLUE,lw=2.0,label='Observed'),
                        Line2D([],[],color=ORANGE,ls='--',lw=1.8,label='Generated'),
                        Line2D([],[],color=INK,ls='',marker='o',mfc='white',label='Start'),
                        Line2D([],[],color=INK,ls='',marker='X',label='End')],
               loc='upper left', bbox_to_anchor=(.051,.935),ncol=4,frameon=False)
    examples = []
    for j, i in enumerate(REQUESTS):
        ref = observed[j]
        lo, hi = map_extent(np.vstack([ref, paths[0][i], paths[1][i]]))
        for row, (ax, xy) in enumerate(zip(axs[:,j], [paths[0][i], paths[1][i]])):
            ax.add_collection(LineCollection(net, colors='#c8d0d2',linewidths=.51,zorder=0))
            for line, color, ls in [(ref, BLUE, '-'), (xy, ORANGE, '--')]:
                points = line / 1000.
                ax.plot(*points.T, color=color, lw=1.75, ls=ls, zorder=2)
                ax.scatter(*points[0],s=33,marker='o',fc='white',ec=color,lw=1.4,zorder=4)
                ax.scatter(*points[-1],s=46,marker='X',fc=color,ec='white',lw=.5,zorder=5)
            d = np.linalg.norm(xy[[0,-1]]-ref[[0,-1]], axis=1)
            ax.set(aspect='equal', xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
            ax.set_title(f'Request {i+1:,}',fontsize=11,loc='left',color=INK,weight='semibold',pad=5)
            ax.text(0,-.13,f'Start {d[0]:,.0f} m   ·   End {d[1]:,.0f} m',transform=ax.transAxes,
                    fontsize=8.5,ha='left',va='top',color=INK)
            if j==0:
                ax.set_ylabel(time_labels[row]+'\nNorth (km)',fontsize=10, weight='semibold',color=INK)
            ax.tick_params(labelsize=8, length=2, color='#87959a')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            for s in ax.spines.values():
                s.set_color('#93a6ab'); s.set_linewidth(.55)
            examples.append(dict(seed=seed,model=time_labels[row],request=i+1,
                                 start_error_m=float(d[0]),end_error_m=float(d[1]),
                                 east_window_km=[float(lo[0]),float(hi[0])],
                                 north_window_km=[float(lo[1]),float(hi[1])]))
    fig.text(.055,.022,'Final epoch 200. Each column uses the same request, initial noise and map bounds in both rows. Axes are local kilometres.',
             fontsize=8.6,color='#50616b')
    fig.savefig(output/f'stability-maps-{seed}.png',dpi=180)
    plt.close(fig)
    return examples


def plot_training(old, new, output):
    fig, axes = plt.subplots(1,2,figsize=(10.5,3.5),sharex=True,sharey=True)
    curves, summaries = [], {}
    for ax,seed in zip(axes, SEEDS):
        summaries[seed] = {}
        for folder, name, color, style in [(old,'Original',ORANGE,'--'), (new,'Clipping + cosine',BLUE,'-')]:
            vals = [json.loads(x) for x in (folder/str(seed)/'epochs.jsonl').read_text().splitlines()]
            assert [r['epoch'] for r in vals] == list(range(1,201))
            xx = [r['epoch'] for r in vals]
            yy = [r['rank0_mean_noise_loss'] for r in vals]
            ax.plot(xx,yy,color=color,linestyle=style,label=name,lw=1.3)
            worst = max(vals[49:],key=lambda v:v['rank0_mean_noise_loss'])
            summaries[seed][name] = dict(final_train_mse=yy[-1],peak_mse_after_epoch49=worst['rank0_mean_noise_loss'],
                                          peak_epoch_after_epoch49=worst['epoch'])
            for r in vals:
                curves.append(dict(seed=seed,model=name,epoch=r['epoch'],noise_mse=r['rank0_mean_noise_loss'],
                                   val_noise_mse=r.get('val_loss',''), lr=r.get('lr',.0002),
                                   clipped_steps=r.get('clipped_steps','')))
        ax.set(xlim=(1,200),xlabel='Epoch',yscale='log')
        ax.set_title(f'Training seed {seed}',loc='left',color=INK,fontsize=13)
        ax.grid(axis='y',alpha=.22)
    axes[0].set_ylabel('Training noise MSE (log scale)')
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,frameon=False,loc='upper center')
    fig.tight_layout(rect=(0,0,1,.88))
    fig.savefig(output/'stability-loss.png',dpi=180)
    plt.close(fig)
    with (output.parent/'stability-training.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(curves[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(curves)
    return summaries


def main(args):
    plots = args.output/'figures'; plots.mkdir(parents=True,exist_ok=True)
    net = map_shapes(args.network)
    assert hashlib.sha256(args.network.read_bytes()).hexdigest() == 'a06b5c0765fb2d49d4ad33cf54048f15f2ff1592b2b3d0180dbeec4feb73bfc0'
    for name in ['observations.npz','prepared.npz']:
        assert (args.new/name).samefile(args.old/name), f'Changed training pack: {name}'
    with np.load(args.old/'observations.npz') as raw, np.load(args.old/'42/draws.npz') as draws:
        pos = np.r_[0, raw['length'].cumsum()]; unit=float(raw['planar_unit_m']); delta=raw['delta']
        ids = draws['ids'].copy()
        assert len(ids) == len(set(ids)) == 1000
        ref = [delta[pos[ids[i]]:pos[ids[i]+1]].cumsum(0)*unit for i in REQUESTS]
    from taxi_od_diagnostic import measurement_basis
    grids, bins = measurement_basis(args.common)
    common_basis = dict(lo=grids['full'][0].tolist(),hi=grids['full'][1].tolist(),length_bins=bins.tolist())
    result = dict(scoring='1,000 matched development requests scored using common 10k-training grid and length bins',
                  primary='epoch 200 versus epoch 200',supplement='Minimum validation noise MSE, scored on the same development population',
                  request_index_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
                  simulation_sampling_seed=42, scores={}, maps=[])
    refs = set()
    for seed in SEEDS:
        paths = []
        result['scores'][seed] = {}
        for name, run in [('original', args.old/str(seed)),('stable', args.new/str(seed)/'final'),('best', args.new/str(seed)/'best')]:
            vals, refhash, idhash = extract_score(run, args.common, run/'od-diagnostic.json', common_basis)
            refs.add(refhash)
            assert idhash == result['request_index_sha256']
            result['scores'][seed][name] = vals
            with np.load(run/'draws.npz') as draw:
                np.testing.assert_array_equal(ids,draw['ids'])
                if name != 'best':
                    assert vals['epoch'] == 200
                    paths.append(draw['xy'].copy())
        result['maps'].extend(map_pair(paths,ref,net,seed,plots))
        result['scores'][seed]['stable_minus_original'] = {
            k:result['scores'][seed]['stable'][k]-result['scores'][seed]['original'][k] for k in MEASURES}
    assert len(refs) == 1
    result['reference_sha256'] = refs.pop()
    result['training'] = plot_training(args.old,args.new,plots)
    tmp = args.output/'stability-comparison.tmp'
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    tmp.replace(args.output/'stability-comparison.json')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('old','new','common','network','output'):
        p.add_argument(key,type=Path)
    main(p.parse_args())
