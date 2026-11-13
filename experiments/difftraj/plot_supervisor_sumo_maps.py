"""Plot saved conditional DiffTraj draws and observed journeys on the MoST map."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


CASES = {
    'supplied': ('sumo-batch32-v1', 'Supplied MoST observations', 'sumo-final.json'),
    'taxi-10k': ('taxi-repro-debug-v1/10k', 'Simulated taxis · 10,000 training trips', 'development.json'),
    'taxi-20k': ('taxi-repro-debug-v1/20k', 'Simulated taxis · 20,000 training trips', 'development.json'),
}
SEEDS = (42, 43)
BLUE = '#006aa3'
ORANGE = '#bf4a13'
INK = '#253641'


def map_shapes(net):
    roads = []
    for edge in ElementTree.parse(net).getroot().findall('edge'):
        lane = edge.find('lane')
        if lane is not None and edge.get('function') != 'internal':
            path = [tuple(map(float, pair.split(',')[:2])) for pair in lane.get('shape').split()]
            if len(path) > 1:
                roads.append(np.asarray(path)/1000)
    return roads


def arc(xy):
    return float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()/1000)


def load_cohort(run, saved_score):
    with np.load(run/'observations.npz') as obs:
        delta = obs['delta']
        ends = np.r_[0, np.cumsum(obs['length'])]
        trip_ids = obs['taxi']
        unit = float(obs['planar_unit_m'])
    result = {}
    for seed in SEEDS:
        with np.load(run/str(seed)/'draws.npz') as draw:
            ids, xy = draw['ids'], draw['xy']
        if seed == SEEDS[0]:
            first_ids = ids.copy()
        else:
            np.testing.assert_array_equal(first_ids, ids)
        digest = hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
        score = json.loads((run/str(seed)/saved_score).read_text())
        assert len(np.unique(ids)) == len(ids) == score['requests'] and digest == score['request_index_sha256']
        result[seed] = []
        for index in [0,len(ids)//2,len(ids)-1]:
            i = ids[index]
            observed = delta[ends[i]:ends[i+1]].cumsum(0) * unit
            result[seed].append(dict(order=index, key=str(trip_ids[i]), observed=observed, generated=xy[index]))
    return result


def class_names(csv_file, keys):
    labels = {}
    with csv_file.open() as f:
        for row in csv.DictReader(f):
            if row['vehicle_id'] in keys:
                labels[row['vehicle_id']] = row['class2']
                if len(labels) == len(keys):
                    break
    assert set(labels) == keys
    return {k:v.replace('private_car','car').replace('bycicle','bicycle') for k,v in labels.items()}


def map_extent(xy):
    low, high = xy.min(0)/1000, xy.max(0)/1000
    center = (low+high)/2
    width = max(.8, (high-low).max()*1.16)
    return center-width/2, center+width/2


def render(tag, cohort, data, roads, output, labels, matching_extents):
    fig, axes = plt.subplots(2, 3, figsize=(10.4,7.7))
    fig.subplots_adjust(left=.088,right=.986,bottom=.12,top=.853,wspace=.18,hspace=.52)
    fig.patch.set_facecolor('white')
    fig.text(.057,.976,cohort,fontsize=17,weight='semibold',va='top',color=INK)
    handles = [Line2D([],[],color=BLUE,lw=2.3,label='Observed'),
               Line2D([],[],color=ORANGE,ls='--',lw=1.9,label='Generated'),
               Line2D([],[],color=INK,ls='',marker='o',markerfacecolor='white',label='Start'),
               Line2D([],[],color=INK,ls='',marker='X',label='End')]
    fig.legend(handles=handles,ncol=4,loc='upper left',bbox_to_anchor=(.051,.935),
               frameon=False,handlelength=2.2,columnspacing=2,borderaxespad=0,fontsize=10)
    stats = []
    for i,seed in enumerate(SEEDS):
        for j,(ax,r) in enumerate(zip(axes[i],data[seed])):
            low, high = matching_extents[i][j]
            ax.add_collection(LineCollection(roads,colors='#c8d0d2',linewidths=.51,zorder=0,
                                             rasterized=True))
            for path,color,dash in [(r['observed'],BLUE,'-'),(r['generated'],ORANGE,'--')]:
                xy=path/1000
                ax.plot(*xy.T,lw=1.75 if dash=='-' else 1.50,color=color,ls=dash,zorder=2)
                ax.scatter(*xy[0],s=34,marker='o',fc='white',ec=color,lw=1.45,zorder=4)
                ax.scatter(*xy[-1],s=50,marker='X',fc=color,ec='white',lw=.5,zorder=5)
            off = np.linalg.norm((r['observed'][[0,-1]] - r['generated'][[0,-1]]),axis=1)
            label = labels[r['key']] if labels else 'taxi'
            ax.set_title(f"Request {r['order']+1} · {label}",fontsize=10.6,loc='left',
                         pad=6,weight='semibold',color=INK)
            ax.text(0,-.135,f"Start error {off[0]:,.0f} m   ·   End error {off[1]:,.0f} m",
                    transform=ax.transAxes,fontsize=8.2,ha='left',va='top',color=INK)
            ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),aspect='equal')
            for spine in ax.spines.values():
                spine.set_color('#9aa8ae'); spine.set_linewidth(.55)
            ax.tick_params(labelsize=7.7,pad=2,color='#9aa8ae',length=2)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if j==0:
                ax.set_ylabel(f'Seed {seed}  ·  North (km)',fontsize=9,color=INK,weight='semibold')
            scalars = dict(request=r['order']+1,model_seed=seed,source_class=label,
                           observed_km=arc(r['observed']), generated_200point_km=arc(r['generated']),
                           start_offset_m=float(off[0]),end_offset_m=float(off[1]))
            stats.append(scalars)
    fig.text(.057,.017,'Axes show local east and north in kilometres.'+
             ('' if tag=='supplied' else ' The 10k and 20k panels use identical limits for each request and seed.'),
             ha='left',va='bottom',fontsize=8.3,color='#50616b')
    fig.savefig(output/f'maps-{tag}.png',dpi=190)
    plt.close(fig)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs','network','original_csv','output'):
        p.add_argument(name,type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    net_sha = hashlib.sha256(args.network.read_bytes()).hexdigest()
    assert net_sha == 'a06b5c0765fb2d49d4ad33cf54048f15f2ff1592b2b3d0180dbeec4feb73bfc0'
    roads = map_shapes(args.network)
    all_data = {key:load_cohort(args.runs/folder,score) for key,(folder,_,score) in CASES.items()}
    for seed in SEEDS:
        for a,b in zip(all_data['taxi-10k'][seed],all_data['taxi-20k'][seed]):
            assert a['key']==b['key'] and a['order']==b['order']
            np.testing.assert_equal(a['observed'],b['observed'])
    labels = class_names(args.original_csv, {r['key'] for r in all_data['supplied'][42]})
    output = dict(selection='First, middle and last requests in the saved order. Same observed requests in both seeds; 10k and 20k share the same observed taxi journeys.',
                  network_sha256=net_sha, groups={})
    for tag,(_,name,_) in CASES.items():
        peers = ['supplied'] if tag=='supplied' else ['taxi-10k','taxi-20k']
        extents = [[map_extent(np.vstack([all_data[k][seed][j][t] for k in peers for t in ['observed','generated']]))
                    for j in range(3)] for seed in SEEDS]
        output['groups'][tag] = render(tag,name,all_data[tag],roads,args.output,
                                       labels if tag=='supplied' else None,extents)
    (args.output.parent/'map-examples.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    main()
