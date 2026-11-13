"""Plot the first saved request, observed and generated, in each city's road context."""
import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np


def inside(line, box):
    return (line[:,0].min() <= box[1] and line[:,0].max() >= box[0]
            and line[:,1].min() <= box[3] and line[:,1].max() >= box[2])


def roads(city, path, box):
    lines = []
    if city == 'SUMO':
        for edge in ET.parse(path).getroot().findall('edge'):
            lane = edge.find('lane')
            if lane is None or edge.get('function') == 'internal':
                continue
            line = np.array([tuple(map(float,p.split(',')[:2])) for p in lane.get('shape').split()])
            if len(line)>1 and inside(line,box):
                lines.append(line)
    else:
        import osmium
        class Ways(osmium.SimpleHandler):
            def way(self,w):
                highway = w.tags.get('highway')
                if not highway or highway in {'path','steps','footway','pedestrian','cycleway','bridleway','construction','proposed','track','corridor'}:
                    return
                try:
                    line = np.array([(n.lon,n.lat) for n in w.nodes])
                except osmium.InvalidLocationError:
                    return
                if len(line)>1 and inside(line,box):
                    lines.append(line)
        Ways().apply_file(str(path),locations=True,idx='flex_mem')
    return lines


def main(args):
    with args.measures.open() as f:
        first = {r['city']:r for r in csv.DictReader(f) if r['seed']=='42' and r['draw_index']=='0'}
    fig,axes = plt.subplots(1,2,figsize=(10.5,5.4))
    for ax,city,folder,mapfile in zip(axes,('SUMO','Porto'),('10k','porto'),(args.sumo_map,args.porto_map)):
        run = args.source/folder
        with np.load(run/'42/draws.npz') as d, np.load(run/'observations.npz') as raw:
            index=int(d['ids'][0])
            assert index == int(first[city]['pack_index'])
            ends=np.r_[0,raw['length'].cumsum()]
            unit = float(raw['planar_unit_m']) if city=='SUMO' else 1.e-6
            obs=raw['delta'][ends[index]:ends[index+1]].cumsum(0)*unit
            gen=d['xy'][0]
        origin = (obs[0]+gen[0])/2
        factor = np.ones(2)/1000 if city=='SUMO' else np.array([111.195*np.cos(np.radians(origin[1])),111.195])
        def local(x):
            return (x-origin)*factor
        point=local(np.vstack([obs,gen]))
        center=(point.min(0)+point.max(0))/2
        span=max(np.ptp(point,axis=0).max()*1.10,.6)
        low,high=center-span/2,center+span/2
        lower,upper=low/factor+origin,high/factor+origin
        box=(lower[0],upper[0],lower[1],upper[1])
        lines=roads(city,mapfile,box)
        assert lines
        ax.add_collection(LineCollection([local(line) for line in lines],color='#c4ced3',linewidths=.45))
        for path,color,dash in [(obs,'#18587a','-'),(gen,'#c34425','--')]:
            xy=local(path)
            ax.plot(*xy.T,color=color,ls=dash,lw=1.9,alpha=.94)
            ax.scatter(*xy[0],s=62,facecolor='white',edgecolors=color,linewidth=1.8,zorder=5)
            ax.scatter(*xy[-1],s=55,color=color,marker='x',linewidth=2.1,zorder=6)
        ax.set_aspect('equal'); ax.set_xlim(low[0],high[0]); ax.set_ylim(low[1],high[1])
        ax.set_xlabel('East (km)'); ax.set_ylabel('North (km)')
        r=first[city]
        ax.set_title(f"{city}   |   Start {float(r['origin_error_m']):,.0f} m, end {float(r['destination_error_m']):,.0f} m",fontsize=11,loc='left')
    fig.legend(handles=[Line2D([],[],color='#18587a',label='Observed'),Line2D([],[],color='#c34425',ls='--',label='Generated'),
                        Line2D([],[],color='#333',ls='',marker='o',markerfacecolor='white',label='Start'),
                        Line2D([],[],color='#333',ls='',marker='x',label='End')],loc='upper center',bbox_to_anchor=(.5,.972),ncol=4,frameon=False)
    fig.suptitle('First saved request in each city, seed 42',fontsize=15)
    fig.subplots_adjust(left=.08,right=.985,bottom=.15,top=.80,wspace=.24)
    fig.text(.08,.025,'Roads: MoST (SUMO) and © OpenStreetMap contributors (Porto). Positions plotted before map matching.',fontsize=8,color='#46545b')
    fig.savefig(args.output,dpi=170,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['source','sumo_map','porto_map','measures','output']:
        parser.add_argument(key,type=Path)
    main(parser.parse_args())
