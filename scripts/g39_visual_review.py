"""Fixed-rule complete-journey visual audit of finished G39 main outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CLASSES=['bus','motorcycle','private_car','bycicle','taxi_or_uber']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def record(pool,index):
    a,b=pool['offsets'][index:index+2];track=pool['txy'][a:b].copy()
    track[:,0]-=track[0,0]
    return track


def draw(root,out):
    closed=json.loads((root/'complete.json').read_text());assert closed['complete'] and closed['evaluation_cells']==112
    episode=root/'390201';reference=load(episode/'reference/references.npz')
    refmap={str(x):i for i,x in enumerate(reference['ids'])}
    pools={};selection={};hashes={};rows={}
    for arm in ('B','SV'):
        p=episode/'generated'/f'{arm}-42';receipt=json.loads((p/'complete.json').read_text())
        for name in ['samples.npz','selections.json','candidates.json']:assert sha(p/name)==receipt['artifacts'][name]
        pools[arm]=load(p/'samples.npz')
        selection[arm]={r['request_id']:r for r in json.loads((p/'selections.json').read_text())['rows'] if r['method']=='R'}
        rows[arm]=json.loads((p/'candidates.json').read_text())['rows'];hashes[arm]=sha(p/'complete.json')
    out.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(5,3,figsize=(16,15),layout='constrained')
    colors={'SUMO reference':'#777777','Base diffusion':'#1876b8','Dual-GAF alignment':'#e47b21'}
    chosen=[]
    for c,name in enumerate(CLASSES):
        eligible=sorted(rid for rid,r in selection['B'].items() if r['label']==c and rid in refmap and rid in selection['SV'])
        assert eligible
        rid=eligible[0];tracks={'SUMO reference':record(reference,refmap[rid]),
            'Base diffusion':record(pools['B'],selection['B'][rid]['index']),
            'Dual-GAF alignment':record(pools['SV'],selection['SV'][rid]['index'])}
        for arm in ('B','SV'):
            rr=rows[arm][selection[arm][rid]['index']]
            assert rr['request_id']==rid and rr['label']==c
        assert np.array_equal(tracks['Base diffusion'][:,0],tracks['Dual-GAF alignment'][:,0])
        origin=tracks['SUMO reference'][0,1:]
        for label,track in tracks.items():
            t=track[:,0];xy=track[:,1:];distance=np.linalg.norm(np.diff(xy,axis=0),axis=1)
            progress=np.r_[0,np.cumsum(distance)];progress/=max(progress[-1],1e-12)
            color=colors[label];width=1.5 if label=='SUMO reference' else 1.1
            axes[c,0].plot(*(xy-origin).T,color=color,lw=width,alpha=.85,label=label)
            axes[c,1].plot(t,progress,color=color,lw=width,alpha=.9,label=f'{label}: {t[-1]:.1f}s')
            midpoint=(t[1:]+t[:-1])/2;speed=distance/np.diff(t);mask=midpoint<=60
            axes[c,2].plot(midpoint[mask],speed[mask],color=color,lw=width,alpha=.85)
        axes[c,0].set_title(name.replace('_',' '));axes[c,0].set_aspect('equal',adjustable='datalim')
        axes[c,0].set_xlabel('East offset (m)');axes[c,0].set_ylabel('North offset (m)')
        axes[c,1].set_xlabel('Elapsed time (s)');axes[c,1].set_ylabel('Normalized travelled distance')
        axes[c,1].legend(fontsize=7,loc='lower right')
        axes[c,2].set_xlabel('Elapsed time (s), first60s');axes[c,2].set_ylabel('Native coordinate speed (m/s)')
        axes[c,2].set_xlim(0,60)
        for ax in axes[c]:ax.grid(alpha=.18)
        chosen.append({'class':name,'request_sha256':hashlib.sha256(rid.encode()).hexdigest(),
            'duration_s':{label:float(t[-1,0]) for label,t in tracks.items()}})
    fig.suptitle('G39 completed-main visual audit | fixed first eligible request per class\n'
        'Episode390201, seed42, first direct draw; same generated route/duration; one SUMO realization',fontsize=13)
    axes[0,0].legend(fontsize=7)
    fig.savefig(out/'complete-journeys.png',dpi=150);plt.close(fig)
    receipt={'complete':True,'main_receipt_sha256':sha(root/'complete.json'),'source_receipts':hashes,
        'reference_receipt_sha256':sha(episode/'reference/complete.json'),'selection':chosen,
        'rule':'First lexicographic jointly supported request per class, fixed episode390201/seed42/K1; no metric-based example selection.',
        'display':'All five classes. Full geometry/time-progress; unsmoothed native coordinate speed in the first60s.',
        'limits':['Illustration, not representative or statistical inference.',
                  'Reference route and duration may differ from generated alternatives; no future route/duration was supplied.',
                  'XY panels show trajectories, not a full map-legality audit.'],
        'source_sha256':sha(Path(__file__)),'image_sha256':sha(out/'complete-journeys.png')}
    (out/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'complete':True,'classes':5,'figure':'complete-journeys.png'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();draw(a.root,a.output)
