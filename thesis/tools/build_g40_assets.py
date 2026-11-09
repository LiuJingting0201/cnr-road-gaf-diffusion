"""Publication figures/tables from verified aggregates, with every arm retained."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ARMS=('B','F-SV','J-S','J-V','J-SV','J-RAW','J-SH','C')
CLASSES=('bus','motorcycle','private_car','bycicle','taxi_or_uber')
NAMES=('Bus','Motorcycle','Private car','Bicycle','Taxi / ride-hailing')
PRIMARY=('selected_native_speed_w1','selected_native_acceleration_w1')
COLORS=['#73818a','#8c78a3','#8b4ea3','#218c7e','#c5653a','#527bb0','#a18b66','#737373']


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);figdir=a.output/'figures';tabdir=a.output/'tables';figdir.mkdir();tabdir.mkdir()
    sources={};made=[];tables=[]
    def read(name):
        p=a.evidence/name;sources[name]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
    r=read('g40-verified.json');assert r['complete'] and r['all_stages']['complete']
    g38=read('g38.json');factor=read('g39-factorial.json');utility39=read('g39-utility.json');audit=read('g40-audit.json')
    report=r['report'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':11})
    def finish(fig,name):
        for ext in ('pdf','png'):fig.savefig(figdir/(name+'.'+ext),bbox_inches='tight',dpi=160)
        plt.close(fig);made.append(name)
    def table(name,caption,label,cols,rows):
        text='\\tab{'+caption+'}{'+label+'}{\\begin{tabular}{'+cols+'}\\toprule\n'
        text+='\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n\\bottomrule\\end{tabular}}\n'
        (tabdir/(name+'.tex')).write_text(text);tables.append(name)
    # An implemented architecture, with distinct training and prospective paths.
    fig,ax=plt.subplots(figsize=(10.8,7.1));ax.set(xlim=(0,12),ylim=(0,8.4));ax.axis('off')
    def box(x,y,w,h,label,color='#edf1f4'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc=color,ec='#71818d',lw=.8))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
    def arrow(x,y,u,v):ax.annotate('',xy=(u,v),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':1.1,'color':'#455b6b'})
    ax.text(.1,8.1,'Training: observed images teach codes that condition trajectory denoising',weight='bold')
    box(.15,6.7,2.1,.85,'Observed timed XY');box(3,7.15,2.7,.7,'Original Hilbert image','#f0e8f4');box(3,6.1,2.7,.7,'Derived speed image','#e2f0ec')
    arrow(2.33,7.2,2.9,7.5);arrow(2.33,7,2.9,6.45)
    box(6.4,6.6,2.2,.9,'Partly trainable CNNs\n32D + 32D codes');arrow(5.8,7.5,6.3,7.15);arrow(5.8,6.45,6.3,6.9)
    box(9.4,6.55,2.3,1.,'Joint training objective\nNoise loss + class CE','#e5edf5');arrow(8.7,7.05,9.3,7.05)
    ax.text(7.5,5.8,'Denoising gradients reach CNN tails',ha='center',fontsize=9,color='#40586a')
    ax.plot([.1,11.8],[5.4,5.4],c='#ccd4da',lw=1)
    ax.text(.1,5.05,'New synthesis: available conditions and sampled codes only',weight='bold')
    box(.15,3.75,2.3,.9,'Map + OD + class\nPublic bus stops');box(3.1,3.75,2.3,.9,'Learned route policy\nDuration prior')
    arrow(2.55,4.2,3,4.2);box(6.2,3.75,2.3,.9,'Conditional code prior\n50 DDIM steps','#f0e8f4');arrow(5.5,4.2,6.1,4.2)
    box(9.3,3.65,2.4,1.1,'Motion diffusion\n100 DDIM steps\nCode at every step','#e5edf5');arrow(8.6,4.2,9.2,4.2)
    box(9.3,1.65,2.4,1.,'Road decoder\nComplete timed XY','#e2f0ec');arrow(10.5,3.55,10.5,2.75)
    box(3.1,1.65,5.4,1.,'Independent physical quality, diversity, coverage\nReal-data augmentation on separate cohorts')
    arrow(9.2,2.15,8.6,2.15);ax.text(.2,.65,'No target future GAF at synthesis. No image inversion. No terminal GAF ranking in G40.',fontsize=10)
    finish(fig,'g40-framework')
    # Four independent OD cohorts; intervals are explicitly descriptive.
    fig,axs=plt.subplots(1,2,figsize=(10,4.2));xx=np.arange(len(ARMS)-1)
    for ax,key,title in zip(axs,PRIMARY,('Native speed','Native acceleration')):
        baseline=report['summary']['B'][key]
        for j,arm in enumerate(ARMS[1:]):
            d=report['deltas'][arm][key];value=100*d['difference_vs_B']/baseline
            lo,hi=100*np.asarray(d['cohort_bootstrap95_descriptive'])/baseline
            ax.bar(j,value,color=COLORS[j+1],width=.65,alpha=.8)
            ax.plot([j,j],[lo,hi],c='#30383d',lw=1);ax.scatter([j]*4,100*np.array(d['cohort_differences'])/baseline,c='#333',s=14,zorder=3)
        ax.axhline(0,color='#4c545b',lw=.8);ax.set_xticks(xx,ARMS[1:],rotation=35,ha='right');ax.set(title=title,ylabel='W1 change relative to B (%)')
        ax.grid(axis='y',alpha=.18)
    fig.suptitle('Direct generation: all controls, no GAF selection',fontsize=13);fig.tight_layout();finish(fig,'g40-primary')
    endpoints=[(PRIMARY[0],'Native speed'),(PRIMARY[1],'Native acceleration'),
        ('physical/mean_absolute_acceleration_mps2/w1','One-second acceleration'),('physical/stopped_fraction/w1','Stopped fraction'),
        ('physical/longest_stop_s/w1','Longest stop'),('physical/longest_steady_s/w1','Longest steady movement')]
    fig,axs=plt.subplots(3,2,figsize=(11,9));matrix={}
    for ax,(metric,title) in zip(axs.flat,endpoints):
        values=[]
        for cls in CLASSES:
            means={}
            for arm in ARMS:
                vals=[]
                for key,result in r['evaluations'].items():
                    if key.split('/')[1].rsplit('-',1)[0]!=arm:continue
                    group=result['results']['R']['by_class'][cls]
                    value=group[metric] if '/' not in metric else group['physical'][metric.split('/')[1]]['w1']
                    assert value is not None;vals.append(value)
                assert len(vals)==16;means[arm]=np.mean(vals)
            values.append([100*(means[arm]/means['B']-1) if means['B']>0 else np.nan for arm in ARMS[1:]])
        values=np.asarray(values);matrix[metric]=[[float(v) if np.isfinite(v) else None for v in row] for row in values]
        im=ax.imshow(values,cmap='RdBu_r',vmin=-20,vmax=20,aspect='auto')
        ax.set_xticks(range(7),ARMS[1:],rotation=35,ha='right');ax.set_yticks(range(5),NAMES);ax.set_title(title)
        for i in range(5):
            for j in range(7):
                v=values[i,j];text='n/a' if not np.isfinite(v) else (f'{v:+.1f}' if abs(v)<100 else f'{v:+.0f}')
                ax.text(j,i,text,ha='center',va='center',fontsize=8,color='white' if abs(v)>14 else '#202c35')
    fig.tight_layout(rect=(0,0,.95,.96));fig.colorbar(im,cax=fig.add_axes([.965,.2,.012,.58]),extend='both',label='W1 change vs B (%)')
    fig.suptitle('Five-class physical quality: all adverse and favorable effects retained',fontsize=13);finish(fig,'g40-five-class-motion')
    # Fixed endpoint pairs also get page-width versions. The overview remains
    # archived; these use the same matrix, color limits and complete arm order.
    for part in range(3):
        fig,axs=plt.subplots(2,1,figsize=(7.2,6.8))
        for ax,(metric,title) in zip(axs,endpoints[part*2:part*2+2]):
            values=np.asarray([[np.nan if x is None else x for x in row] for row in matrix[metric]])
            im=ax.imshow(values,cmap='RdBu_r',vmin=-20,vmax=20,aspect='auto')
            ax.set_xticks(range(7),ARMS[1:]);ax.set_yticks(range(5),NAMES)
            ax.tick_params(labelsize=11);ax.set_title(title,fontsize=13)
            for i in range(5):
                for j in range(7):
                    v=values[i,j];label='n/a' if not np.isfinite(v) else (f'{v:+.1f}' if abs(v)<100 else f'{v:+.0f}')
                    ax.text(j,i,label,ha='center',va='center',fontsize=11,
                        color='white' if abs(v)>14 else '#202c35')
        fig.subplots_adjust(left=.24,right=.88,top=.94,bottom=.08,hspace=.38)
        fig.colorbar(im,cax=fig.add_axes([.91,.18,.022,.64]),extend='both',label='W1 change vs B (%)')
        finish(fig,f'g40-class-motion-{part+1}')
    # Utility reports both downstream seeds rather than an unsupported narrow CI.
    ur=r['utility']['results'];regimes=('real','plus-empirical','plus-B','plus-F-SV','plus-J-S','plus-J-V','plus-J-SV','plus-J-RAW')
    labels=('Real','+ resampled','+ B','+ F-SV','+ J-S','+ J-V','+ J-SV','+ RAW');u={}
    for regime in regimes:
        fits=[ur[f'{regime}-{s}'] for s in (4001,4002)]
        u[regime]={'macro_f1':np.mean([f['pooled']['macro_f1'] for f in fits]),
            'seeds':[f['pooled']['macro_f1'] for f in fits],
            'class_f1':np.mean([f['pooled']['per_class_f1'] for f in fits],axis=0).tolist()}
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.4),gridspec_kw={'width_ratios':[1,1.2]})
    axs[0].bar(range(8),[100*u[k]['macro_f1'] for k in regimes],color=['#354d60',*COLORS[:7]],alpha=.85)
    for j,k in enumerate(regimes):axs[0].scatter([j-.08,j+.08],100*np.array(u[k]['seeds']),c='#252c31',s=15)
    axs[0].set_xticks(range(8),labels,rotation=40,ha='right');axs[0].set(ylabel='Macro F1 (%)',ylim=(0,100),title='Independent raw-series task')
    values=100*np.asarray([u[k]['class_f1'] for k in regimes]).T
    im=axs[1].imshow(values,vmin=0,vmax=100,cmap='Blues',aspect='auto');axs[1].set_yticks(range(5),NAMES)
    axs[1].set_xticks(range(8),labels,rotation=40,ha='right');axs[1].set_title('Per-class F1, mean of two fits')
    for i in range(5):
        for j in range(8):axs[1].text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if values[i,j]>60 else '#23323e')
    fig.tight_layout();finish(fig,'g40-augmentation')
    rows=[['Arm','Speed $\\W$','Acceleration $\\W$','Speed wins','Accel. wins']]
    for arm in ARMS:
        vals=[f"{report['summary'][arm][k]:.5f}" for k in PRIMARY]
        wins=['--','--'] if arm=='B' else [str(report['deltas'][arm][k]['better_cells_of16'])+'/16' for k in PRIMARY]
        rows.append([arm,*vals,*wins])
    table('g40-primary','Prospective direct-draw quality on eight episodes, two model seeds and all five classes. Wins count scene/seed means; lower distances are better.','g40-primary','lrrrr',rows)
    allregimes=('real',)+tuple(f'{kind}-{source}' for kind in ('only','plus') for source in ('empirical','B','F-SV','J-S','J-V','J-SV','J-RAW'))
    rows=[['Training data','F1, seed 4001','F1, seed 4002','Mean F1']]
    for regime in allregimes:
        vals=[100*ur[f'{regime}-{s}']['pooled']['macro_f1'] for s in (4001,4002)]
        rows.append([regime,*[f'{v:.2f}' for v in vals],f'{np.mean(vals):.2f}'])
    table('g40-utility','All fifteen downstream regimes. Percent macro F1 on real simulated references from the last two OD cohorts. Each fit has the same optimization budget.','g40-utility','lrrr',rows)
    # Earlier evidence motivates the intervention without mixing populations.
    methods=('Shortest','Uniform','R','R+S','R+V','R+SV');keys=('route_edge_usage_js','weighted_native_speed_w1','weighted_native_acceleration_w1')
    fig,axs=plt.subplots(1,3,figsize=(10.5,3.6))
    for ax,key,title in zip(axs,keys,('Road-edge usage JS','Native speed W1 (m/s)','Native acceleration W1 (m/s²)')):
        vals=[[cell['results'][method]['macro'][key] for cell in g38['results'].values()] for method in methods]
        ax.bar(range(6),[np.mean(x) for x in vals],color=['#777','#a0abb3','#37799a','#9462a6','#258a7b','#c7673c'])
        for j,x in enumerate(vals):ax.scatter([j]*len(x),x,c='#30373c',s=12)
        ax.set_xticks(range(6),methods,rotation=40,ha='right');ax.set_title(title);ax.set_ylim(bottom=0)
    fig.tight_layout();finish(fig,'g38-route-motion')
    methods=('Global-S','Global-V','Global-SV','FixedRoute-S','FixedRoute-V','FixedRoute-SV','FixedRouteT-S','FixedRouteT-V','FixedRouteT-SV')
    fig,axs=plt.subplots(1,2,figsize=(10.3,4.1));positions=np.arange(9)
    for ax,key,title in zip(axs,('weighted_native_speed_w1','weighted_native_acceleration_w1'),('Speed distribution','Acceleration distribution')):
        vals=[factor['methods'][m]['endpoints'][key]['relative_percent'] for m in methods]
        ax.bar(positions,vals,color=[COLORS[2],COLORS[3],COLORS[4]]*3);ax.axhline(0,c='#555',lw=.8)
        ax.set_xticks(positions,['S','V','SV']*3);ax.set(title=title,ylabel='W1 change vs matched R (%)')
        for j,lab in zip((1,4,7),('Global','Fixed route','Fixed route + T')):ax.text(j,-.23,lab,transform=ax.get_xaxis_transform(),ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.28,wspace=.3);finish(fig,'g39-fixed-marginals')
    oldpath=a.evidence/'g39-main.csv';sources[oldpath.name]=hashlib.sha256(oldpath.read_bytes()).hexdigest()
    oldrows=list(csv.DictReader(oldpath.open()));oldarms=('S','V','SV','P','SH','RAW')
    oldmean=lambda arm,k:np.mean([float(x[k]) for x in oldrows if x['arm']==arm])
    fig,axs=plt.subplots(1,2,figsize=(10.2,4));xx=np.arange(6)
    for j,(metric,title,color) in enumerate(zip(PRIMARY,('Speed','Acceleration'),('#477c9d','#cd8050'))):
        baseline=oldmean('B',metric);values=[100*(oldmean(arm,metric)/baseline-1) for arm in oldarms]
        axs[0].bar(xx+(j-.5)*.34,values,.34,label=title,color=color)
    axs[0].set_xticks(xx,oldarms);axs[0].axhline(0,c='#555',lw=.8);axs[0].legend()
    axs[0].set(title='Frozen-feature alignment: direct generation',ylabel='W1 change vs paired B (%)')
    oldregimes=('real','plus-empirical','plus-B','plus-S','plus-V','plus-SV','plus-RAW')
    oldlabels=('Real','+ resampled','+ B','+ S','+ V','+ SV','+ RAW')
    axs[1].bar(range(7),[100*utility39[k]['macro_f1_mean'] for k in oldregimes],color='#608b9c')
    for j,k in enumerate(oldregimes):axs[1].scatter([j-.07,j+.07],100*np.array(utility39[k]['macro_f1_seeds']),c='#333',s=13)
    axs[1].set_xticks(range(7),oldlabels,rotation=35,ha='right');axs[1].set(ylim=(0,100),title='Independent downstream augmentation',ylabel='Macro F1 (%)')
    fig.tight_layout();finish(fig,'g39-alignment-utility')
    variants=('warp-0.1','warp-0.3','jitter-0.1','jitter-0.3','smooth-4','smooth-16')
    means=lambda k:[np.mean([audit['summary'][c][v]['metrics'][k]['mean'] for c in CLASSES]) for v in variants]
    fig,axs=plt.subplots(1,2,figsize=(10,3.7));xx=np.arange(6)
    axs[0].bar(xx-.17,means('spatial_feature_cosine_distance'),.34,color=COLORS[2],label='Spatial')
    axs[0].bar(xx+.17,means('speed_feature_cosine_distance'),.34,color=COLORS[3],label='Speed');axs[0].legend()
    axs[0].set(title='Representation sensitivity',ylabel='Centered-feature cosine distance')
    axs[1].bar(xx,means('acceleration_native_w1_change'),color='#527b9b');axs[1].set(title='Physical change from the same perturbation',ylabel='Native acceleration W1 (m/s²)')
    for ax in axs:ax.set_xticks(xx,variants,rotation=35,ha='right')
    fig.tight_layout();finish(fig,'g40-sensitivity')
    save={'complete':True,'sources':sources,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'figures':made,'tables':tables,'g40_class_relative_matrix':matrix,'utility_summary':u,
        'rules':'All arms/classes retained; class/seed/window/cohort equal means. No cross-round pooled effects. Cohort intervals descriptive.',
        'artifacts':{str(p.relative_to(a.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.rglob('*') if p.is_file()}}
    (a.output/'complete.json').write_text(json.dumps(save,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
