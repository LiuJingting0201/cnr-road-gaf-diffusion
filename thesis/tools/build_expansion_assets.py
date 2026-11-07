"""Reproduce v17 figures/tables from immutable G35/G36 aggregate receipts."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];O=R/'thesis/figures';T=R/'thesis/tables'
paths=['docs/results/g35-training/completed-summary.json','docs/results/g35-training/paired-teacher-audit.json','docs/results/g36-generation/completed-metrics.json','docs/results/g36-statistics/statistics.json']
a,b,g,z=[json.loads((R/p).read_text()) for p in paths];rr=g['results'];stats=z['contrasts'];classes=['bus','motorcycle','private_car','bycicle','taxi_or_uber'];names=['Bus','Motorcycle','Private car','Bicycle','Taxi / ride-hailing'];colors=['#2c6b8e','#d08b38','#417f63']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'font.family':'DejaVu Sans','axes.titleweight':'bold'})
made=[]
def finish(fig,name):
 fig.savefig(O/(name+'.pdf'),bbox_inches='tight');fig.savefig(O/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig);made.append(name)
def table(name,caption,label,columns,rows):
 text='\\tab{'+caption+'}{'+label+'}{\\begin{tabular}{'+columns+'}\\toprule\n'+'\n'.join(' & '.join(row)+r' \\' for row in rows)+ '\n\\bottomrule\\end{tabular}}\n'
 (T/(name+'.tex')).write_text(text)
rec=a['receipts'];groups=rec['g35-data/data-complete.json']['groups'];teach={v:rec[f'g35-data/teachers/{v}-42/complete.json'] for v in ['spatial','speed']}
fig,ax=plt.subplots(figsize=(8,3.6));xx=np.arange(5);ax.bar(xx-.19,[groups['train']['class_counts'][c] for c in classes],.38,label='Training',color=colors[0]);ax.bar(xx+.19,[groups['val']['class_counts'][c] for c in classes],.38,label='Development',color=colors[1]);ax.set_xticks(xx,names);ax.tick_params(axis='x',labelsize=9);ax.set_yscale('log');ax.set_ylabel('Complete vehicle legs (log scale)');ax.legend();finish(fig,'g35-corpus')
rows=[['Class','Train legs','Train parents','Dev. legs']]+[[names[i].replace(' / ','/'),str(groups['train']['class_counts'][c]),str(groups['train']['class_parent_counts'][c]),str(groups['val']['class_counts'][c])] for i,c in enumerate(classes)]+[['Total','29,895','24,073','7,230']];table('g35-cohort','Expanded native-demand corpus. Activity legs and distinct parent identities are separate units.','g35-cohort','lrrr',rows)
fig,axs=plt.subplots(1,2,figsize=(8.6,3.6));models=['old_spatial','new_spatial','old_speed','new_speed'];labels=['Pilot\nspatial','Expanded\nspatial','Pilot\nspeed','Expanded\nspeed']
for ax,k,title in zip(axs,['accuracy','macro_f1'],['Accuracy','Macro F1']):
 vals=[100*b['point']['leg'][m][k] for m in models];ax.bar(range(4),vals,color=[colors[0],colors[2],colors[0],colors[2]]);ax.set_xticks(range(4),labels);ax.set_ylim(0,105);ax.set_title(title);ax.set_ylabel('% on shared development set')
 for i,v in enumerate(vals):ax.text(i,v+2,f'{v:.1f}',ha='center',fontsize=9)
fig.tight_layout();finish(fig,'g35-teacher-comparison')
fig,axs=plt.subplots(1,2,figsize=(8.8,4.2))
for ax,view in zip(axs,['spatial','speed']):
 cm=np.array(teach[view]['val']['confusion']);norm=100*cm/cm.sum(1,keepdims=True);im=ax.imshow(norm,vmin=0,vmax=100,cmap='Blues');ax.set_xticks(range(5),['Bus','Moto','Car','Bike','Taxi'],rotation=35,ha='right');ax.set_yticks(range(5),names if view=='spatial' else ['']*5);ax.set_xlabel('Predicted class');ax.set_title(view.capitalize()+' GAF')
 for i in range(5):
  for j in range(5):ax.text(j,i,f'{norm[i,j]:.0f}',ha='center',va='center',color='white' if norm[i,j]>55 else '#233442',fontsize=8)
axs[0].set_ylabel('True class');fig.subplots_adjust(wspace=.12,right=.89);fig.colorbar(im,cax=fig.add_axes([.91,.17,.02,.65]),label='Row percentage');finish(fig,'g35-confusion')
rows=[['Reader','Accuracy','Macro F1','Taxi F1']]
for m in models:
 v=b['point']['leg'][m];rows.append([m.replace('_',' ').title(),f"{100*v['accuracy']:.2f}",f"{100*v['macro_f1']:.2f}",f"{100*v['per_class_f1'][4]:.2f}" if 'per_class_f1' in v else '--'])
table('g35-readers','Fixed old and expanded readers on the same 7,230 development legs. Percentages; no checkpoint selection.','g35-readers','lrrr',rows)
fig,axs=plt.subplots(1,2,figsize=(8.4,3.8));blocks=['Morning\nseed 42','Morning\nseed 43','Later\nseed 42','Later\nseed 43']
for ax,k,title in zip(axs,['native_speed_w1_mps','native_acceleration_w1_mps2'],['Native speed distribution','Native acceleration distribution']):
 vals=stats['expanded_vs_small_pool']['W1'][k]['macro']['block_relative_changes_pct'];ax.bar(range(4),vals,color=[colors[1] if v>0 else colors[0] for v in vals]);ax.axhline(0,color='#666',lw=.7);ax.set_xticks(range(4),blocks);ax.set_ylabel('W1 change vs pilot (%)');ax.set_title(title)
 for i,v in enumerate(vals):ax.text(i,v+(2 if v>=0 else -2),f'{v:+.1f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=9)
 ax.margins(y=.2)
fig.tight_layout();finish(fig,'g36-generator-context')
metrics=list(stats['new_S_vs_Uniform']['W1']);labels={'duration_s':'Duration [control]','arc_length_m':'Observed arc [control]','longest_steady_s':'Longest steady movement','mean_absolute_lag4_speed_change_mps':'Four-second speed change','mean_speed_mps':'Mean speed [control]','mean_absolute_acceleration_mps2':'Mean absolute acceleration','stopped_fraction':'Stopped fraction','speed_p95_mps':'95th-percentile speed','longest_stop_s':'Longest stop','stop_episodes_per_minute':'Stop episodes / minute','native_speed_w1_mps':'Native speed distribution','native_acceleration_w1_mps2':'Native acceleration distribution'}
fig,ax=plt.subplots(figsize=(8.2,6))
ks=list(stats['new_S_vs_Uniform']['W1']);vals=[]
for k in ks:
 row=[]
 for kind in ['W1','MAE']:
  for m in ['S','V','SV']:
   d=stats[f'new_{m}_vs_Uniform'][kind].get(k);row.append(d['macro']['relative_change_of_means_pct'] if d else np.nan)
 vals.append(row)
vals=np.array(vals);im=ax.imshow(vals,cmap='RdBu_r',vmin=-12,vmax=12,aspect='auto')
ax.set_xticks(range(6),['W1: S','W1: V','W1: SV','MAE: S','MAE: V','MAE: SV'],fontsize=10);ax.set_yticks(range(len(ks)),[labels[k] for k in ks],fontsize=10);ax.axvline(2.5,color='white',lw=3)
for i in range(len(ks)):
 for j in range(6):ax.text(j,i,f'{vals[i,j]:+.1f}' if np.isfinite(vals[i,j]) else 'n/a',ha='center',va='center',fontsize=10)
fig.tight_layout();finish(fig,'g36-metric-audit')
fig,ax=plt.subplots(figsize=(8,4.6));ys=np.arange(5)
for j,m in enumerate(['S','V','SV']):
 q=stats[f'new_{m}_vs_Uniform']['MAE']['longest_stop_s']['classes'];vals=[q[c]['relative_change_of_means_pct'] for c in classes];ax.barh(ys+(j-1)*.24,vals,.23,color=colors[j],label=m)
ax.axvline(0,color='#555',lw=.8);ax.set_yticks(ys,names);ax.invert_yaxis();ax.set_xlabel('Longest-stop request MAE change vs Uniform (%)');ax.legend(title='GAF policy');finish(fig,'g36-class-stopping')
fig,ax=plt.subplots(figsize=(8,4.3));cs=['bus','motorcycle'];marks=['o','s']
for i,c in enumerate(cs):
 vals=stats['new_S_vs_Uniform']['MAE']['longest_stop_s']['classes'][c]['block_relative_changes_pct'];ax.plot(range(4),vals,marker=marks[i],color=colors[i],label=names[classes.index(c)],lw=1.7)
ax.axhline(0,color='#666',lw=.8);ax.set_xticks(range(4),blocks);ax.set_ylabel('Longest-stop request MAE change (%)');ax.legend();ax.set_title('Spatial GAF: an exploratory class-specific pattern');finish(fig,'g36-stopping-blocks')
fig,ax=plt.subplots(figsize=(8,4.2));keys=['longest_stop_s','longest_steady_s','speed_p95_mps','stopped_fraction','stop_episodes_per_minute'];xx=np.arange(len(keys))
for j,m in enumerate(['S','V','SV']):
 vals=[stats[f'new_vs_old_teacher_{m}']['MAE'][k]['macro']['relative_change_of_means_pct'] for k in keys];ax.bar(xx+(j-1)*.23,vals,.23,color=colors[j],label=m)
ax.axhline(0,color='#555',lw=.8);ax.set_xticks(xx,['Longest\nstop','Longest\nsteady','p95\nspeed','Stopped\nfraction','Stop\nfrequency']);ax.set_ylabel('MAE change: expanded vs pilot teacher (%)');ax.legend(title='Same new-generator candidates');finish(fig,'g36-teacher-effects')
for kind in ['W1','MAE']:
 rows=[['Endpoint','Spatial S','Speed V','Dual SV']]
 for k in stats['new_S_vs_Uniform'][kind]:
  cells=[]
  for m in ['S','V','SV']:
   v=stats[f'new_{m}_vs_Uniform'][kind][k]['macro'];cells.append(f"{v['relative_change_of_means_pct']:+.1f} ({v['improved_blocks']}/4)")
  rows.append([labels[k].replace(' / ','/'),*cells])
 table('g36-'+kind.lower(),kind+' change relative to Uniform (percent), with number of improving scene/seed cells in parentheses. Post-hoc equal-cell summary; lower is better. No significance test is implied.','g36-'+kind.lower(),'lrrr',rows)
rows=[['Scene','Seed','Speed change','Acceleration change']]
for i,(e,seed) in enumerate([(e,s) for e in [360201,360202] for s in [42,43]]):rows.append(['Morning' if e==360201 else 'Later',str(seed),*[f"{stats['expanded_vs_small_pool']['W1'][k]['macro']['block_relative_changes_pct'][i]:+.1f}\\%" for k in ['native_speed_w1_mps','native_acceleration_w1_mps2']]])
table('g36-generator','Expanded versus pilot native generators, full candidate pools: class-macro native W1 changes. Each cell uses its own training-side duration prior.','g36-generator','llrr',rows)
manifest={'sources':{p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in paths},'figures':made,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Aggregate receipts only; no fabricated trajectory examples or new evaluation'}
(R/'docs/results/thesis-v17-assets.json').write_text(json.dumps(manifest,indent=2)+'\n')
