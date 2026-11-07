"""Five-class presentation of existing frozen G35/G36 results; no new evaluation."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[2];z=json.loads((r/'docs/results/g36-statistics/statistics.json').read_text())['contrasts'];b=json.loads((r/'docs/results/g35-training/paired-teacher-audit.json').read_text())['point']['leg']
classes=['bus','motorcycle','private_car','bycicle','taxi_or_uber'];names=['Bus','Motorcycle','Private car','Bicycle','Taxi / ride-hailing'];methods=['S','V','SV']
keys=['native_speed_w1_mps','native_acceleration_w1_mps2','longest_stop_s','stopped_fraction','stop_episodes_per_minute','longest_steady_s'];titles=['Native speed distribution','Native acceleration distribution','Longest-stop distribution','Stopped-fraction distribution','Stop-frequency distribution','Longest steady-movement distribution']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
fig,axs=plt.subplots(3,2,figsize=(9,9.8))
for ax,k,title in zip(axs.flat,keys,titles):
 vals=np.array([[z[f'new_{m}_vs_Uniform']['W1'][k]['classes'][c]['relative_change_of_means_pct'] for m in methods] for c in classes]);im=ax.imshow(vals,cmap='RdBu_r',vmin=-50,vmax=50,aspect='auto');ax.set_xticks(range(3),methods);ax.set_yticks(range(5),names,fontsize=10);ax.set_title(title,fontsize=11)
 for i,c in enumerate(classes):
  for j,m in enumerate(methods):
   n=z[f'new_{m}_vs_Uniform']['W1'][k]['classes'][c]['improved_blocks'];ax.text(j,i,f'{vals[i,j]:+.1f}%\n({n}/4)',ha='center',va='center',fontsize=10,color='white' if abs(vals[i,j])>35 else '#17242d')
fig.subplots_adjust(left=.17,right=.98,top=.93,bottom=.1,hspace=.48,wspace=.95);fig.suptitle('Five-class trajectory quality after GAF selection',fontsize=14);fig.text(.5,.025,'Change from Uniform: blue is lower error; red is higher error.\nParentheses: improving scene / seed cells, not independent significance tests.',ha='center',fontsize=11)
for ext in ['pdf','png']:fig.savefig(r/f'thesis/figures/g36-five-class-motion.{ext}',bbox_inches='tight',dpi=180)
plt.close(fig)
def table(name,caption,label,align,rows):
 (r/f'thesis/tables/{name}.tex').write_text('\\tab{'+caption+'}{'+label+'}{\\begin{tabular}{'+align+'}\\toprule\n'+'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n\\bottomrule\\end{tabular}}\n')
rows=[['Class','Development legs','Spatial F1','Velocity F1']]
for i,c in enumerate(classes):rows.append([names[i],str(int(b['new_spatial']['weighted_class_support'][i])),f"{100*b['new_spatial']['per_class_f1'][i]:.2f}",f"{100*b['new_speed']['per_class_f1'][i]:.2f}"])
rows.append(['Macro F1','7,230',f"{100*b['new_spatial']['macro_f1']:.2f}",f"{100*b['new_speed']['macro_f1']:.2f}"])
table('g35-five-class-readers','Current GAF readers on the same development population. F1 values are percentages. Support counts are legs, not independent simulator scenes.','g35-five-class-readers','lrrr',rows)
rows=[['Class / endpoint','Uniform','S','V','SV']]
for c,name in zip(classes,names):
 for k,title in [('native_speed_w1_mps','speed'),('native_acceleration_w1_mps2','accel.')]:
  v=z['new_S_vs_Uniform']['W1'][k]['classes'][c];rows.append([name+' / '+title,f"{v['baseline_mean']:.3f}",*[f"{z[f'new_{m}_vs_Uniform']['W1'][k]['classes'][c]['method_mean']:.3f}" for m in methods]])
table('g36-five-class-native','Absolute native-distribution W1 errors of the current generator and readers, averaged equally over the four scene/seed cells. Speed uses m/s and acceleration uses m/s$^2$; lower is better. Units must not be pooled.','g36-five-class-native','lrrrr',rows)
paths=['docs/results/g36-statistics/statistics.json','docs/results/g35-training/paired-teacher-audit.json'];manifest={'sources_sha256':{p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in paths},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Full five-class presentation of completed results, descriptive post-hoc cell aggregation; no new evaluation or experiment'};(r/'docs/results/thesis-v19-assets.json').write_text(json.dumps(manifest,indent=2)+'\n')
