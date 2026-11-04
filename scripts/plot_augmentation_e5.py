"""Publication-exportable E5 validation contrasts; intervals include all outcomes."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('docs/results/augmentation-e5')
results=json.loads((root/'summary.json').read_text())['models']
groups=['repeat','timewarp','synthetic_random','synthetic_gaf','synthetic_tcn']
labels=['Repeated real','Time warp','Random synthetic','GAF-selected synthetic','TCN-selected synthetic']
colors=['#777777','#777777','#4477AA','#EE7733','#228833']
fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharex=True,sharey=True)
for ax,arch in zip(axes,['tcn','gru']):
    for i,(group,color) in enumerate(zip(groups,colors)):
        d=results[arch][group]['paired_bootstrap_macro_f1_delta']['real_only']
        lo,hi=[v*100 for v in d['percentile95']]
        ax.plot([lo,hi],[i,i],color=color,lw=2)
        ax.plot(d['observed']*100,i,'o',color=color,ms=6)
    ax.axvline(0,color='black',ls='--',lw=1)
    ax.set_title(arch.upper()+' downstream learner')
    ax.set_yticks(range(len(labels)),labels)
    ax.set_xlabel('Macro-F1 change vs real-only (percentage points)')
    ax.grid(axis='x',alpha=.2)
axes[0].invert_yaxis()
fig.suptitle('Synthetic-data utility on real validation trajectories')
fig.text(.5,.025,'Paired 95% vehicle-bootstrap intervals; three learner seeds, one generator. Exploratory validation, not final test.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.06,1,.94))
fig.savefig(root/'utility.png',dpi=180)
fig.savefig(root/'utility.pdf')
