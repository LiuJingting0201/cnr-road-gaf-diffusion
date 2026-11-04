from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('docs/results/g6-analysis');rows=json.loads((root/'summary.json').read_text())['rows']
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for family,color,marker in [('path-g5-repeat','#2677b9','o'),('preserved-g6','#cf6b26','s')]:
 for ax,key in zip(axes,['gaf_accuracy','tcn_accuracy']):
  subset=[next(r for r in rows if r['family']==family and r['variant']==v) for v in ['none','gaf_0.3','gaf_1','gaf_3']]
  ax.plot([r['speed_w1_mps']['mean'] for r in subset],[100*r[key]['mean'] for r in subset],color=color,marker=marker,label='Standard GAF' if family=='path-g5-repeat' else 'Travel-preserving GAF')
  for r,w in zip(subset,[0,.3,1,3]):
   ax.annotate(str(w),(r['speed_w1_mps']['mean'],100*r[key]['mean']),xytext=(4,5 if family=='path-g5-repeat' else -12),textcoords='offset points',fontsize=8,color=color)
  ax.set(xlabel='Speed Wasserstein distance (m/s), lower is better',ylabel='Independent class consistency (%)',title='GAF audit' if key=='gaf_accuracy' else 'Sequence audit');ax.grid(alpha=.2);ax.legend()
fig.suptitle('GAF guidance: three-generator means; labels are guidance strengths; validation only')
fig.tight_layout();fig.savefig(root/'guidance_tradeoff.png',dpi=160);fig.savefig(root/'guidance_tradeoff.svg')
