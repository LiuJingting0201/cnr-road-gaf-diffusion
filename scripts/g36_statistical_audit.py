"""Post-hoc descriptive audit of every G36 endpoint; no new fitting or selection.

Equal scene/seed and equal class summaries. Four cells are not independent
replications. W1 and matched-request MAE answer different questions; neither
is substituted for the predeclared endpoint after seeing the outcomes.
"""
import argparse,json,statistics
from pathlib import Path
EPISODES=(360201,360202);SEEDS=(42,43)
CLASSES=('bus','motorcycle','private_car','bycicle','taxi_or_uber')
def avg(v):return statistics.mean(v) if v and all(x is not None for x in v) else None
def value(r,metric,kind,cl=None):
 cs=[cl] if cl else CLASSES
 if metric.startswith('native_'):return avg([r['paired_by_class'][c][metric] for c in cs])
 values=[]
 for c in cs:
  endpoint=r['paired_by_class'][c]['physical'][metric]
  values.append(endpoint['w1'] if kind=='W1' else endpoint['matched_request_mean_absolute_error'].get('mean'))
 return avg(values)

def contrast(pairs,metric,kind,cl=None):
 a=[value(x,metric,kind,cl) for x,y in pairs];b=[value(y,metric,kind,cl) for x,y in pairs]
 if None in a+b:return {'supported':False}
 rel=[100*(y/x-1) if x else None for x,y in zip(a,b)]
 return {'supported':True,'baseline_mean':avg(a),'method_mean':avg(b),'relative_change_of_means_pct':100*(avg(b)/avg(a)-1) if avg(a) else None,'block_relative_changes_pct':rel,'improved_blocks':sum(y<x for x,y in zip(a,b)),'total_blocks':len(a)}
def main(source,output):
 x=json.loads(source.read_text());assert x['complete'] and len(x['results'])==56
 r=x['results'];metrics=list(next(iter(r.values()))['paired_macro_w1'])
 groups={}
 for m in ('S','V','SV'):
  groups['new_'+m+'_vs_Uniform']=[(r[f'{e}/new-{s}-new/Uniform'],r[f'{e}/new-{s}-new/{m}']) for e in EPISODES for s in SEEDS]
  groups['new_vs_old_teacher_'+m]=[(r[f'{e}/new-{s}-old/{m}'],r[f'{e}/new-{s}-new/{m}']) for e in EPISODES for s in SEEDS]
 groups['expanded_vs_small_pool']=[(r[f'{e}/old-{s}/pool'],r[f'{e}/new-{s}/pool']) for e in EPISODES for s in SEEDS]
 out={'post_hoc':True,'blocks':[(e,s) for e in EPISODES for s in SEEDS],'contrasts':{},'independence':'Two episodes on shared source plans; two generator seeds are paired technical replicates. No p-values or population confidence intervals.'}
 for name,pairs in groups.items():
  out['contrasts'][name]={kind:{k:{'macro':contrast(pairs,k,kind),'classes':{c:contrast(pairs,k,kind,c) for c in CLASSES}} for k in metrics+( ['native_speed_w1_mps','native_acceleration_w1_mps2'] if kind=='W1' else [])} for kind in ('W1','MAE')}
 output.mkdir(parents=True,exist_ok=True);(output/'statistics.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=['# G36 statistical perspective audit','', '**Post-hoc descriptive analysis.** All original endpoints/results retained. Changes below compare equally weighted scene/seed cell means; each cell gives equal weight to five classes. Negative is lower error. Four cells are not four independent scenes. Counts are descriptive, not significance tests.','']
 for kind in ('W1','MAE'):
  lines+=['## '+kind,'','| Endpoint | Spatial S | Speed V | Dual SV |','|---|---:|---:|---:|']
  for k in out['contrasts']['new_S_vs_Uniform'][kind]:
   cells=[]
   for m in ('S','V','SV'):
    a=out['contrasts']['new_'+m+'_vs_Uniform'][kind][k]['macro'];cells.append(f"{a['relative_change_of_means_pct']:+.1f}% ({a['improved_blocks']}/4)" if a['supported'] and a['relative_change_of_means_pct'] is not None else 'unsupported')
   lines+=['| '+k+' | '+' | '.join(cells)+' |']
  lines+=['']
 (output/'TABLES.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('docs/results/g36-generation/completed-metrics.json'));p.add_argument('--output',type=Path,default=Path('docs/results/g36-statistics'));a=p.parse_args();main(a.source,a.output)
