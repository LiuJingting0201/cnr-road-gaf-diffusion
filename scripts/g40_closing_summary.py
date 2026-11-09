"""Summarize all fixed condition interventions and downstream episode checks.

Post-completion descriptive review; no fits, sample selection or endpoint changes.
"""
import hashlib,json,statistics
from pathlib import Path
import argparse

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
 r=json.loads(a.audit.read_text());e=json.loads(a.evidence.read_text())
 assert r['complete'] and len(r['results'])==16 and e['all_stages']['complete']
 arms=('B','F-SV','J-S','J-V','J-SV','J-RAW','J-SH','C');summary={}
 for arm in arms:
  rows=[row for key,v in r['results'].items() if key.rsplit('-',1)[0]==arm for row in v['records']]
  assert len(rows)==250
  summary[arm]={}
  for level in (50,250,500,750,950):
   selected=[x for x in rows if x['noise_index']==level];assert len(selected)==50
   summary[arm][str(level)]={field:{k:statistics.mean(x[field][k] for x in selected) for k in selected[0][field]} for field in ('noise_mse','prediction_rms_change')}
  if arm=='B':assert all(max(x['prediction_rms_change'].values())==0 for x in rows)
  if arm=='C':assert all(x['prediction_rms_change']['same_class_other_code']==x['prediction_rms_change']['sampled_prior']==0 for x in rows)
 regimes=sorted({x.rsplit('-',1)[0] for x in e['utility']['results']});cohorts={}
 for regime in regimes:
  cohorts[regime]={str(cohort):{str(seed):statistics.mean(e['utility']['results'][f'{regime}-{seed}']['per_episode'][str(ep)]['macro_f1'] for ep in episodes) for seed in (4001,4002)} for cohort,episodes in ((3,(400205,400206)),(4,(400207,400208)))}
  for v in cohorts[regime].values():v['mean']=statistics.mean(v.values())
 a.output.mkdir(parents=True,exist_ok=False)
 payload={'complete':True,'audit_sha256':sha(a.audit),'evidence_sha256':sha(a.evidence),'source_sha256':sha(Path(__file__)),'condition_interventions':summary,'utility_episode_equal_cohorts':cohorts,'limits':['Condition interventions are in-sample on 25 parents, five fixed levels and two model seeds.','Utility cohort summaries average episode macro F1, unlike the pooled main endpoint. Two windows share parents; two learner seeds are not new cohorts.']}
 (a.output/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
 lines=[r'\tab{Condition use on training parents: J-SV noise-prediction MSE under four fixed interventions. Means over 25 parents and two model seeds. This is an in-sample mechanism check, not prospective quality.}{g40-condition-use}{\begin{tabular}{rrrrr}\toprule',r'Noise index & Assigned & Same-class swap & Zero & Sampled prior \\\midrule']
 for n in (50,250,500,750,950):
  m=summary['J-SV'][str(n)]['noise_mse'];lines.append(str(n)+' & '+' & '.join(f'{m[k]:.6f}' for k in ('assigned_training_code','same_class_other_code','zero_code','sampled_prior'))+r' \\')
 lines.append(r'\bottomrule\end{tabular}}');(a.output/'g40-condition-use.tex').write_text('\n'.join(lines)+'\n')
 lines=[r'\tab{Utility by evaluation OD cohort. Entries average episode macro F1 over both windows and both learner seeds (percent); they are not the pooled main score. All fifteen regimes are retained.}{g40-utility-cohorts}{\begin{tabular}{lrr}\toprule',r'Training data & Cohort 3 & Cohort 4 \\\midrule']
 for k in regimes:lines.append(k+' & '+' & '.join(f'{100*cohorts[k][c]["mean"]:.2f}' for c in ('3','4'))+r' \\')
 lines.append(r'\bottomrule\end{tabular}}');(a.output/'g40-utility-cohorts.tex').write_text('\n'.join(lines)+'\n')
 for i,sub in enumerate((arms[:4],arms[4:]),1):
  lines=[r'\tab{All-arm condition interventions, part '+str(i)+r'. Noise-prediction MSE on the fixed training-parent audit; means over parents and seeds.}{g40-condition-all-'+str(i)+r'}{\begin{tabular}{lrrrrr}\toprule',r'Arm & Index & Assigned & Swap & Zero & Prior \\\midrule']
  for arm in sub:
   for n in (50,250,500,750,950):
    m=summary[arm][str(n)]['noise_mse'];lines.append(arm+' & '+str(n)+' & '+' & '.join(f'{m[k]:.6f}' for k in ('assigned_training_code','same_class_other_code','zero_code','sampled_prior'))+r' \\')
  lines.append(r'\bottomrule\end{tabular}}');(a.output/f'g40-condition-all-{i}.tex').write_text('\n'.join(lines)+'\n')
 (a.output/'complete.json').write_text(json.dumps({'complete':True,'source_sha256':sha(Path(__file__)),'artifacts':{p.name:sha(p) for p in a.output.iterdir()}},indent=2)+'\n')
 print(json.dumps({'complete':True,'models':16,'interventions':len(arms)*5*4,'utility_regimes':len(regimes),'J-S_minus_B_cohort_percentage_points':{c:100*(cohorts['plus-J-S'][c]['mean']-cohorts['plus-B'][c]['mean']) for c in ('3','4')}}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('audit','evidence','output'):p.add_argument('--'+k,type=Path,required=True)
 main(p.parse_args())
