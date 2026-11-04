"""Calibrate on real held-out vehicles before inspecting generated trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from cnr.data import CLASSES
from cnr.topology import LaneConsistency

p=argparse.ArgumentParser();p.add_argument('--output',default='docs/results/topology-t1')
p.add_argument('--radius',type=float,default=2.);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
matcher=LaneConsistency('data/maps/most.net.xml',radius=a.radius)
config={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'radius_m':a.radius,'max_candidates':12,'reverse_tolerance_m':.05,
        'route_budget':'min(200m, 2 * observed displacement + 5m)',
        'real_selection':'first validation manifest window of each distinct vehicle; no pedestrian topology',
        'map_sha256':hashlib.sha256(Path('data/maps/most.net.xml').read_bytes()).hexdigest(),
        'unresolved_connections':matcher.unresolved_connections,
        'gate':'real geometry coverage >=95% overall, graph consistency among geometry-covered >=95% overall and >=90% each vehicle class; no unresolved connections',
        'limits':['graph consistency within a geometric envelope is not physical trajectory validity',
                  'permissive adjacent-lane changes on same normal edge','no traffic signal or collision model',
                  'pedestrian topology excluded; category-to-vClass remains a hypothesis',
                  '2D map cannot by itself certify bridge elevation or source simulation version',
                  'candidate cap and bounded detour can cause false incompatibility']}
(out/'config.json').write_text(json.dumps(config,indent=2))

def audit(xy,labels,indices=None):
    rows=[]
    for i,(x,label) in enumerate(zip(xy,labels)):
        if CLASSES[int(label)]=='pedestrian':continue
        row=matcher.match(x,CLASSES[int(label)])
        rows.append({'index':int(indices[i]) if indices is not None else i,'class':CLASSES[int(label)],**row})
    perclass={}
    for name in CLASSES[1:]:
        rr=[r for r in rows if r['class']==name];covered=[r for r in rr if r['geometry_covered']]
        if not rr:continue
        perclass[name]={'trajectories':len(rr),'geometry_covered_fraction':float(np.mean([r['geometry_covered'] for r in rr])),
            'graph_consistent_fraction':float(np.mean([r['graph_consistent'] for r in rr])),
            'graph_consistent_given_geometry':float(np.mean([r['graph_consistent'] for r in covered])) if covered else None,
            'generated_point_coverage':float(np.mean([r['generated_point_coverage'] for r in rr])),
            'truncation_warning_fraction':float(np.mean([r['candidate_truncation_warning'] for r in rr]))}
    covered=[r for r in rows if r['geometry_covered']]
    summary={'trajectories':len(rows),'per_class':perclass,
        'geometry_covered_fraction':float(np.mean([r['geometry_covered'] for r in rows])),
        'graph_consistent_fraction':float(np.mean([r['graph_consistent'] for r in rows])),
        'graph_consistent_given_geometry':float(np.mean([r['graph_consistent'] for r in covered])) if covered else None}
    return {'summary':summary,'rows':rows}

data=Path('data/processed/most64');real=np.load(data/'val.npz')
manifest=pd.read_csv(data/'manifest.csv',dtype={'vehicle_id':str})
selected=manifest[manifest.split=='val'].drop_duplicates('vehicle_id').array_index.to_numpy()
calibration=audit(real['xy'][selected],real['labels'][selected],selected)
s=calibration['summary']
passed=(matcher.unresolved_connections==0 and s['geometry_covered_fraction']>=.95
        and (s['graph_consistent_given_geometry'] or 0)>=.95
        and all((v['graph_consistent_given_geometry'] or 0)>=.90 for v in s['per_class'].values()))
report={'config':config,'calibration':calibration,'calibration_gate_passed':passed,'generated':{},
        'interpretation':'diagnostic only; passing is not proof of legal driving'}
# Reversal preserves point geometry but need not be illegal on a bidirectional road.
moving=selected[np.linalg.norm(real['xy'][selected,-1]-real['xy'][selected,0],axis=-1)>5]
report['reversed_real_control']=audit(real['xy'][moving,::-1],real['labels'][moving],moving)
if passed:
    root=Path('runs/candidate-selection-g3')
    for policy in ['random','road','road_delta_gaf','road_delta_tcn']:
        sample=np.load(root/f'selected_{policy}.npz')
        report['generated'][policy]=audit(sample['xy'],sample['labels'])
        print(json.dumps({'policy':policy,'summary':report['generated'][policy]['summary']}),flush=True)
else:
    report['generation_evaluation_skipped']='Real-data calibration gate failed; do not label generated tracks invalid using an uncalibrated matcher.'
report['elapsed_s']=time.monotonic()-started
(out/'results.json').write_text(json.dumps(report,indent=2))
lines=['# T1：有向车道连接一致性诊断','',
       f"真实数据校验门槛：{'通过' if passed else '未通过；本轮不评价生成轨迹'}。候选半径{a.radius}m；行人单列，未进行拓扑评价。",'',
       '同一路口可能存在多条候选车道，使用全序列动态规划；保留via内部车道，不按平面交叉自动连边。',
       '同edge相邻车道切换为宽松近似；通过不等于合法驾驶，未校验信号、碰撞及真实vClass。','',
       '| 数据/策略 | 轨迹数 | 全点有候选 | 全序列图一致 | 几何覆盖条件下图一致 |',
       '|---|---:|---:|---:|---:|']
for name,result in [('real_unique_vehicles',calibration),('reversed_moving_real',report['reversed_real_control'])]+list(report['generated'].items()):
    s=result['summary'];conditional='N/A' if s['graph_consistent_given_geometry'] is None else f"{100*s['graph_consistent_given_geometry']:.2f}%"
    lines.append(f"| {name} | {s['trajectories']} | {100*s['geometry_covered_fraction']:.2f}% | {100*s['graph_consistent_fraction']:.2f}% | {conditional} |")
lines+=['','反转轨迹仅为保持空间点不变的方向扰动，并非所有情况下都是不合法反例。不同半径可能允许匹配到邻近反向车道。',
        '逐车型结果、候选截断警告、最少不兼容转移次数及所有配置见results.json。此轮只使用验证车辆。',
        '方法依据：[SUMO路网格式](https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html)；[行人规则](https://sumo.dlr.de/docs/Simulation/Pedestrians.html)。','']
(out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)
