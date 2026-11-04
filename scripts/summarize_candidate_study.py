"""Summarize the preregistered fixed-pool selection study without hiding tradeoffs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--report',default='docs/results/candidate-selection-g3.json')
p.add_argument('--output',default='docs/results/candidate-selection-g3');a=p.parse_args()
r=json.loads(Path(a.report).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
lines=['# G3：地图条件生成器上的公平语义筛选','',
       f"验证集固定 {r['conditions']} 个车辆条件、{r['candidates_total']:,} 条候选，每组8选1。所有策略使用同一候选池。",
       '生成器固定为G2 map/seed42/末步EMA，DDIM100；这评估候选筛选，不是生成器训练收益。','',
       '| 策略 | 生成点贴路率（5m） | 独立审计类别一致性 | 每类速度W1均值 m/s | 同条件离散距离 m |',
       '|---|---:|---:|---:|---:|']
for name,v in r['policies'].items():
    lines.append(f"| {name} | {100*v['road_within_5m']:.2f}% | {100*v['audit_macro_target_accuracy']:.2f}% | {v['mean_class_speed_wasserstein_mps']:.3f} | {v['within_condition_mean_pairwise_distance_m']:.2f} |")
lines+=['','## 成对差值及探索性95%区间','',
        '按类别分层、以车辆条件为单位重采样2,000次。下表单位均为百分点；没有多重检验校正，不是最终测试结论。','',
        '| 比较（前者减后者） | 贴路率差值 [区间] | 类别一致性差值 [区间] |','|---|---:|---:|']
for name,v in r['paired_comparisons'].items():
    def cell(key):
        x=v[key];return f"{100*x['difference']:+.2f} [{100*x['ci95'][0]:+.2f}, {100*x['ci95'][1]:+.2f}]"
    lines.append(f"| {name} | {cell('road_within_5m')} | {cell('audit_target_accuracy')} |")
lines+=['','## 可支持的结论','',
        '位移GAF评分相比随机选择提高了独立语义一致性；它确实有筛选效用。但同输入TCN取得更高一致性，道路组合后两者没有明确差异，因此暂不支持GAF具有独特或不可替代的优势。',
        '道路筛选提高贴路率，同时会改变运动分布；加入GAF可恢复部分语义一致性，但牺牲部分几何贴合。这一冲突应作为方法需要解决的问题。',
        '不能将贴路率称为路径合法率；尚未验证单行方向、连接、碰撞和信号约束。同条件离散距离只衡量输出分散程度，不证明有效多样性。',
        '审计器是独立权重的XY/motion TCN，对TCN评分可能存在架构偏好；需要不同架构与下游真实测试集评价。选择器/审计器都可能对生成分布失效。全部为一个生成器种子、验证集探索。','']
(out/'REPORT.md').write_text('\n'.join(lines))
fig,ax=plt.subplots(figsize=(10,6))
offsets={'random':(8,-18),'road':(8,-15),'hilbert':(-45,8),'road_hilbert':(8,-15),
         'delta_gaf':(-75,8),'road_delta_gaf':(-112,12),'sequence':(8,8),
         'road_sequence':(8,8),'delta_tcn':(8,8),'road_delta_tcn':(-112,-20)}
for name,v in r['policies'].items():
    x=100*v['road_within_5m'];y=100*v['audit_macro_target_accuracy']
    ax.scatter(x,y,s=65,marker='s' if name.startswith('road') else 'o',
               color='#ac4b21' if 'gaf' in name else '#236c9a')
    ax.annotate(name,(x,y),xytext=offsets[name],textcoords='offset points',fontsize=9)
ax.set(xlabel='Generated points within 5 m of permitted lanes (%)',
       ylabel='Independent auditor target-class agreement (%)',
       title='G3: equal candidate budget, geometry–semantics tradeoff',xlim=(49,94),ylim=(48,92))
ax.grid(alpha=.2);fig.tight_layout()
fig.savefig(out/'tradeoff.png',dpi=180);fig.savefig(out/'tradeoff.svg');plt.close(fig)
