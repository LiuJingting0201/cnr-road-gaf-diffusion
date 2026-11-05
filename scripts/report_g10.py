"""Report frozen speed endpoint first and render its independent replication plot."""
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
root=Path('docs/results/g10-independent')
d=json.loads((root/'results.json').read_text())
assert d['mid']['denominator']==9*d['N']==d['terminal']['denominator']
index=['速度 W1 (m/s)','加速度 W1 (m/s²)','行程 W1 (m)','停留比例误差']
lines=['# 独立 SUMO 复核：弯曲道路第 76 步 GAF guidance',
f"8 次独立车流仿真；每批 100 个测试车辆、120 辆背景车，按预定规则筛选。共 {d['N']} 个起点，其中 {d['n_high']} 个属于类别内转向较多组。三组冻结扩散权重，各三组新噪声。每种主策略返回 {d['mid']['denominator']} 条轨迹。",
'主要比较采样中减末端（相同 ESS 和随机规则）；W1 差值为负有利于采样中引导。95% 区间按整次 SUMO 仿真配对重采样 2,000 次。高减低的交互与主终点分别列示。']
for g,title in [('high','较多转弯'),('low','较少转弯'),('all','全部道路')]:
 lines.extend(['## '+title,'|指标|采样中|末端|差值 [95%]|','|---|---:|---:|---:|'])
 for j,k in enumerate(index):
  a=d[g];l,h=a['CI'][j]
  lines.append(f"|{k}|{d['mid'][g][j]:.4f}|{d['terminal'][g][j]:.4f}|{a['delta'][j]:+.4f} [{l:+.4f}, {h:+.4f}]|")
lines+=['## 适用范围',f"路网通过数（采样中/末端）：{d['mid']['graph_pass']}/{d['terminal']['graph_pass']}；耗尽数：{d['mid']['caps']}/{d['terminal']['caps']}。独立输出间距离：{d['mid']['diversity']:.2f}/{d['terminal']['diversity']:.2f} m。",
'地图与训练相同；独立的是 SUMO 需求、车辆及仿真随机种子。受控 OD 需求没有原时刻表、派单或真实流量标定。片段按完整观测选取；冻结模型只读取起点/类别/独立道路。不能解释为新城市或真实道路证据。']
(root/'REPORT.md').write_text('\n\n'.join(lines[:3])+'\n\n'+'\n'.join(lines[3:])+'\n')
old=json.loads(Path('docs/results/g9-timing/report76.json').read_text())
c=canvas.Canvas('thesis/figures/g10-replication.pdf',pagesize=(480,152))
for j,(metric,name) in enumerate([(0,'Speed discrepancy (m/s)'),(2,'Travel discrepancy (m)')]):
 x=65+j*238;w=150
 values=[(old['mid-terminal_high']['point'][metric],old['mid-terminal_high']['CI'][metric]),(d['high']['delta'][metric],d['high']['CI'][metric])]
 low=min(0,*[v[1][0] for v in values]);hi=max(0,*[v[1][1] for v in values]);pad=(hi-low)*.13;low-=pad;hi+=pad
 axis=lambda v:x+(v-low)/(hi-low)*w
 c.setFillColor(HexColor('#263b49'));c.setFont('Helvetica-Bold',10);c.drawString(x,136,name)
 c.setStrokeColor(HexColor('#9eacb7'));c.setDash(2,2);c.line(axis(0),22,axis(0),119);c.setDash()
 for i,(p,(a,b)) in enumerate(values):
  y=94-52*i;col=HexColor(['#748b99','#126959'][i]);c.setStrokeColor(col);c.setFillColor(col);c.setLineWidth(2);c.line(axis(a),y,axis(b),y);c.circle(axis(p),y,3,stroke=0,fill=1)
  c.setFont('Helvetica',8);c.drawRightString(x-4,y-3,['Development','New SUMO'][i]);c.drawCentredString(x+w/2,y-15,f'{p:+.3f}   [{a:+.3f}, {b:+.3f}]')
c.save()
