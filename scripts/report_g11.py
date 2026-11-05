"""Summarize the predeclared G11 outcomes and create the thesis figure."""
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
root=Path('docs/results/g11-state');d=json.loads((root/'results.json').read_text());g=json.loads((root/'groups.json').read_text())
arm=d['arms'];hist={int(k):v for k,v in d['trigger_histogram'].items()};units=['速度 W1，m/s','加速度 W1，m/s²','行程 W1，m','停留比例误差']
lines=['# 新 SUMO 实验：候选运动稳定后再评分',f"独立仿真 {d['episodes']} 批，{d['n']} 个新请求；每策略 {arm['mid']['total']} 条输出。主指标为速度 W1，负差值有利于采样中引导。", '|指标|状态触发|匹配末端|差值 [整批配对95%区间]|状态收益减打乱时机收益 [95%]|','|---|---:|---:|---:|---:|']
for j,k in enumerate(units):
 vals=[]
 for name in ['adaptive','interaction']:
  x=d[name];lo,hi=x['CI'][j];vals.append(f"{x['delta'][j]:+.4f} [{lo:+.4f}, {hi:+.4f}]")
 lines.append(f"|{k}|{arm['mid']['quality'][j]:.4f}|{arm['terminal']['quality'][j]:.4f}|"+'|'.join(vals)+'|')
lines+=['','候选内的连续变化/候选间差异按中位数判定，检查间隔 5 次更新；这是一条预定的启发式规则。置换对照在同模型、同类别、同重复内打乱触发时间，保留完整时刻分布。每种策略与在相同更新处注入随机扰动的末端规则配对。','',str(hist),'',f"道路检查通过数：状态 {arm['mid']['valid']}、末端 {arm['terminal']['valid']}；耗尽数 {arm['mid']['caps']}、{arm['terminal']['caps']}。独立交付样本多样性 {arm['mid']['diversity']:.2f}、{arm['terminal']['diversity']:.2f}m。",'','高低转弯组的补充点估计与区间在 groups.json；不是另一次独立验证。真实路线只用于仿真/参考，不输入生成器。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
c=canvas.Canvas('thesis/figures/g11-trigger.pdf',pagesize=(454,173));c.setFont('Helvetica-Bold',10);c.drawString(30,152,'When did candidate motion become distinguishable?')
left,w=36,390;bar=12;peak=max(hist.values())
for k in range(15,101,5):
 x=left+(k-15)/85*w;v=hist.get(k,0);h=95*v/peak
 c.setFillColor(HexColor('#24756c'));c.rect(x-bar/2,38,bar,h,stroke=0,fill=1)
 c.setFont('Helvetica',7);c.setFillColor(HexColor('#163548'));c.drawCentredString(x,25,str(k))
 if v:c.drawCentredString(x,42+h,str(v))
c.setFont('Helvetica',9);c.drawCentredString(227,8,'Reverse update (100 denotes fallback to terminal selection)');c.save()
