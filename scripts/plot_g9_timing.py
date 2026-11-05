import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
c=canvas.Canvas('thesis/figures/g9-timing.pdf',pagesize=(480,185))
for j,(metric,title,unit) in enumerate([(2,'Travel distance','m'),(0,'Speed','m/s')]):
 x=63+j*235;w=148;low,hi=((-28,30) if j==0 else (-.30,.25))
 axis=lambda v:x+w*(v-low)/(hi-low)
 c.setFillColor(HexColor('#18394a'));c.setFont('Helvetica-Bold',10);c.drawString(x,168,title+' discrepancy ('+unit+')')
 c.setStrokeColor(HexColor('#9eacb7'));c.setDash(2,2);c.line(axis(0),28,axis(0),155);c.setDash()
 for k,st in enumerate([40,76,91]):
  d=json.loads(Path(f'docs/results/g9-timing/report{st}.json').read_text())
  for l,(g,color) in enumerate([('low','#a65027'),('high','#267499')]):
   a=d['mid-terminal_'+g];v=a['point'][metric];left,right=a['CI'][metric];y=149-42*k-14*l
   c.setStrokeColor(HexColor(color));c.setFillColor(HexColor(color));c.setLineWidth(1.4)
   c.line(axis(left),y,axis(right),y);c.circle(axis(v),y,2.8,stroke=0,fill=1)
   c.setFont('Helvetica',8);c.drawRightString(x-5,y-2,f'{st} / '+g)
 for t in ((-20,0,20) if j==0 else (-.2,0,.2)):
  c.setFillColor(HexColor('#18394a'));c.drawCentredString(axis(t),18,str(t))
c.save()
