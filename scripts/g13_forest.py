"""Two physically meaningful endpoints on separate scales, all class contrasts."""
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from g13_figures import label,line,ink,teal,muted,pale,fig,base
from reportlab.lib.colors import HexColor
d=json.loads((base/'contrasts.json').read_text())
c=canvas.Canvas(str(fig/'class-effects.pdf'),pagesize=(490,206))
label(c,8,188,'Shared-road effect of changing the private-car label',13,bold=True)
for j,(key,title,lo,hi,scale,ticks) in enumerate([('travel','Trip distance (m)',-720,240,1,[-600,-300,0,200]),('stop','Near-stop fraction (pp)',-9,18,100,[-5,0,5,10,15])]):
    left=90+j*206;width=172
    ax=lambda v:left+(v-lo)/(hi-lo)*width
    label(c,left,157,title,10,bold=True);line(c,[ax(0),ax(0)],[30,146],muted,.8,(2,2))
    for n,(cl,name) in enumerate([('bus','Bus'),('motorcycle','Motorcycle'),('bycicle','Bicycle'),('taxi_or_uber','Taxi / Uber')]):
        row=d[cl];v=row[key];y=134-n*27
        if j==0:label(c,8,y-3,f"{name} ({row['routes']})",9)
        a,b=[z*scale for z in v['ci']];line(c,[ax(a),ax(b)],[y,y],teal,2)
        c.setFillColor(HexColor(teal));c.circle(ax(v['difference']*scale),y,3.1,fill=1,stroke=0)
    for t in ticks:
        label(c,ax(t)-5,28,str(t),8,muted)
label(c,8,8,'Points: paired means. Bars: 95% episode-bootstrap intervals. Parentheses: compatible routes.',8,muted)
c.save()
