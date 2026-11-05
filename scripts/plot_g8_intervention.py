"""Within-route label intervention. One line per route, averaged over slots and models."""
import json
from pathlib import Path
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
root=Path('docs/results/g8-development')
ds=[json.loads((root/p).read_text())['rows'] for p in ['intervention.json','intervention-43.json','intervention-44.json']]
c=canvas.Canvas('thesis/figures/g8-switch.pdf',pagesize=(488,208))
for j,(key,title,mult) in enumerate([('travel','Travel distance (m)',1),('stop','Time below 0.1 m/s (%)',100)]):
    a=np.asarray([[[np.mean(r[t][key])*mult for t in ['bus','private_car']] for r in rows] for rows in ds]).mean(0)
    x=42+238*j; y=32; h=145; top=np.ceil(a.max()/ (500 if j==0 else 10))*(500 if j==0 else 10)
    for frac in [0,.5,1]:
        c.setStrokeColor(HexColor('#dce3e8'));c.line(x-3,y+h*frac,x+136,y+h*frac)
        c.setFillColor(HexColor('#273d49'));c.setFont('Helvetica',8);c.drawRightString(x-6,y+h*frac-2,f'{top*frac:g}')
    for pair in a:
        c.setStrokeColor(HexColor('#9DAEBA'));c.setLineWidth(.85);c.line(x,y+h*pair[0]/top,x+130,y+h*pair[1]/top)
    m=a.mean(0);c.setStrokeColor(HexColor('#b14c19'));c.setLineWidth(3);c.line(x,y+h*m[0]/top,x+130,y+h*m[1]/top)
    c.setFont('Helvetica-Bold',10);c.drawString(x,192,title);c.setFont('Helvetica',9)
    c.drawCentredString(x,y-15,'Bus');c.drawCentredString(x+130,y-15,'Private car')
c.save()
