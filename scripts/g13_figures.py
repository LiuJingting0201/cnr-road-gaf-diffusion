"""Vector thesis plates from a rule-selected replay and all eligible routes."""
import json,sys
from pathlib import Path
import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from xml.etree.ElementTree import iterparse
base=Path('docs/results/g13-shared-road');fig=Path('thesis/figures')
ink='#1b3545';muted='#536b75';teal='#00796f';rust='#b45427';pale='#dbe4e6'
def label(c,x,y,s,size=9,color=ink,bold=False):
    c.setFont('Helvetica-Bold' if bold else 'Helvetica',size);c.setFillColor(HexColor(color));c.drawString(x,y,s)
def line(c,x,y,color,width=1,dash=()):
    c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.setDash(dash);p=c.beginPath()
    for i,(a,b) in enumerate(zip(x,y)):(p.lineTo if i else p.moveTo)(a,b)
    c.drawPath(p);c.setDash()
def case():
    ex=json.loads((base/'example.json').read_text());x=np.array(ex['xy']);road=np.array(ex['route'])
    v=np.linalg.norm(np.diff(x,axis=1),axis=-1)/.25;dist=np.c_[np.zeros(2),np.cumsum(v*.25,axis=1)]
    c=canvas.Canvas(str(fig/'shared-road-case.pdf'),pagesize=(490,324))
    label(c,15,303,'A   Same starting point and planned road',11,bold=True)
    label(c,260,303,'B   Motion over 127.75 seconds',11,bold=True)
    # Equal-distance map axes; clipping prevents distant context changing scale.
    lo=x.min(axis=(0,1));hi=x.max(axis=(0,1));ctr=(hi+lo)/2
    span=max((hi-lo).max()*1.23,60);left,bottom,w=15,62,226
    trans=lambda coords:(coords-ctr)/span*w+[left+w/2,bottom+w/2]
    c.saveState();clip=c.beginPath();clip.rect(left,bottom,w,w);c.clipPath(clip,stroke=0)
    for _,el in iterparse(sys.argv[1],events=('end',)):
        if el.tag=='lane' and el.get('shape'):
            pts=np.array([list(map(float,pt.split(',')[:2])) for pt in el.get('shape').split()])
            if ((pts.max(0)>=ctr-span*.52)&(pts.min(0)<=ctr+span*.52)).all():
                a=trans(pts);line(c,*a.T,pale,.52)
        el.clear()
    line(c,*trans(road).T,muted,1.8)
    for arr,col,t,dx in zip(x,[teal,rust],['Bus','Car'],[0,1]):
        a=trans(arr);line(c,*a.T,col,1.4,(3,2) if dx else ())
        end=a[-1];c.setFillColor(HexColor(col));c.circle(*end,3.2,fill=1,stroke=0)
    a=trans(x[0,0:1])[0];c.setFillColor(HexColor(ink));c.rect(a[0]-2.4,a[1]-2.4,4.8,4.8,fill=1,stroke=0);c.restoreState()
    bar=10**np.floor(np.log10(span*.23));bar*=2 if 2*bar<=span*.23 else 1
    line(c,[left+4,left+4+bar*w/span],[bottom+4]*2,ink,1.4);label(c,left+4,bottom+9,f'{bar:g} m',7)
    for series,y,ht,title in [(v,194,72,'Speed (m/s)'),(dist,70,75,'Cumulative distance (m)')]:
        xmax=467;l=278;mx=max(1,series.max())
        label(c,260,y+ht+12,title,9,bold=True)
        for frac in (0,.5,1):
            line(c,[l,xmax],[y+frac*ht]*2,pale,.55)
            c.setFillColor(HexColor(muted));c.setFont('Helvetica',7);c.drawRightString(l-4,y+frac*ht-2,f'{mx*frac:.0f}')
        for arr,col in zip(series,[teal,rust]):
            coord=np.arange(len(arr))+(1 if series is v else 0)
            line(c,l+(xmax-l)*coord/511,y+ht*arr/mx,col,1)
        for pos,val in [(l,'0'),((l+xmax)/2,'64'),(xmax,'128 s')]:label(c,pos-2,y-10,val,7,muted)
    for xpos,name,col in [(15,'Bus',teal),(80,'Private car',rust)]:
        line(c,[xpos,xpos+14],[43,43],col,2);label(c,xpos+18,40,name,8)
    label(c,15,22,'Square: shared start. Dots: final positions. Thin gray lines: SUMO lane geometry.',8,muted)
    c.save()
def system():
    c=canvas.Canvas(str(fig/'generation-system.pdf'),pagesize=(460,205))
    label(c,12,185,'A trajectory is a route plus a time profile',15,bold=True)
    stages=[('01   Plan a road','Network, start, class','Sample permitted','lane successors.'),('02   Generate motion','Path-conditioned diffusion','Denoise 511 forward','distance increments.'),('03   Export coordinates','SUMO metres and seconds','Decode 512 positions','at 0.25 s intervals.')]
    for j,(a,b,d,e) in enumerate(stages):
        xx=12+j*151;c.setFillColor(HexColor('#f0f5f5'));c.roundRect(xx,86,141,84,5,stroke=0,fill=1)
        label(c,xx+8,148,a,10,ink,True);label(c,xx+8,132,b,8.2,teal);label(c,xx+8,110,d,8.5);label(c,xx+8,98,e,8.5)
    line(c,[12,455],[73,73],pale,1)
    label(c,12,53,'Optional class feedback',10,rust,True)
    label(c,12,36,'Decode candidates, map their ordered positions to Hilbert images and score the target class.',9)
    label(c,12,20,'A shared candidate budget isolates the effect of scoring. Held-out motion measures its value.',9,muted)
    c.save()
if __name__=='__main__':case();system()
