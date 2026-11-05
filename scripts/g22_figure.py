"""Illustrate the fixed, actual OD sample delivered by the portable model."""
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor


def plot(csvfile,metadata,net,out):
    rows={}
    with csvfile.open() as src:
        for r in csv.DictReader(src):
            rows.setdefault(r["id"],[]).append([float(r[k]) for k in ("time","x","y")])
    tracks=[np.array(v) for v in rows.values()]
    info=json.loads(metadata.read_text())
    c=canvas.Canvas(str(out),pagesize=(500,225))
    dark="#173b46"; teal="#087e78"; orange="#c26544"
    def txt(x,y,s,n=8,color=dark,bold=False):
        c.setFillColor(HexColor(color));c.setFont("Helvetica-Bold" if bold else "Helvetica",n);c.drawString(x,y,s)
    def line(xy,color,width):
        c.setStrokeColor(HexColor(color));c.setLineWidth(width)
        p=c.beginPath();p.moveTo(*xy[0])
        for point in xy[1:]:p.lineTo(*point)
        c.drawPath(p)
    txt(9,205,"One planned bus route, four sampled arrivals",15,bold=True)
    txt(9,190,"All four draws start and finish at the same locations. Their travel times are learned.",8)
    x=tracks[0][:,1:]
    low=x.min(0);high=x.max(0);center=(low+high)/2
    scale=min(180/max(1,high[0]-low[0]),133/max(1,high[1]-low[1]))
    trans=lambda xy:(xy-center)*scale+np.array([106,98])
    c.saveState()
    box=c.beginPath();box.rect(7,21,201,151);c.clipPath(box,stroke=0,fill=0)
    for edge in ET.parse(net).getroot().findall("edge"):
        if edge.get("function")=="internal":continue
        lane=edge.find("lane")
        if lane is None:continue
        pts=np.array([[float(a) for a in pair.split(",")[:2]] for pair in lane.get("shape","").split()])
        if len(pts)>1 and (pts.max(0)>=low-70).all() and (pts.min(0)<=high+70).all():
            line(trans(pts),"#dce4e4",.55)
    line(trans(x[::2]),teal,1.5)
    for point,col in [(x[0],dark),(x[-1],orange)]:
        a,b=trans(point);c.setFillColor(HexColor(col));c.circle(a,b,3,stroke=0,fill=1)
    c.restoreState()
    txt(9,6,"Start",7,dark);txt(41,6,"Destination",7,orange)
    txt(223,164,"Distance along the trip (km)",8,bold=True)
    left,base,width,height=224,40,253,110
    xmax=330.;ymax=2.5
    line(np.array([[left,base+height],[left,base],[left+width,base]]),"#a6b7b8",.6)
    for tick in (0,1,2):
        y=base+height*tick/ymax
        txt(left-13,y-2,str(tick),7)
    for tick in (0,60,120,180,240,300):
        txt(left+tick/xmax*width-5,base-12,str(tick),7)
    c.setDash([2,2]);line(np.array([[left+127.75/xmax*width,base],[left+127.75/xmax*width,base+height]]),"#afbbbb",.7);c.setDash([])
    txt(left+127.75/xmax*width-18,base+height+3,"127.75 s",6.8,"#667878")
    for k,track in enumerate(tracks):
        dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(track[:,1:],axis=0),axis=1))]/1000
        pts=np.c_[left+width*track[:,0]/xmax,base+height*dist/ymax]
        color=[teal,orange,"#436c89","#829348"][k]
        line(pts,color,1.25)
        c.setFillColor(HexColor(color));c.circle(*pts[-1],2.2,stroke=0,fill=1)
    txt(407,6,"Time since departure (s)",7)
    c.save()


if __name__=="__main__":plot(*map(Path,sys.argv[1:]))
