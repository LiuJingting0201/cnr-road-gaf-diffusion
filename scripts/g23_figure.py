"""Show how two actual full-trip candidates become two speed angular images."""
import csv
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor


def make(src,out):
    rows=defaultdict(list)
    with src.open() as f:
        for row in csv.DictReader(f):rows[row["id"]].append([float(row[k]) for k in ("time","x","y")])
    c=canvas.Canvas(str(out),pagesize=(500,222))
    dark=HexColor("#173b46")
    def text(x,y,s,n=8,b=False):
        c.setFillColor(dark);c.setFont("Helvetica-Bold" if b else "Helvetica",n);c.drawString(x,y,s)
    text(7,201,"The same bus route, seen as two whole-trip speed images",14,True)
    text(7,185,"Left: speed in physical time. Right: speed GASF; both axes run from departure to arrival.",8)
    for i,records in enumerate(list(rows.values())[:2]):
        ar=np.array(records);t=ar[:,0]
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(ar[:,1:],axis=0),axis=1))]
        v=np.diff(np.interp(np.arange(int(t[-1])+1),t,arc))
        u=np.clip(np.interp(np.linspace(0,len(v)-1,128),np.arange(len(v)),v)/35,0,1)
        b=np.sqrt(1-u*u)
        g=np.outer(u,u)-np.outer(b,b)
        f=np.clip((g+1)/.65,0,1)[:,:,None]
        colors=(1-f)*np.array([10,40,61])+f*np.array([246,194,105])
        y=107-i*85
        c.drawImage(ImageReader(Image.fromarray(colors.astype("uint8"))),384,y-6,67,67)
        text(7,y+46,f"Draw {i+1}",9,True);text(7,y+32,f"{t[-1]:.0f} s",8)
        x0=64;width=288;height=50
        c.setStrokeColor(HexColor("#bac9c9"));c.setLineWidth(.6);c.line(x0,y,x0+width,y);c.line(x0,y,x0,y+height)
        for tick in (0,60,120,180,240):text(x0+tick/290*width-3,y-9,str(tick),6)
        for tick in (0,10,20):text(x0-14,y+height*tick/22-2,str(tick),6)
        c.setStrokeColor(HexColor("#087e78" if i==0 else "#b45837"));c.setLineWidth(1)
        p=c.beginPath()
        for j,s in enumerate(v):
            xy=x0+j/290*width,y+height*s/22
            if j==0:p.moveTo(*xy)
            else:p.lineTo(*xy)
        c.drawPath(p)
    text(72,4,"Time since departure (seconds); vertical scale in m/s.",7)
    c.save()


if __name__=="__main__":make(*map(Path,sys.argv[1:]))
