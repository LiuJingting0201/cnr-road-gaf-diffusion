"""Four fixed synthesis draws for the first validation request of each class."""
import csv
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

NAMES = ["pedestrian","bus","motorcycle","private_car","bycicle","taxi_or_uber","truck","delivery"]
DISPLAY = ["Pedestrian","Bus","Motorcycle","Private car","Bicycle","Taxi","Truck","Delivery vehicle"]


def figure(source, output):
    with ZipFile(source) as archive:
        meta = json.loads(archive.read("conditions.json"))
        first = {}
        for item in meta:
            if item["model_seed"] == 42:
                first.setdefault(item["mode"], item["request"])
        tracks = {r["id"]:[] for r in meta if r["model_seed"]==42 and r["request"]==first[r["mode"]]}
        with archive.open("trajectories.csv") as binary:
            for row in csv.DictReader(io.TextIOWrapper(binary)):
                if row["id"] in tracks:
                    tracks[row["id"]].append([float(row["x"]),float(row["y"])])
    curves = {label:[] for label in NAMES}
    for record in meta:
        if record["id"] not in tracks:
            continue
        pts = np.array(tracks[record["id"]])
        assert pts.shape == (512,2)
        dist = np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
        curves[record["mode"]].append(dist)
    ceiling = max(500,500*np.ceil(max(s[-1] for group in curves.values() for s in group)/500))
    c = canvas.Canvas(str(output),pagesize=(490,285))
    dark = HexColor("#23414b")
    muted = HexColor("#546e76")
    teal = HexColor("#137e79")
    c.setFillColor(dark);c.setFont("Helvetica-Bold",13)
    c.drawString(5,262,"How far do four generated participants travel on the same path?")
    c.setFont("Helvetica",8);c.setFillColor(muted)
    c.drawString(5,247,"Each panel fixes one request. Both axes have the same scale in all eight panels.")
    for index,label in enumerate(NAMES):
        x = 18+index%4*120
        y = 151-index//4*109
        c.setFillColor(dark);c.setFont("Helvetica-Bold",9);c.drawString(x,y+71,DISPLAY[index])
        c.setStrokeColor(HexColor("#c7d6d6"));c.setLineWidth(.5)
        c.line(x,y,x+96,y);c.line(x,y,x,y+58)
        c.setFont("Helvetica",6.8);c.setFillColor(muted)
        c.drawString(x+2,y+55,f"{ceiling:.0f} m")
        c.drawString(x,y-10,"0");c.drawRightString(x+96,y-10,"128 s")
        for i,curve in enumerate(curves[label]):
            c.setStrokeColor(teal);c.setLineWidth(.85)
            c.setDash([2,1] if i%2 else [])
            pts=c.beginPath();pts.moveTo(x,y)
            for t in list(range(1,511,3))+[511]:
                pts.lineTo(x+t/511*96,y+curve[t]/ceiling*58)
            c.drawPath(pts)
        c.setDash([])
    c.setFillColor(muted);c.setFont("Helvetica",7)
    c.drawString(5,9,"First supported validation request per class, fixed model seed 42, all four draws shown.")
    c.save()


if __name__=="__main__":
    figure(Path(sys.argv[1]),Path(sys.argv[2]))
