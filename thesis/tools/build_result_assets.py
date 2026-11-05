"""Regenerate thesis tables and case plots from the archived evaluation and CSV export."""
import argparse
import csv
import gzip
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from reportlab.lib.colors import Color, HexColor
from reportlab.pdfgen import canvas

parser = argparse.ArgumentParser()
parser.add_argument("results", type=Path, help="experiment repository's docs/results")
parser.add_argument("archive", type=Path, help="final_test_synthetic_trajectories.zip")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
fig = root / "figures"
tab = root / "tables"
fig.mkdir(exist_ok=True)
tab.mkdir(exist_ok=True)
def load(p):
    return json.loads((args.results / p).read_text())
families = ["diffusion", "empirical"]
pools = {f: [load(f"final-{f}-e7/seed{s}/results.json") for s in [42, 43, 44]] for f in families}
aux = load("final-auxiliary-e7/summary.json")
down = load("final-downstream-e7/summary.json")
manifest = load("final-test-e7/map_examples.json")["examples"]
classes = ["pedestrian", "bus", "motorcycle", "private_car", "bycicle", "taxi_or_uber"]
short = ["Ped.", "Bus", "Moto.", "Car", "Bike", "Taxi"]
names = ["Pedestrian", "Bus", "Motorcycle", "Private car", "Bicycle", "Taxi or Uber"]
groups = ["real_only", "repeat", "timewarp", "synthetic_random", "synthetic_gaf", "synthetic_tcn"]
gnames = ["Observed only", "Repeat", "Time warp", "Random synthetic", "GAF synthetic", "TCN synthetic"]
variants = ["all_candidates", "random", "gaf", "tcn", "shuffled_gaf"]
vnames = ["All candidates", "Random", "GAF", "TCN", "Shuffled GAF"]
def table(caption, label, cols, header, rows):
    line = lambda row: " & ".join(map(str, row)) + r"\\"
    return (r"\tab{" + caption + "}{" + label + r"}{\begin{tabular}{" + cols + r"}\toprule"+"\n"
            + line(header) + r"\midrule" + "\n" + "\n".join(map(line, rows))
            + r"\bottomrule\end{tabular}}"+"\n")
metrics = ["speed_w1_mps","acceleration_w1_mps2","travel_w1_m","stop_fraction","real_stop_fraction"]
counts = [0,7,4,33,7,2]
notes = {
"bus":"Buses supply seven reference conditions. Empirical transfer reproduces the observed stop fraction more closely in this class, yet its travel-distance discrepancy is larger. A closer match in the total time spent nearly stationary does not specify where those stationary intervals occur or how far the vehicle travels between them.",
"motorcycle":"The four motorcycle references spend roughly a quarter of sampled intervals below the stop threshold. Both generators spend only about seven percent. Transporting observed profiles from the training library therefore leaves a substantial mismatch even before a learned motion distribution is used. Whether the gap reflects different route mixes or insufficient training coverage requires independent conditions.",
"private_car":"Private cars account for 33 of the 53 requests. Their stop fraction is closely matched by diffusion, and their travel discrepancy is smaller than with empirical transfer. The empirical baseline, however, gives a much smaller acceleration discrepancy for this class. These results distinguish agreement on total distance from agreement on changes in speed.",
"bycicle":"The seven observed bicycle windows spend little sampled time below the stop threshold. The two generators produce similar stop fractions of about three percent. Both acceleration discrepancies are comparatively large. Since acceleration here differentiates planar speed, quarter-second variations in the decoded geometry and timing can affect this metric even on a connected route.",
"taxi_or_uber":"Only two taxi requests supply the reference distribution. In this class empirical transfer has smaller speed and travel discrepancies than diffusion, while both generators spend considerably less time nearly stationary than the two observed vehicles. The direction of the class-specific result is informative, but two vehicles provide very little information about the broader taxi population."}
wanted = {i for e in manifest for i in e["candidate_indices"]}
tracks = {i: [] for i in wanted}
with zipfile.ZipFile(args.archive) as z, z.open("diffusion_all_candidates.csv.gz") as f, gzip.GzipFile(fileobj=f) as gz:
    for row in csv.DictReader(io.TextIOWrapper(gz)):
        ident = row["id"]
        if ident.startswith("syn_all_candidates_s42_i"):
            i = int(ident.rsplit("i",1)[1])
            if i in wanted:
                tracks[i].append((float(row["x"]),float(row["y"])))
assert all(len(p)==512 for p in tracks.values())
def case_plot(e):
    idx = e["candidate_indices"]
    pick = e["selected"]["gaf"]
    c = canvas.Canvas(str(fig / f"case-{e['class']}.pdf"),pagesize=(460,232))
    for k,(bottom,height,title) in enumerate([(137,71,"Speed (m/s)"),(28,71,"Travel (m)")]):
        curves={}
        for i in idx:
            d=np.linalg.norm(np.diff(np.asarray(tracks[i]),axis=0),axis=1)
            curves[i] = d*4 if k==0 else np.r_[0,d.cumsum()]
        ymax=max(float(a.max()) for a in curves.values())
        c.setFont("Helvetica-Bold",9);c.setFillColor(HexColor("#18364b"));c.drawString(42,bottom+height+8,title)
        for v in (0,.5,1):
            y=bottom+height*v;c.setStrokeColor(HexColor("#dce2e5"));c.line(42,y,446,y)
            c.setFont("Helvetica",8);c.drawRightString(37,y-2,f"{v*ymax:.1f}")
        for i in [j for j in idx if j!=pick]+[pick]:
            vals=curves[i];poly=c.beginPath()
            for j,val in enumerate(vals):
                px=42+404*(j+(k==0))/511;py=bottom+height*val/max(ymax,1e-8)
                (poly.moveTo if j==0 else poly.lineTo)(px,py)
            c.setLineWidth(1.35 if i==pick else .55);c.setStrokeColor(HexColor("#b25320" if i==pick else "#9eb5c1"));c.drawPath(poly)
        for t in (0,32,64,96,127.75):
            c.setFont("Helvetica",8);c.drawCentredString(42+404*t/127.75,bottom-11,f"{t:g}")
    c.drawCentredString(244,3,"Time from requested start (s)")
    c.save()
pages=[]
for e in manifest:
    cl=e["class"];j=classes.index(cl); name=names[j]; case_plot(e)
    stats={f: np.mean([[d["rows"][0]["per_class"][cl][k] for k in metrics] for d in pools[f]],axis=0) for f in families}
    rows=[(f.capitalize(),*(f"{v:.3f}" for v in a[:3]),f"{100*a[3]:.1f}") for f,a in stats.items()]
    idx=e["selected"]
    pages.append(r"\clearpage\section{"+name+"}\n"+notes[cl]+"\n\n"
       +f"The fixed request contains eight candidates. The random, image and coordinate rules returned candidate indices {idx['random']}, {idx['gaf']} and {idx['tcn']}, respectively. The observed class stop fraction across the {counts[j]} references is {100*stats['diffusion'][4]:.1f}"+r"\%."+"\n\n"
       +r"\begin{center}\includegraphics[width=.98\linewidth]{figures/case-"+cl+r".pdf}\end{center}"+"\n"
       +r"\noindent The highlighted line is the GAF-selected trace. The seven other trajectories are shown in blue-gray. Axes are scaled separately for each class."+"\n"
       +table("All-candidate "+name.lower()+r" results. Units are m/s, m/s$^2$, m and percent, respectively.","case"+str(j),"lrrrr",["Motion","Speed "+r"$\W$","Accel. "+r"$\W$","Travel "+r"$\W$",r"Stop \%"],rows)+r"\FloatBarrier"+"\n")
(tab/"case-pages.tex").write_text("\n".join(pages))
rows=[]
for family in families:
    for n,(key,title) in enumerate(zip(variants,vnames)):
        ds=[next(row for row in d["rows"] if row["variant"]==key) for d in pools[family]]
        ks=metrics[:3]+["gaf_audit_accuracy","tcn_audit_accuracy"]
        means=[np.mean([r[k] for r in ds]) * (100 if k.endswith("accuracy") else 1) for k in ks]
        rows.append((family.capitalize() if n==0 else "", title,*[f"{a:.2f}" for a in means]))
(tab/"selection.tex").write_text(table("Locked five-class macro results by motion model and selection rule. Units are m/s, m/s$^2$, m and percent.","fullselection","llrrrrr",["Motion","Rule","Speed","Accel.","Travel","GAF","TCN"],rows))
rows=[]
for key,title in zip(groups,gnames):
    ds=[down["models"][l]["synthetic_gaf"]["paired_delta_f1"][key] for l in ["tcn","gru"]]
    def interval(a):
        lo,hi=a["percentile95"]
        return f'{100*a["point"]:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]'
    rows.append((title,interval(ds[0]),interval(ds[1])))
(tab/"downstream.tex").write_text(table("GAF-selected augmentation minus each comparator in macro-F1 percentage points, with 95\\% paired intervals.","contrasts","lrr",["Comparator","TCN","GRU"],rows))
rows=[]
for family in ["tcn","gru"]:
    for i,(key,title) in enumerate(zip(groups,gnames)):
        rec=np.mean([m["per_class_recall"] for m in down["models"][family][key]["individual"]],axis=0)
        rows.append((family.upper() if i==0 else "",title,*[f"{100*v:.1f}" for v in rec]))
(tab/"recall.tex").write_text(table("Downstream per-class recall on the observed test set, in percent. Entries average the three learner initializations.","recall","llrrrrrr",["Model","Group",*short],rows))
rows=[]
for i,n in enumerate(names):
    vals=[]
    for f in ["gaf","tcn"]:
        for metric in ["per_class_recall","per_class_f1"]:
            vals.append(100*np.mean([m[metric][i] for m in aux["models"][f]["individual"]]))
    rows.append((n,aux["class_counts"][i],*[f"{v:.1f}" for v in vals]))
(tab/"auxiliary.tex").write_text(table("Observed-window recognition by class. Recall and F1 are given as percentages.","auxclasses","lrrrrr",["Class","IDs","GAF recall","GAF F1","TCN recall","TCN F1"],rows))
c=canvas.Canvas(str(fig/"confusion.pdf"), pagesize=(530,260))
for k,f in enumerate(["gaf","tcn"]):
    cm=np.mean([np.asarray(m["confusion"])/np.asarray(aux["class_counts"])[:,None] for m in aux["models"][f]["individual"]],axis=0)
    x0=30+k*265;cell=34;top=222
    c.setFont("Helvetica-Bold",11);c.drawString(x0,245,f.upper()+"   |   observed test windows")
    for i in range(6):
        c.setFillColor(HexColor("#233d4d"));c.setFont("Helvetica",7)
        c.drawCentredString(x0+(i+.5)*cell,top+3,short[i]);c.drawRightString(x0-3,top-(i+.55)*cell,short[i])
        for j in range(6):
            v=cm[i,j];c.setFillColor(Color(1-.87*v,1-.64*v,1-.58*v));c.rect(x0+j*cell,top-(i+1)*cell,cell,cell,fill=1,stroke=0)
            c.setFillColor(Color(1,1,1) if v>.65 else Color(.1,.16,.19));c.drawCentredString(x0+(j+.5)*cell,top-(i+.55)*cell,f"{100*v:.0f}")
c.save()
