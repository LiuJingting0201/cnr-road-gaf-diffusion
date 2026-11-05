import json
import os
import hashlib
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile
import numpy as np
from g19_gaf import original_metadata
from g21_train import ROOT

r=json.loads((ROOT/"results.json").read_text())
kit=ROOT/"kit";assets=kit/"assets";assets.mkdir(parents=True,exist_ok=True)
cfg=r["selection"]
(assets/"selection.json").write_text(json.dumps(cfg))
meta=original_metadata()
if cfg["spatial"]=="fine":meta=dict(meta,hilbert_eps=meta["hilbert_eps"]/16)
(assets/"encoding.json").write_text(json.dumps({k:meta[k] for k in ["xy_min","xy_max","hilbert_eps","channel_min","channel_max"]}))
for arm in ("original","physical","temporal"):
    for seed in (42,43):
        directory=Path("g21-fine") if arm=="original" and cfg["spatial"]=="fine" else ROOT
        shutil.copyfile(directory/f"{arm}-{seed}/model.pt",assets/f"{arm}-{seed}.pt")
(kit/"scripts").mkdir(exist_ok=True)
for name in ["g21_infer","g21_train","g19_gaf","g16_data","g12_fit"]:
    f=next(p/f"{name}.py" for p in map(Path,["scripts","../g17-job/scripts","../g19-job/scripts","../scripts"]) if (p/f"{name}.py").exists())
    shutil.copyfile(f,kit/f"scripts/{name}.py")
for src,dst in [("src/cnr","src/cnr"),("../vendor/original_cnr","vendor/original_cnr")]:
    shutil.copytree(src,kit/dst,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
# Only use a previously generated diffusion candidate for the example.
with np.load("../g17-job/g17-data/C-42/g15.npz") as q:
    x=q["xy"][0,0]
with (kit/"example.csv").open("w") as f:
    w=csv.writer(f);w.writerow(["time","id","x","y","type"])
    w.writerows((j*.25,"example",float(point[0]),float(point[1]),"unknown") for j,point in enumerate(x))
(kit/"example-output.jsonl").unlink(missing_ok=True)
args=["scripts/g21_infer.py","--assets","assets","--csv","example.csv","--out","example-output.jsonl"]
subprocess.run([sys.executable,*args],cwd=kit,env={**os.environ,"PYTHONPATH":"src"},check=True)
args[-1]="new-predictions.jsonl"
(kit/"README.md").write_text("# 八类短窗 CNN\n\nCUDA/PyTorch、torchvision、numpy、pandas、scipy、shapely>=2、pyts。\n\n    PYTHONPATH=src python "+" ".join(args)+"\n\n只适用于512点的0.25秒窗口。original 为原空间 Hilbert/GAF 变换的内部监测胜出网格，physical 为物理速度和加速度的角场图像 CNN，temporal 为运动时序卷积和全局统计。每种均为两个种子的概率均值，fusion 权重只用内部独立场次选择。")
out=ROOT/"eight-class-short-window-cnns.zip"
with ZipFile(out,"w",compression=ZIP_DEFLATED,compresslevel=2) as z:
    for f in sorted(kit.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:z.write(f,f.relative_to(kit))
with ZipFile(out) as z:assert z.testzip() is None
(ROOT/"delivery.json").write_text(json.dumps({out.name:hashlib.sha256(out.read_bytes()).hexdigest()}))
