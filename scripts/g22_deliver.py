"""Export complete 0.25-s trajectories and a runnable OD-conditioned model."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED,ZipFile
import numpy as np
from g16_data import NAMES
from g19_gaf import original_metadata
from g22_diffusion import ROOT,load,decode


def export(d,filename="complete-trip-examples.zip"):
    out=ROOT/filename
    manifest=[]
    with ZipFile(out,"w",compression=ZIP_DEFLATED,compresslevel=4) as z:
        with z.open("trajectories.csv","w") as binary, io.TextIOWrapper(binary,newline="") as f:
            w=csv.writer(f);w.writerow(["time","id","x","y","type"])
            # All four draws of seed42 on the first two routes of each class in every evaluation episode.
            samples=np.load(ROOT/"model-42/samples.npz")
            picks=np.load(ROOT/"model-42/native-picks.npz")
            assert np.array_equal(samples["ids"],d["ids"]) and np.array_equal(picks["ids"],d["ids"])
            counts={}
            for j,ident in enumerate(d["ids"]):
                lab=int(d["labels"][j]);key=(str(ident).split("_")[0],lab)
                if counts.get(key,0)>=2:continue
                counts[key]=counts.get(key,0)+1
                for k in range(4):
                    T=float(samples["duration"][j,k])
                    t,xy=decode(d["paths"][j],T,samples["progress"][j,k])
                    sid=f"G22_{ident}_{k}"
                    w.writerows((f"{tick:.8f}",sid,f"{x:.4f}",f"{y:.4f}",NAMES[lab]) for tick,(x,y) in zip(t,xy))
                    manifest.append(dict(id=sid,route=str(ident),slot=k,duration_s=T,rows=len(t),endpoint=xy[-1].tolist(),
                                         gaf=bool(picks["gaf"][j]==k),temporal=bool(picks["temporal"][j]==k),
                                         score_scope="both complete quarter-second series"))
        z.writestr("manifest.json",json.dumps(manifest))
        z.writestr("README.txt","Full journeys, grouped by participant. Time in seconds from departure, XY in SUMO metres. Each final time is the sampled arrival. All four draws from the predeclared illustrative requests, not the full evaluation pool. Separate GAF and temporal choices are annotations.\n")
    with ZipFile(out) as z:
        assert z.testzip() is None
        with z.open("trajectories.csv") as f:
            assert sum(1 for _ in f)==1+sum(r["rows"] for r in manifest)
    return out


def kit(d):
    work=ROOT/"kit";assets=work/"assets";assets.mkdir(parents=True,exist_ok=True)
    source=ROOT/"model-42"
    for file,name in [(source/"model.pt","diffusion.pt"),(source/"stats.json","stats.json"),
                      (ROOT/"cnn-native-gaf/model.pt","gaf.pt"),(ROOT/"cnn-native-full/model.pt","temporal.pt"),
                      (Path("data/maps/most.net.xml"),"most.net.xml")]:
        shutil.copyfile(file,assets/name)
    m=original_metadata()
    (assets/"encoding.json").write_text(json.dumps({k:m[k] for k in ("xy_min","xy_max","hilbert_eps","channel_min","channel_max")}))
    (work/"scripts").mkdir(exist_ok=True)
    for name in ("g22_infer","g22_temporal","g22_route","g22_sim","g22_diffusion","g19_gaf","g21_train","g16_data","g12_fit"):
        f=next(p/f"{name}.py" for p in map(Path,["scripts","../g19-job/scripts","../g21-job/scripts","../g17-job/scripts","../scripts"]) if (p/f"{name}.py").exists())
        shutil.copyfile(f,work/f"scripts/{name}.py")
    for stem in ("src/cnr","../vendor/original_cnr"):
        target=work/("src/cnr" if stem.startswith("src") else "vendor/original_cnr")
        shutil.copytree(stem,target,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    i=np.flatnonzero(d["labels"]==1)[0]
    np.savez(work/"example-route.npz",path=d["paths"][i],length=d["length"][i],label=1)
    cmd=["scripts/g22_infer.py","--assets","assets","--mode","bus","--seed","22042","--route","example-route.npz","--out","example.csv"]
    (work/"example.csv").unlink(missing_ok=True)
    ans=subprocess.run([sys.executable,*cmd],cwd=work,env={**os.environ,"PYTHONPATH":"src"},capture_output=True,text=True)
    if ans.returncode:raise RuntimeError(ans.stderr)
    meta=json.loads(ans.stdout)
    (work/"example-result.json").write_text(json.dumps(meta,indent=2))
    cmd[-1]="new-trips.csv"
    (work/"README.md").write_text("# 完整行程扩散生成与整程 CNN\n\nCUDA 版 PyTorch、torchvision、numpy、pandas、scipy、shapely>=2、pyts。解压后运行：\n\n"
        +"    PYTHONPATH=src python "+" ".join(cmd)+"\n\n也可把 --route example-route.npz 替换成 --start X Y --destination X Y。坐标为 MoST 米制 XY。端点须位于目标类别许可的道路2米以内；行人仅支持同一条人行道。"
        "一次输出四条从出发到终点的独立样本，时间步长0.25秒，最后一行是实际生成的到达时刻。轨迹长度随行程而变。"
        "同时打印两个完整行程识别器的评分：自训原 Hilbert–GAF ResNet18，以及运动时序 CNN。二者单独输出，不把分类分数作真实性证明。"
        "example.csv 与 example-result.json 为打包后实际运行的结果。模型种子42，地图来源 MoSTScenario。")
    out=ROOT/"complete-trip-diffusion-and-cnns.zip"
    with ZipFile(out,"w",compression=ZIP_DEFLATED,compresslevel=3) as z:
        for f in sorted(work.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:z.write(f,f.relative_to(work))
    return out


if __name__=="__main__":
    ev=load("evaluation")
    packages=[export(ev),kit(ev)]
    (ROOT/"delivery.json").write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in packages},indent=2))
