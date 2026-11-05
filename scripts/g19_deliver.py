"""Build an independently runnable seed-42 kit and export every eight-class candidate."""
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
from g16_data import NAMES,ORIGINAL
from g19_field import ROOT
from g19_train import OLD


def write_samples(q):
    meta=[]
    out=ROOT/"eight-class-trajectories.zip"
    with ZipFile(out,"w",compression=ZIP_DEFLATED) as z:
        with z.open("trajectories.csv","w") as raw, io.TextIOWrapper(raw,newline="") as f:
            w=csv.writer(f)
            w.writerow(("time","id","x","y","type"))
            for seed in (42,43):
                p=np.load(ROOT/f"road-{seed}/g15.npz")
                s=np.load(ROOT/f"road-{seed}/g15-picks.npz")
                assert np.array_equal(q["ids"],p["ids"]) and np.array_equal(p["ids"],s["ids"])
                for i,ident in enumerate(q["ids"]):
                    for k,row in enumerate(p["xy"][i]):
                        sid=f"G19_{seed}_{i}_{k}"
                        name=NAMES[q["labels"][i]]
                        w.writerows((t*.25,sid,f"{x:.3f}",f"{y:.3f}",name) for t,(x,y) in enumerate(row))
                        meta.append(dict(id=sid,request=str(ident),label=name,model_seed=seed,
                                         slot=k,gaf=bool(k==s["gaf"][i]),uniform=bool(k==s["uniform"][i]),
                                         capped=bool(p["caps"][i,k])))
        z.writestr("selection.json",json.dumps(meta))
    with ZipFile(out) as z:
        assert z.testzip() is None
        with z.open("trajectories.csv") as f:
            assert sum(1 for _ in f)==1+512*len(meta)
    return out


def kit(q):
    work=ROOT/"kit"
    assets=work/"assets"
    assets.mkdir(parents=True,exist_ok=True)
    for file,name in [(ROOT/"road-42/ema.pt","road.pt"),(ROOT/"gaf8.pt","gaf8.pt"),(ROOT/"factor.npy","factor.npy"),
                      (ORIGINAL/"route_prior.json","route_prior.json"),(Path("data/maps/most.net.xml"),"most.net.xml")]:
        shutil.copyfile(file,assets/name)
    config=json.loads((ORIGINAL/"dataset.json").read_text())
    (assets/"normalization.json").write_text(json.dumps({k:config[k] for k in ("mean","std")}))
    m=json.loads(Path("../runs/long-path-e3/seed42/config.json").read_text())["dataset"]["source_protocol"]
    (assets/"encoding.json").write_text(json.dumps({k:m[k] for k in ("xy_min","xy_max","hilbert_eps","channel_min","channel_max")}))
    (work/"scripts").mkdir(exist_ok=True)
    for name in ["g19_generate","g19_train","g19_field","g19_gaf","g16_data","g16_train","g12_fit"]:
        src=next(p for p in [Path("scripts"),Path("../g17-job/scripts"),Path("../scripts")] if (p/f"{name}.py").exists())
        shutil.copyfile(src/f"{name}.py",work/f"scripts/{name}.py")
    shutil.copytree("src/cnr",work/"src/cnr",dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copytree("../vendor/original_cnr",work/"vendor/original_cnr",dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    i=np.flatnonzero(q["labels"]==1)[0]
    x,y=q["start"][i]
    cli=["scripts/g19_generate.py","--assets","assets","--mode","bus","--x",str(float(x)),"--y",str(float(y)),
         "--seed","19042","--out","example.csv"]
    (work/"example.csv").unlink(missing_ok=True)
    result=subprocess.run([sys.executable,*cli],cwd=work,env={**os.environ,"PYTHONPATH":"src"},
                          text=True,capture_output=True)
    if result.returncode: raise RuntimeError(result.stderr)
    with (work/"example.csv").open() as f: assert sum(1 for _ in f)==2049
    (work/"example-selection.json").write_text(result.stdout)
    cli[-1]="new_sample.csv"
    (work/"README.md").write_text("# 八类路网扩散模型与原 Hilbert–GAF 反馈\n\n"
        "CUDA 环境安装 torch、torchvision、numpy、pandas、scipy、shapely>=2、pyts。解压后从本目录执行：\n\n"
        +"    PYTHONPATH=src python "+" ".join(cli)+"\n\n输出四条 512 点轨迹，终端同时打印 GAF 选出的 ID、编码是否成功及路径耗尽标记。"
        "类名："+", ".join(NAMES)+"。起点为 MoST 米制 XY，须有许可路径；行人限长单段人行道。"
        "扩散和反馈 CNN 是本项目自训模型，CNN 直接使用原组编码程序。")
    path=ROOT/"eight-class-model-and-gaf.zip"
    with ZipFile(path,"w",compression=ZIP_DEFLATED) as z:
        for file in sorted(work.rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts:
                z.write(file,file.relative_to(work))
    return path


if __name__=="__main__":
    with np.load(OLD/"requests.npz") as q:
        a=write_samples(q)
        b=kit(q)
    (ROOT/"delivery.json").write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                                 for p in (a,b)},indent=2))
