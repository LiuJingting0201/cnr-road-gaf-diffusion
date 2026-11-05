"""Package the four predeclared seed-42 training arms as an overlay on the G22 kit."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np
from g26_train import ARMS,OUT,LAYOUT


def make():
    kit=OUT/"overlay";assets=kit/"assets";scripts=kit/"scripts"
    assets.mkdir(parents=True,exist_ok=True);scripts.mkdir(exist_ok=True)
    for arm in ARMS:
        src=OUT/f"{arm}-42";shutil.copyfile(src/"model.pt",assets/f"{arm}.pt");shutil.copyfile(src/"stats.json",assets/f"{arm}.json")
    for name in ("g26_train","g26_generate","g23_select","g23_fit","g23_modes"):
        f=next(p/f"{name}.py" for p in (Path("scripts"),Path("../g23-job/scripts")) if (p/f"{name}.py").exists())
        shutil.copyfile(f,scripts/f.name)
    with np.load("../g23-job/g23-data/train.npz") as d:np.savez(assets/"mode-prior.npz",**{k:d[k] for k in ("length","participant","y")})
    base=Path("../g22-job/g22-data/kit").resolve()
    test=OUT/"kit-check";test.mkdir(exist_ok=True)
    for name in ("src","vendor","example-route.npz"):
        if not (test/name).exists():(test/name).symlink_to(base/name)
    for name in ("assets","scripts"):
        (test/name).mkdir(exist_ok=True)
        for f in (base/name).iterdir():
            if f.is_file() and not (test/name/f.name).exists():(test/name/f.name).symlink_to(f)
        for f in (kit/name).iterdir():
            dest=test/name/f.name
            if dest.is_symlink():dest.unlink()
            shutil.copyfile(f,dest)
    cmd=["scripts/g26_generate.py","--assets","assets","--arm","HG","--route","example-route.npz","--mode","bus",
         "--seed","26042","--out","g26-example.csv"]
    (test/"g26-example.csv").unlink(missing_ok=True)
    proc=subprocess.run([sys.executable,*cmd],cwd=test,env={**os.environ,"PYTHONPATH":"src"},text=True,capture_output=True)
    if proc.returncode:raise RuntimeError(proc.stderr)
    shutil.copyfile(test/"g26-example.csv",kit/"g26-example.csv")
    (kit/"g26-example.json").write_text(proc.stdout)
    cmd[-1]="new-trips.csv"
    base_name="连续全程进度" if LAYOUT=="flat" else "分层进度"
    (kit/"README-training-arms.md").write_text("# 完整行程"+base_name+"扩散\n解压到 G22 完整行程模型包根目录，CUDA/PyTorch 环境执行：\n\n    PYTHONPATH=src python "+" ".join(cmd)+
        "\n\n输出四条独立完整候选。H=基础表示；HM=再加模式条件；HV=再加直接速度曲线训练损失；HG=改用速度角场训练损失。可指定 --motion-mode，或从同类训练分布采样。第42种子固定，不按验证成绩挑权重。")
    out=OUT/("continuous-mode-addon.zip" if LAYOUT=="flat" else "multiscale-training-addon.zip")
    with ZipFile(out,"w",compression=ZIP_DEFLATED) as z:
        for f in sorted(kit.rglob("*")):
            if f.is_file():z.write(f,f.relative_to(kit))
    with ZipFile(out) as z:assert z.testzip() is None
    (OUT/"delivery.json").write_text(json.dumps({out.name:hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__=="__main__":make()
