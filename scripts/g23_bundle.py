"""Overlay the speed-pattern scorer onto the tested complete-journey package."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from g23_modes import OUT

kit=OUT/"overlay";(kit/"scripts").mkdir(parents=True,exist_ok=True);(kit/"assets").mkdir(exist_ok=True)
for name in ("g23_score_trip","g23_select","g23_fit","g23_modes"):
    shutil.copyfile(f"scripts/{name}.py",kit/f"scripts/{name}.py")
shutil.copyfile(OUT/"speed-mode.pt",kit/"assets/speed-mode.pt")
shutil.copyfile(OUT/"definition.json",kit/"assets/speed-mode-definition.json")
with np.load(OUT/"train.npz") as z:
    np.savez(kit/"assets/mode-prior.npz",**{k:z[k] for k in ("participant","length","y")})
# Test exactly the overlay layout without overwriting the earlier release.
base=Path("../g22-job/g22-data/kit").resolve()
test=OUT/"overlay-test";test.mkdir(exist_ok=True)
for name in ("assets","scripts"):
    (test/name).mkdir(exist_ok=True)
    for f in (base/name).glob("*"):
        if f.is_file() and not (test/name/f.name).exists():(test/name/f.name).symlink_to(f)
    for f in (kit/name).iterdir():
        shutil.copyfile(f,test/name/f.name)
for name in ("src","vendor","example.csv"):
    if not (test/name).exists():(test/name).symlink_to(base/name)
length=json.loads((base/"example-result.json").read_text())["route_length_m"]
cmd=["scripts/g23_score_trip.py","--assets","assets","--csv","example.csv","--mode","bus","--route-length",str(length),
     "--seed","42","--request-id","example"]
ans=subprocess.run([sys.executable,*cmd],cwd=test,text=True,capture_output=True,env={**os.environ,"PYTHONPATH":"src"})
if ans.returncode:raise RuntimeError(ans.stderr)
(kit/"speed-feedback-example.json").write_text(ans.stdout)
(kit/"README-speed-feedback.md").write_text("# 两种 GAF 全程反馈\n解压覆盖到完整行程模型包根目录，在 CUDA 环境运行：\n\n    PYTHONPATH=src python "+" ".join(cmd)+"\n\n"
    "空间 GAF 的目标是公交等参与者类别，速度 GAF 的目标只包含均速档与波动档。两者均看整趟行程。目标模式从训练侧同车型、相近规划路线长度的记录抽取；编码定义在 assets/speed-mode-definition.json。返回两个单独的候选 ID 和一个等权组合 ID。研究比较只用独立运动指标。")
out=OUT/"speed-gaf-feedback-addon.zip"
with ZipFile(out,"w",compression=ZIP_DEFLATED,compresslevel=3) as z:
    for f in sorted(kit.rglob("*")):
        if f.is_file():z.write(f,f.relative_to(kit))
with ZipFile(out) as z:assert z.testzip() is None
(OUT/"addon-sha.json").write_text(json.dumps({out.name:hashlib.sha256(out.read_bytes()).hexdigest()}))
