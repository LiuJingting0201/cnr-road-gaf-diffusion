"""Package the predeclared seed-42 model with its map, normalization, driver and example."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from g16_data import BASE, NAMES, ORIGINAL


def build():
    kit = BASE/"eight-class-kit"
    assets = kit/"assets"
    assets.mkdir(parents=True,exist_ok=True)
    for file, dest in [(BASE/"C-42/ema.pt","ema.pt"), (ORIGINAL/"route_prior.json","route_prior.json"),
                       (Path("data/maps/most.net.xml"),"most.net.xml")]:
        shutil.copyfile(file,assets/dest)
    config = json.loads((ORIGINAL/"dataset.json").read_text())
    (assets/"normalization.json").write_text(json.dumps({k:config[k] for k in ["mean","std"]}))
    for name in ("g16_generate","g16_data","g16_train"):
        (kit/"scripts").mkdir(exist_ok=True)
        shutil.copyfile(Path(f"scripts/{name}.py"),kit/f"scripts/{name}.py")
    (kit/"src/cnr").mkdir(parents=True,exist_ok=True)
    for name in ("__init__","data","models","path_diffusion","roads","route_paths","topology"):
        shutil.copyfile(Path(f"src/cnr/{name}.py"),kit/f"src/cnr/{name}.py")
    q = np.load(BASE/"requests.npz")
    i = np.flatnonzero(q["labels"]==1)[0]
    x,y = q["start"][i]
    args = ["scripts/g16_generate.py","--assets","assets","--mode","bus","--x",str(float(x)),
            "--y",str(float(y)),"--seed","160116","--out","example.csv"]
    result = subprocess.run([sys.executable,*args],cwd=kit,text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    with (kit/"example.csv").open() as f:
        if sum(1 for _ in f)!=2049:
            raise ValueError("Incomplete exported example")
    args[-1]="new_sample.csv"
    (kit/"README.md").write_text(
        "# 八类路网条件轨迹扩散模型\n\n固定种子 42 的 G16-C，PyTorch CUDA，原训练服务器使用 torch 2.5.1。"
        "在 CUDA 版 PyTorch 环境安装 numpy、pandas、shapely>=2，然后解压，从此目录运行：\n\n"
        + "`PYTHONPATH=src python " + " ".join(args) + "`\n\n"
        + "输出四条带相对时间的轨迹。示例 example.csv 已由同一命令实跑。换 seed 同时改变规划器和扩散噪声；"
        + "起点是 MoST 米制 XY，须在可通行车道 2 米内。行人仅支持至少 220 米的单条人行道。"
        + f"可用标签：{', '.join(NAMES)}。路径条件仅编码前 64 米，实际解码沿完整规划路径。"
        + "仓库模型为扩散运动生成器，不含已训练的八类 GAF 反馈。地图来源：MoSTScenario/scenario/in/most.net.xml。\n")
    path = BASE/"eight-class-model.zip"
    with ZipFile(path,"w",compression=ZIP_DEFLATED) as z:
        for file in sorted(kit.rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts:
                z.write(file,file.relative_to(kit))
    print(result.stdout, path, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__=="__main__":
    build()
