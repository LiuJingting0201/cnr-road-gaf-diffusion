"""Generate fixed OD pools using the pre-existing complete-trip diffusion weights."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from cnr.path_diffusion import PathDenoiser
from g22_diffusion import sample
from g22_hires import run as replay
from g23_modes import OUT


def create():
    torch.set_num_threads(2)
    files=sorted(Path("g22-data/fresh").glob("*/trips.npz"))
    assert len(files)==8
    keys=("ids","labels","paths","length")
    data={k:[] for k in keys}
    for f in files:
        with np.load(f) as q:
            for k in keys:data[k].append(q[k])
    req={k:np.concatenate(v) for k,v in data.items()}
    for seed in (42,43):
        out=OUT/f"pool-{seed}.npz"
        if out.exists():continue
        source=Path("../g22-job/g22-data")/f"model-{seed}"
        net=PathDenoiser(classes=8).cuda().eval()
        net.load_state_dict(torch.load(source/"model.pt",weights_only=True,map_location="cuda"))
        result=sample(net,req,json.loads((source/"stats.json").read_text()),seed)
        np.savez_compressed(out,**result)
    with ProcessPoolExecutor(max_workers=8) as pool:
        list(pool.map(replay,[f.parent for f in files]))
    (OUT/"pools-ready.txt").write_text(str(len(req["ids"])))


if __name__=="__main__":create()
