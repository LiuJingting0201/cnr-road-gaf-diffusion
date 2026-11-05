"""Fit a speed-mode CNN with no participant-label input or loss."""
import json
import numpy as np
import torch
from torchvision.models import resnet18
from g23_modes import OUT,speed_images,prepare


def classifier():
    net=resnet18(weights=None);net.fc=torch.nn.Linear(512,6)
    return net.cuda()


def metrics(y,p):
    cm=np.bincount(6*y+p,minlength=36).reshape(6,6)
    f=2*np.diag(cm)/np.maximum(cm.sum(1)+cm.sum(0),1)
    return dict(accuracy=float(np.trace(cm)/len(y)),macro_f1=float(f.mean()),cm=cm.tolist())


def train():
    torch.set_num_threads(2);torch.manual_seed(23042);rng=np.random.default_rng(23042)
    tr=np.load(OUT/"train.npz");mon=np.load(OUT/"monitor.npz")
    model=classifier()
    # Use ImageNet initialization, not a participant-trained motion backbone.
    from pathlib import Path
    pre=torch.load(Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth",weights_only=True,map_location="cpu")
    del pre["fc.weight"], pre["fc.bias"];model.load_state_dict(pre,strict=False)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.005)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,18,eta_min=2e-5)
    x=torch.from_numpy(tr["v"]).cuda();vy=torch.from_numpy(tr["y"]).cuda()
    mx=torch.from_numpy(mon["v"]).cuda();best=-1;history=[]
    for epoch in range(1,19):
        model.train()
        for ix in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/64))):
            logits=model(speed_images(x[ix]))
            loss=torch.nn.functional.cross_entropy(logits,vy[ix],label_smoothing=.02)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        sched.step();model.eval()
        with torch.no_grad():
            pred=torch.cat([model(speed_images(block)).argmax(1) for block in mx.split(96)]).cpu().numpy()
        r=metrics(mon["y"],pred);history.append(r["macro_f1"])
        if r["macro_f1"]>best:
            best=r["macro_f1"];chosen=epoch;torch.save(model.state_dict(),OUT/"speed-mode.pt")
        print(epoch,r["accuracy"],r["macro_f1"],flush=True)
    (OUT/"fit.json").write_text(json.dumps(dict(fit=len(x),monitor=len(mx),chosen=chosen,monitor_macro_f1=best,history=history),indent=2))


if __name__=="__main__":
    prepare();train()
