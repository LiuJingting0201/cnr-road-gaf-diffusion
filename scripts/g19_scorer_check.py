"""Descriptive transfer check on G15 development references; no checkpoint selection."""
import json
import numpy as np
import torch
from g12_fit import features
from g19_gaf import images,model
from g19_field import ROOT
from g19_train import OLD

with np.load(OLD/"references.npz") as r, np.load(OLD/"requests.npz") as q:
    assert np.array_equal(q["ids"],r["ids"])
    xy,labels=r["xy"],q["labels"]
net=model().eval()
net.load_state_dict(torch.load(ROOT/"gaf8.pt",weights_only=True,map_location="cuda"))
cm=np.zeros((8,8),int)
for start in range(0,len(xy),16):
    rgb,ok=images(xy[start:start+16])
    with torch.no_grad():
        pred=net(features(torch.from_numpy(rgb).cuda())).argmax(1).cpu().numpy()
    np.add.at(cm,(labels[start:start+16][ok],pred[ok]),1)
recall=np.diag(cm)/np.maximum(1,cm.sum(1))
f1=2*np.diag(cm)/np.maximum(1,cm.sum(0)+cm.sum(1))
(ROOT/"gaf-dev.json").write_text(json.dumps(dict(supported=int(cm.sum()),requests=len(xy),accuracy=float(np.trace(cm)/cm.sum()),
    macro_f1=float(f1.mean()),recall=recall.tolist(),cm=cm.tolist()),indent=2))
