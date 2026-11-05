"""Frozen original-encoding scorer on the pre-existing G14 B pools, with no ref at selection."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torchvision.models import resnet18
from cnr.original_gaf import trajectory_sequence, encode_frozen
from cnr.episode_w1 import episode_w1
from g12_fit import predict
from g16_data import BASE


def score_pool(net, meta, temperature, xy, label):
    seq, ok = [], []
    for track in xy:
        try:
            h,_ = trajectory_sequence(track,np.array(meta["xy_min"]),np.array(meta["xy_max"]),eps=meta["hilbert_eps"])
            ok.append(True)
        except ValueError:
            h = np.zeros(224)
            ok.append(False)
        seq.append(h)
    img,_ = encode_frozen(seq,np.array(meta["channel_min"]),np.array(meta["channel_max"]),batch=4)
    logits = predict(net,torch.from_numpy(img).cuda())
    return torch.tensor(logits/temperature).softmax(1)[:,label].numpy(),np.array(ok)


def run():
    shared = Path("..")
    g12 = shared/"g12-data"
    cfg = json.loads((shared/"runs/long-path-e3/seed42/config.json").read_text())
    meta = cfg["dataset"]["source_protocol"]
    temperature = json.loads((g12/"reg-fit42.json").read_text())["T"]
    net = resnet18(weights=None)
    net.fc = torch.nn.Linear(512,6)
    net.load_state_dict(torch.load(g12/"reg-scorer42.pt",weights_only=True,map_location="cpu"))
    net.cuda().eval()
    req = np.load(BASE/"g14-requests.npz")
    labels,ep = req["labels"],np.unique(req["episodes"],return_inverse=True)[1]
    selections, failures = [], 0
    for seed in [42,43]:
        pool = np.load(BASE/f"B-{seed}/g14.npz")
        assert np.array_equal(pool["ids"],req["ids"])
        indices = []
        for i,xy in enumerate(pool["xy"]):
            prob,ok = score_pool(net,meta,temperature,xy,labels[i])
            perm = np.random.default_rng(17100+i).permutation(4)
            eligible = perm[ok[perm]]
            failures += int((~ok).sum())
            if not len(eligible):
                eligible = perm
            indices.append((eligible[0],eligible[np.argmax(prob[eligible])],eligible))
        selections.append((pool["xy"],indices))
    ref = np.load(BASE/"g14-references.npz")
    assert np.array_equal(ref["ids"],req["ids"])
    speed = lambda xy: 4*np.linalg.norm(np.diff(xy.astype(float),axis=-2),axis=-1)
    obs = speed(ref["xy"])
    counts = np.vstack([np.ones(8),np.random.default_rng(17512).multinomial(8,np.ones(8)/8,2000)])
    speed_delta, travel_delta = np.zeros(len(counts)),np.zeros(len(counts))
    for xy,indices in selections:
        v = speed(xy)
        pool_cost = np.array([[wasserstein_distance(row,reference) for row in group] for group,reference in zip(v,obs)])
        shifts,rand,gaf = [],[],[]
        for i,(uniform,selected,ok) in enumerate(indices):
            shifts.append(pool_cost[i,selected]-pool_cost[i,ok].mean())
            rand.append(v[i,uniform].sum()/4)
            gaf.append(v[i,selected].sum()/4)
        for c in range(1,6):
            use = labels==c
            e = ep[use]
            n = np.bincount(e,minlength=8)
            by_ep = np.bincount(e,weights=np.asarray(shifts)[use],minlength=8)
            speed_delta += (counts@by_ep)/(counts@n)/10
            ref_travel = obs[use].sum(-1,keepdims=True)/4
            for sign,arm in [(1,gaf),(-1,rand)]:
                travel_delta += sign*episode_w1(np.array(arm)[use,None],ref_travel,e,False,counts)/10
    def result(a):
        return dict(delta=float(a[0]),ci=np.quantile(a[1:],[.025,.975]).tolist())
    (BASE/"gaf.json").write_text(json.dumps(dict(candidates=len(ep)*8,encoding_failures=failures,
        within_speed_W1_mps=result(speed_delta),delivered_travel_W1_m=result(travel_delta)),indent=2))


if __name__=="__main__":
    torch.set_num_threads(2)
    run()
