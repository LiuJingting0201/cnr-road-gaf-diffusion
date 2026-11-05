"""Account for every planned trip and report the generated physical duration distribution."""
from collections import Counter
import json
import numpy as np
from g16_data import NAMES
from g22_diffusion import ROOT,load


def quantiles(x):
    return dict(min=float(x.min()),q10_q50_q90=np.quantile(x,[.1,.5,.9]).tolist(),max=float(x.max()),
                over_127_75s=int((x>127.75).sum()),total=int(x.size))


if __name__=="__main__":
    report={}
    for split in ("train","monitor","evaluation"):
        d=load(split);reasons=Counter();total=0
        files=sorted((ROOT/split).glob("*/complete.json"))
        for f in files:
            info=json.loads(f.read_text());total+=info["attempted"];reasons.update(info["reasons"])
        report[split]=dict(episodes=len(files),requested=total,retained=len(d["ids"]),rejected=reasons,
                           durations=quantiles(d["duration"]),classes={NAMES[c]:int(np.sum(d["labels"]==c)) for c in range(8)})
    T=np.concatenate([np.load(ROOT/f"model-{s}/samples.npz")["duration"] for s in [42,43]],axis=1)
    report["generated"]=quantiles(T)
    report["generated_classes"]={NAMES[c]:quantiles(T[d["labels"]==c]) for c in range(8)}
    report["bounds"]={str(s):json.loads((ROOT/f"model-{s}/complete.json").read_text()) for s in [42,43]}
    (ROOT/"summary.json").write_text(json.dumps(report,indent=2))
