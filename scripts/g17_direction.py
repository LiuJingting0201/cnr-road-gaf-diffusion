"""Carry seed-level primary discrepancies and signed travel calibration into the report."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
from g16_data import BASE,NAMES


def distance(x):
    return np.linalg.norm(np.diff(x.astype(float),axis=-2),axis=-1).sum(-1)


def summarize(domain):
    prefix = "" if domain == "g15" else "g14-"
    q = np.load(BASE/(prefix+"requests.npz"))
    ref = np.load(BASE/(prefix+"references.npz"))
    assert np.array_equal(q["ids"],ref["ids"])
    ids = {k:i for i,k in enumerate(q["ids"])}
    rows = {}
    for arm in "ADBC":
        rows[arm] = {}
        for seed in [42,43]:
            generated = np.load(BASE/f"{arm}-{seed}"/(domain+".npz"))
            indices = [ids[k] for k in generated["ids"]]
            real, syn = distance(ref["xy"][indices]),distance(generated["xy"])
            labs = q["labels"][indices]
            row = {}
            for c in np.unique(labs):
                a, b = real[labs==c],syn[labs==c].ravel()
                row[NAMES[c]] = dict(reference_median=float(np.median(a)),generated_median=float(np.median(b)),
                                      travel_w1=float(wasserstein_distance(a,b)))
            rows[arm][seed] = dict(five_macro=float(np.mean([row[NAMES[c]]["travel_w1"] for c in range(1,6)])),
                                  classes=row)
    return rows


if __name__=="__main__":
    (BASE/"seed-travel.json").write_text(json.dumps({q:summarize(q) for q in ("g14","g15")},indent=2))
