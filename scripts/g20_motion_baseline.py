"""A fixed motion-feature probe on precisely the G19 scorer's train and monitor IDs."""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from g16_data import csv_tracks


def features(xy):
    delta = np.diff(xy.astype(float), axis=0)
    speed = np.linalg.norm(delta,axis=1)*4
    one_second = speed[:508].reshape(127,4).mean(1)
    acceleration = np.diff(one_second)
    return np.r_[np.quantile(one_second,[0,.1,.25,.5,.75,.9,1]),one_second.mean(),one_second.std(),
                 np.quantile(acceleration,[0,.1,.25,.5,.75,.9,1]),abs(acceleration).mean(),
                 np.mean(one_second<.3),np.mean(abs(acceleration)<.5),
                 np.mean(abs(one_second[4:]-one_second[:-4]))]


def scores(y,pred):
    cm = np.bincount(8*y+pred,minlength=64).reshape(8,8)
    f1 = 2*np.diag(cm)/np.maximum(1,cm.sum(0)+cm.sum(1))
    return dict(accuracy=float(np.trace(cm)/len(y)),macro_f1=float(f1.mean()),cm=cm.tolist())


if __name__=="__main__":
    index=np.load("../g19-job/g19-data/image-index.npz")
    pos={str(v):i for i,v in enumerate(index["ids"])}
    records={}
    for ident,label,xy in csv_tracks("train"):
        if ident in pos:
            i=pos[ident]
            assert index["labels"][i]==label
            records[i]=features(xy)
    feat=np.stack([records[i] for i in range(len(pos))])
    fit=index["valid"]&~index["monitor"]
    held=index["valid"]&index["monitor"]
    clf=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=5,random_state=20,n_jobs=4)
    clf.fit(feat[fit],index["labels"][fit])
    root=Path("../g17-job/g17-data")
    with np.load(root/"requests.npz") as q, np.load(root/"references.npz") as r:
        assert np.array_equal(q["ids"],r["ids"]) and set(q["ids"]).isdisjoint(pos)
        test=scores(q["labels"],clf.predict(np.stack([features(x) for x in r["xy"]])))
    result=dict(fit_ids=int(fit.sum()),monitor=scores(index["labels"][held],clf.predict(feat[held])),development=test)
    Path("motion.json").write_text(json.dumps(result,indent=2))
    print(test)
