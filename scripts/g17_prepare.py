"""Independent evaluation starts from the G14 validation episodes only."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from cnr.route_paths import RoutePaths
from g16_data import BASE, NAMES, ORIGINAL

root = Path("../g14-job/g14-data")
matcher = RoutePaths("data/maps/most.net.xml",radius=2.)
prior = json.loads((ORIGINAL/"route_prior.json").read_text())
counts = {(r["class"],matcher.ids[r["from"]],matcher.ids[r["to"]]):r["count"] for r in prior}
rows, failures = [], Counter()
for ep in range(261065,261073):
    with np.load(root/str(ep)/"eligible.npz") as z:
        ids,labs,tracks = z["ids"],z["labels"],z["xy"]
    for label in range(1,6):
        rng = np.random.default_rng(ep*10+label)
        n = 0
        for i in sorted(np.flatnonzero(labs==label),key=lambda i:ids[i]):
            start = tracks[i,0]
            plan = matcher.sample(start,NAMES[label],counts,rng,budget=4096.)
            if plan is None or plan[1]<64:
                failures[NAMES[label]] += 1
                continue
            rows.append((ids[i],label,ep,start,plan[0],plan[1],tracks[i]))
            n += 1
            if n==3:
                break
ids,labs,episodes,starts,routes,lengths,refs = map(np.asarray,zip(*rows))
np.savez(BASE/"g14-requests.npz",ids=ids,labels=labs,episodes=episodes,start=starts,
         route=routes.astype(np.float32),length=lengths.astype(np.float32),lane=np.full(len(ids),-1))
np.savez(BASE/"g14-references.npz",ids=ids,xy=refs)
(BASE/"g14-coverage.json").write_text(json.dumps(dict(conditions=len(ids),failures=failures,
    counts=Counter(NAMES[l] for l in labs)),indent=2))
