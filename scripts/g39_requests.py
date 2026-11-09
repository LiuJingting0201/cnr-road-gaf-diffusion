"""Eight outcome-independent references, four disjoint nonbus OD cohorts."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import g36_requests as base
from g33_train import digest,save

EPISODES=tuple(range(390201,390209))


def main(a):
    cohort=(a.episode-390201)//2
    base.SELECTION_SEED=39001+cohort;base.SIMULATION_SEED=a.episode
    base.DEPARTURE=18000. if a.episode%2 else 32400.;base.END=base.DEPARTURE+5000.
    if a.action=='prepare':
        prior=[Path(f'../g{g}-job/g{g}-data/{e}/requests/requests.json')
               for g in (36,38) for e in (g*10000+201,g*10000+202)]
        # G36/G38 episode identifiers are six digits, e.g.360201.
        prior += [a.root/str(390201+2*c)/'requests/requests.json' for c in range(cohort)]
        excluded={r['source_parent'] for p in prior for r in json.loads(p.read_text())['requests'] if r['label']!=0}
        original=base.child_nodes;public=(a.source/'scenario/in/route/most.pedestrian.rou.xml').resolve()
        def children(path):
            for node in original(path):
                if Path(path).resolve()==public and node.tag=='vehicle' and node.get('id') in excluded:continue
                yield node
        base.child_nodes=children;base.prepare(a.source,a.splits,a.prepared)
        manifest=json.loads((a.prepared/'requests.json').read_text())
        assert all(r['source_parent'] not in excluded for r in manifest['requests'] if r['label']!=0)
        save(a.prepared/'g39-cohort.json',{'complete':True,'episode':a.episode,'cohort':cohort,
             'selection_seed':base.SELECTION_SEED,'excluded_prior_nonbus_parents':len(excluded),
             'prior_manifest_hashes':[digest(p) for p in prior],'requests_sha256':digest(a.prepared/'requests.json'),
             'bus_templates_shared':True,'future_outcomes_opened':False})
    elif a.action=='run':
        base.run(a.source,a.prepared,a.simulation,a.sumo)
        last=None
        for _,n in ET.iterparse(a.simulation/'summary.xml',events=('end',)):
            if n.tag=='step':last=float(n.get('time'));n.clear()
        assert last==base.END-.25
        save(a.simulation/'clock-verified.json',{'complete':True,'last_time':last,'expected':base.END-.25})
    else:
        assert json.loads((a.simulation/'clock-verified.json').read_text())['complete']
        base.extract(a.prepared,a.simulation,a.output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','extract']);p.add_argument('--episode',type=int,choices=EPISODES,required=True)
    p.add_argument('--root',type=Path,default=Path('g39-data'));p.add_argument('--source',type=Path,default=Path('../g30-job/g30-data/source'))
    p.add_argument('--splits',type=Path,default=Path('../data/processed/most64/dataset.json'))
    for k in ('prepared','simulation','output'):p.add_argument('--'+k,type=Path)
    p.add_argument('--sumo',type=Path,default=Path('../g10-env/sumo/bin/sumo'));main(p.parse_args())
