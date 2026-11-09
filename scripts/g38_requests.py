"""Reuse audited SUMO mechanics with a new, outcome-independent request cohort."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import g36_requests as base
from g33_evaluate import digest, save


def main(a):
    base.SELECTION_SEED=38001
    base.SIMULATION_SEED=a.episode
    base.DEPARTURE=18000. if a.episode==380201 else 32400.
    base.END=base.DEPARTURE+5000.
    if a.action=='prepare':
        old=[Path('../g36-job/g36-data')/str(e)/'requests/requests.json' for e in (360201,360202)]
        excluded={r['source_parent'] for p in old for r in json.loads(p.read_text())['requests'] if r['label']!=0}
        original=base.child_nodes
        public=(a.source/'scenario/in/route/most.pedestrian.rou.xml').resolve()
        def children(path):
            for node in original(path):
                if Path(path).resolve()==public and node.tag=='vehicle' and node.get('id') in excluded: continue
                yield node
        base.child_nodes=children
        base.prepare(a.source,a.splits,a.prepared)
        manifest=json.loads((a.prepared/'requests.json').read_text())
        assert all(r['source_parent'] not in excluded for r in manifest['requests'] if r['label']!=0)
        save(a.prepared/'g38-cohort.json',{'complete':True,'selection_seed':38001,'episode':a.episode,
            'excluded_G36_nonbus_source_parents':len(excluded),'source_manifest_hashes':[digest(p) for p in old],
            'requests_sha256':digest(a.prepared/'requests.json'),'bus_templates_shared':True,
            'future_outcomes_used_for_cohort':False,'script_sha256':digest(__file__)})
    elif a.action=='run':
        base.run(a.source,a.prepared,a.simulation,a.sumo)
        last=None
        for _,n in ET.iterparse(a.simulation/'summary.xml',events=('end',)):
            if n.tag=='step':last=float(n.get('time'));n.clear()
        assert last==base.END-.25,(last,base.END)
        save(a.simulation/'clock-verified.json',{'complete':True,'last_time':last,'expected':base.END-.25})
    elif a.action=='extract':
        assert json.loads((a.simulation/'clock-verified.json').read_text())['complete']
        base.extract(a.prepared,a.simulation,a.output)
    else:print(json.dumps(base.self_test()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','extract','checks']);p.add_argument('--episode',type=int,choices=[380201,380202],required=True)
    p.add_argument('--source',type=Path,default=Path('../g30-job/g30-data/source'));p.add_argument('--splits',type=Path,default=Path('../data/processed/most64/dataset.json'))
    p.add_argument('--prepared',type=Path);p.add_argument('--simulation',type=Path);p.add_argument('--output',type=Path);p.add_argument('--sumo',type=Path,default=Path('../g10-env/sumo/bin/sumo'))
    main(p.parse_args())
