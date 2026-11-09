"""Goal-conditioned masked road-choice behavioral cloning; no future-motion features."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from g31_paths import source_inputs
from g32_routes import VehicleRouteNetwork
from g33_evaluate import digest, save
from g33_model import CLASSES
from g37_routes import graph, bind_stops


class RoadFeatures:
    """All choice features come from the static map and declared next goal."""
    def __init__(self, network):
        self.net = VehicleRouteNetwork(network)
        self.graphs = {}

    def graph(self, vc):
        if vc not in self.graphs:
            self.graphs[vc] = graph(self.net, vc)[0]
        return self.graphs[vc]

    @lru_cache(maxsize=256)
    def distances(self, vc, goal):
        return nx.single_source_dijkstra_path_length(self.graph(vc).reverse(copy=False), goal, weight='weight')

    @lru_cache(maxsize=65536)
    def edge(self, vc, edge):
        lanes = self.net.permitted(edge, vc)
        lane = self.net.lanes[sorted(lanes)[0]]
        xy = np.asarray(lane['shape'])
        direction = xy[-1] - xy[0]
        direction /= max(np.linalg.norm(direction), 1e-9)
        return (xy[0], xy[-1], direction,
                min(self.net.length(l) for l in lanes),
                max(float(self.net.lanes[l]['attributes']['speed']) for l in lanes), len(lanes))

    def choices(self, vc, current, goal):
        dist = self.distances(vc, goal)
        successors = sorted(v for v in self.graph(vc).successors(current) if v in dist)
        if not successors or current not in dist:
            raise ValueError('no reachable successor')
        start, end, heading, _, _, _ = self.edge(vc, current)
        goal_xy = self.edge(vc, goal)[1]
        goal_vec = goal_xy - end
        goal_dir = goal_vec / max(np.linalg.norm(goal_vec), 1e-9)
        values, costs = [], []
        for nxt in successors:
            a, b, h, length, speed, lanes = self.edge(vc, nxt)
            cost = length + dist[nxt]
            # Fixed physical scales, no fitted validation statistics.
            values.append(np.r_[end / 10000, b / 10000, goal_xy / 10000,
                (goal_xy-end) / 10000, (goal_xy-b) / 10000,
                math.log1p(length) / 10, speed / 40, lanes / 5,
                heading @ h, heading[0]*h[1]-heading[1]*h[0], h @ goal_dir,
                dist[current] / 10000, dist[nxt] / 10000,
                (cost-dist[current]) / 1000, float(nxt == goal)])
            costs.append(cost)
        x = np.asarray(values, np.float32)
        if not np.isfinite(x).all():
            raise ValueError('nonfinite map feature')
        return successors, x, np.asarray(costs, np.float32)


def goals_for_route(edges, stop_edges):
    """Bind ordered public goals to edge occurrences; no realized stop times."""
    at, cursor = [], 0
    for stop in stop_edges:
        indices = [i for i in range(cursor, len(edges)) if edges[i] == stop]
        if not indices:
            raise ValueError('public stop absent from realized training route')
        cursor = indices[0]
        at.append(cursor)
    at.append(len(edges)-1)
    result = []
    for i in range(len(edges)-1):
        goal_index = next(j for j in at if j > i)
        goal = edges[goal_index]
        if goal in edges[i:goal_index]:
            raise ValueError('non-first-passage segment')
        result.append(goal)
    return result


def prepare(data, source, out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        receipt = json.loads((data/'complete.json').read_text())
        meta = data/'leg-metadata.json.gz'
        assert receipt['complete'] and receipt['classes'] == CLASSES
        assert receipt['artifacts'][meta.name]['sha256'] == digest(meta)
        raw = json.load(gzip.open(meta, 'rt'))['accepted']
        assert set(r['partition'] for r in raw) == {'train', 'val'}
        sets = {s: {r['parent_base_id'] for r in raw if r['partition'] == s} for s in ('train', 'val')}
        assert not sets['train'] & sets['val']
        network, types = source_inputs(source)
        roads = RoadFeatures(network)
        stops = {n.get('id'): n.get('lane').rsplit('_', 1)[0] for n in
                 ET.parse(source/'scenario/in/add/most.busstops.add.xml').getroot().findall('busStop')}
        contexts, context_ids, xs, costs = [], {}, [], []
        state_ids, target, leg_ids, legs, rejects = [], [], [], [], Counter()
        for ri, r in enumerate(raw):
            edges = r['realized_route_audit_only']['edges']
            try:
                if len(edges) < 2:
                    raise ValueError('fewer than two road edges')
                vc = types[r['vtype']]
                g = roads.graph(vc)
                if any(e not in g for e in edges) or any(not g.has_edge(u, v) for u, v in zip(edges, edges[1:])):
                    raise ValueError('unsupported permitted directed transition')
                stop_edges = [stops[s['busStop']] for s in r['source_plan'].get('planned_stops', [])] if r['label'] == 0 else []
                goals = goals_for_route(edges, stop_edges)
                local = []
                for u, v, goal in zip(edges, edges[1:], goals):
                    key = (vc, u, goal)
                    if key not in context_ids:
                        choices, features, cost = roads.choices(*key)
                        context_ids[key] = len(contexts)
                        contexts.append(choices); xs.append(features); costs.append(cost)
                    si = context_ids[key]
                    if v not in contexts[si]:
                        raise ValueError('observed successor not reachable')
                    if len(contexts[si]) > 1:
                        local.append((si, contexts[si].index(v)))
                if not local:
                    raise ValueError('no branching decision')
                li = len(legs)
                legs.append({'id': r['leg_id'], 'parent': r['parent_base_id'], 'label': r['label'],
                    'split': r['partition'], 'vc': vc, 'branching': len(local), 'transitions': len(edges)-1,
                    'od': [r['label'], edges[0], edges[-1]], 'route_hash': r['realized_route_hash']})
                state_ids.extend(s for s, _ in local); target.extend(y for _, y in local); leg_ids.extend([li]*len(local))
            except (ValueError, KeyError, nx.NetworkXException) as e:
                rejects[(r['partition'], CLASSES[r['label']], str(e))] += 1
            if (ri+1) % 1000 == 0:
                save(out/'progress.json', {'processed': ri+1, 'accepted': len(legs), 'states': len(contexts), 'seconds': time.monotonic()-started})
                print(json.dumps({'processed': ri+1, 'accepted': len(legs)}), flush=True)
        train_od={tuple(r['od']) for r in legs if r['split']=='train'}
        train_routes={r['route_hash'] for r in legs if r['split']=='train'}
        for r in legs:
            r['unseen_od']=tuple(r['od']) not in train_od
            r['shared_route']=r['route_hash'] in train_routes
        width = max(map(len, contexts)); dim = xs[0].shape[1]
        x = np.zeros((len(xs), width, dim), np.float32)
        mask = np.zeros(x.shape[:2], bool); cost = np.zeros(x.shape[:2], np.float32)
        for i, a in enumerate(xs):
            x[i, :len(a)] = a; mask[i, :len(a)] = True; cost[i, :len(a)] = costs[i]
        np.savez_compressed(out/'decisions.npz', x=x, mask=mask, cost=cost,
            state=np.asarray(state_ids, np.int32), target=np.asarray(target, np.int64), leg=np.asarray(leg_ids, np.int32))
        save(out/'legs.json', {'rows': legs})
        summary = {s: {c: {'input_legs': sum(r['partition']==s and r['label']==ci for r in raw),
                    'accepted_legs': sum(r['split']==s and r['label']==ci for r in legs),
                    'parents': len({r['parent'] for r in legs if r['split']==s and r['label']==ci}),
                    'branch_decisions': sum(r['branching'] for r in legs if r['split']==s and r['label']==ci),
                    'all_transitions': sum(r['transitions'] for r in legs if r['split']==s and r['label']==ci),
                    'unseen_od_legs': sum(r['split']==s and r['label']==ci and r['unseen_od'] for r in legs)}
                    for ci, c in enumerate(CLASSES)} for s in ('train', 'val')}
        save(out/'complete.json', {'complete': True, 'seconds': time.monotonic()-started,
            'summary': summary, 'rejections': [{'split': s, 'class': c, 'reason': e, 'n': n} for (s,c,e),n in rejects.items()],
            'network_sha256': digest(network), 'metadata_sha256': digest(meta), 'script_sha256': digest(__file__),
            'features': dim, 'max_choices': width, 'contexts': len(contexts),
            'artifacts': {n: digest(out/n) for n in ['decisions.npz','legs.json']},
            'test_accessed': False, 'motion_used_as_feature': False})
    except BaseException as e:
        save(out/'failed.json', {'error': repr(e)}); raise


class Policy(nn.Module):
    def __init__(self, dim, kind):
        super().__init__()
        self.net = nn.Linear(dim, 5) if kind == 'linear' else nn.Sequential(nn.Linear(dim,128), nn.SiLU(), nn.Linear(128,64), nn.SiLU(), nn.Linear(64,5))

    def forward(self, x, label, mask):
        logits = self.net(x).gather(2, label[:,None,None].expand(-1,x.shape[1],1)).squeeze(2)
        return logits.masked_fill(~mask, -1e9)


def load_data(data):
    receipt = json.loads((data/'complete.json').read_text())
    assert receipt['complete'] and receipt['script_sha256'] == digest(__file__)
    for n,h in receipt['artifacts'].items(): assert digest(data/n) == h
    arrays = dict(np.load(data/'decisions.npz', allow_pickle=False))
    legs = json.loads((data/'legs.json').read_text())['rows']
    return arrays, legs, receipt


def train(data, out, kind, seed, steps=4000):
    out.mkdir(parents=True, exist_ok=False); start=time.monotonic()
    try:
        torch.set_num_threads(2); torch.manual_seed(seed); rng=np.random.default_rng(seed)
        a, legs, receipt=load_data(data)
        x=torch.from_numpy(a['x']).cuda(); mask=torch.from_numpy(a['mask']).cuda()
        labels=np.asarray([r['label'] for r in legs]); states=a['state']; targets=a['target']
        byleg=[[] for _ in legs]
        for i,li in enumerate(a['leg']): byleg[li].append(i)
        groups=[defaultdict(list) for _ in CLASSES]
        for i,r in enumerate(legs):
            if r['split']=='train': groups[r['label']][r['parent']].append(i)
        parents=[list(g.values()) for g in groups]
        assert all(parents)
        model=Policy(x.shape[-1],kind).cuda(); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        config={'kind':kind,'seed':seed,'steps':steps,'batch':512,'data_sha256':digest(data/'complete.json'),
                'script_sha256':digest(__file__),'features':x.shape[-1],'network_sha256':receipt['network_sha256'],
                'checkpoint':'fixed_final','sampling':'class,parent,leg,branch each uniform; train only'}
        save(out/'config.json',config)
        with (out/'training.jsonl').open('x',buffering=1) as log:
            for step in range(1,steps+1):
                cl=rng.integers(0,5,512); ids=[]
                for c in cl:
                    p=parents[c][rng.integers(len(parents[c]))]; li=p[rng.integers(len(p))]
                    ids.append(byleg[li][rng.integers(len(byleg[li]))])
                ids=np.asarray(ids); si=torch.as_tensor(states[ids],device='cuda',dtype=torch.long)
                opt.zero_grad(set_to_none=True)
                pred=model(x[si],torch.as_tensor(cl,device='cuda'),mask[si])
                loss=F.cross_entropy(pred,torch.as_tensor(targets[ids],device='cuda'))
                assert torch.isfinite(loss)
                loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.); assert torch.isfinite(norm); opt.step()
                if step==1 or step%100==0:
                    row={'step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-start}
                    log.write(json.dumps(row)+'\n'); save(out/'progress.json',row)
        torch.save({'state':model.state_dict(),'config':config},out/'model.pt')
        save(out/'complete.json',{'complete':True,'seconds':time.monotonic()-start,'model_sha256':digest(out/'model.pt'),
            'config_sha256':digest(out/'config.json'),'steps':steps,'seed':seed,'kind':kind,'validation_used':False})
    except BaseException as e:
        save(out/'failed.json',{'error':repr(e),'seconds':time.monotonic()-start}); raise


def load_policy(folder, device='cpu'):
    receipt=json.loads((folder/'complete.json').read_text()); config=json.loads((folder/'config.json').read_text())
    assert receipt['complete'] and digest(folder/'model.pt')==receipt['model_sha256'] and digest(folder/'config.json')==receipt['config_sha256']
    assert config['script_sha256']==digest(__file__)
    payload=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    assert payload['config']==config
    model=Policy(config['features'],config['kind']).to(device); model.load_state_dict(payload['state'])
    return model.eval().requires_grad_(False),config


def evaluate(data, models, out):
    out.mkdir(parents=True,exist_ok=False); a,legs,receipt=load_data(data)
    labels=np.asarray([r['label'] for r in legs]); val=np.asarray([r['split']=='val' for r in legs])[a['leg']]
    ix=np.flatnonzero(val); state=a['state'][ix]; target=a['target'][ix]; ll=a['leg'][ix]
    predictors={}
    for kind in ('linear','neural'):
        for seed in (42,43):
            model,config=load_policy(models/f'{kind}-{seed}','cuda'); assert config['data_sha256']==digest(data/'complete.json')
            predictors[f'{kind}-{seed}']=model
    all_results={}
    for name in ['Uniform','ShortestCost',*predictors]:
        measures=[]
        for start in range(0,len(ix),8192):
            ids=state[start:start+8192]; mask=torch.as_tensor(a['mask'][ids],device='cuda'); y=torch.as_tensor(target[start:start+8192],device='cuda')
            with torch.no_grad():
                if name=='Uniform': logits=torch.zeros_like(mask,dtype=torch.float32)
                elif name=='ShortestCost': logits=-torch.as_tensor(a['cost'][ids],device='cuda')/1000.
                else: logits=predictors[name](torch.as_tensor(a['x'][ids],device='cuda'),torch.as_tensor(labels[ll[start:start+8192]],device='cuda'),mask)
                lp=logits.masked_fill(~mask,-1e9).log_softmax(1); p=lp.exp(); confidence,guess=p.max(1)
                nll=-lp.gather(1,y[:,None]).squeeze(1); brier=p.square().sum(1)-2*p.gather(1,y[:,None]).squeeze(1)+1
                measures.append(torch.stack([nll,(guess==y).float(),brier,confidence],1).cpu().numpy())
        values=np.concatenate(measures); byleg={}
        for li in np.unique(ll): byleg[int(li)]=values[ll==li].mean(0)
        groups={}
        for subset in ('all','unseen_od','seen_od'):
            groups[subset]={}
            for c,classname in enumerate(CLASSES):
                chosen=[li for li in byleg if legs[li]['label']==c and (subset=='all' or legs[li]['unseen_od']==(subset=='unseen_od'))]
                parent=defaultdict(list)
                for li in chosen: parent[legs[li]['parent']].append(byleg[li])
                means=np.asarray([np.mean(v,0) for v in parent.values()])
                # Ten fixed bins, same parent/leg/decision weights as NLL.
                chosen_set=set(chosen);weights=np.zeros(len(ll))
                parent_legs=Counter(legs[li]['parent'] for li in chosen)
                for i,li in enumerate(ll):
                    if li in chosen_set:weights[i]=1/(len(parent)*parent_legs[legs[li]['parent']]*legs[li]['branching'])
                ece=0.
                for low in np.arange(0.,1.,.1):
                    selected=(values[:,3]>=low)&(values[:,3]<(low+.1) if low<.85 else values[:,3]<=1.)
                    w=weights[selected];v=values[selected]
                    if w.sum()>0:ece+=abs(float(np.sum(w*(v[:,1]-v[:,3]))))
                groups[subset][classname]={'legs':len(chosen),'parents':len(parent),'decisions':int(sum(legs[li]['branching'] for li in chosen)),
                    'ece_10_bins':ece if chosen else None,
                    **({k:float(v) for k,v in zip(['nll','accuracy','brier','mean_confidence'],means.mean(0))} if len(means) else {})}
        all_results[name]=groups
    save(out/'complete.json',{'complete':True,'results':all_results,'data_sha256':digest(data/'complete.json'),
        'scope':'development branching decisions; equal parent then leg then branch per class; no checkpoint selection',
        'model_hashes':{p.name: digest(p/'model.pt') for p in models.iterdir() if p.is_dir() and (p/'model.pt').exists()},
        'shared_route_val_legs':sum(r['split']=='val' and r['shared_route'] for r in legs)})


def route_logprob(roads, model, vc, label, edges, stops):
    goals=goals_for_route(edges,[s['lane'].rsplit('_',1)[0] for s in stops]); result=0.
    with torch.no_grad():
        for current,nxt,goal in zip(edges,edges[1:],goals):
            choices,x,_=roads.choices(vc,current,goal)
            if nxt not in choices: raise ValueError('catalog edge missing from policy choice set')
            if len(choices)>1:
                logits=model(torch.from_numpy(x)[None],torch.tensor([label]),torch.ones((1,len(x)),dtype=torch.bool))
                result+=float(logits.log_softmax(1)[0,choices.index(nxt)])
    return result


def checks():
    assert goals_for_route(['a','b','c','d'],[])==['d','d','d']
    assert goals_for_route(['a','b','c','d'],['b','c'])==['b','c','d']
    try: goals_for_route(['a','d','b','d'],[])
    except ValueError: pass
    else: raise AssertionError('first passage accepted repeated destination')
    for kind in ['linear','neural']:
        m=Policy(20,kind); x=torch.randn(2,3,20); mask=torch.tensor([[True,True,False],[True,False,False]])
        logits=m(x,torch.tensor([0,4]),mask); p=logits.softmax(1)
        assert p[0,2]==0 and p[1,0]==1
        perm=torch.tensor([1,0,2]); assert torch.allclose(m(x[:,perm],torch.tensor([0,4]),mask[:,perm]),logits[:,perm])
    print('G38 checks passed: public-goal segmentation, first passage, masked choice, permutation equivariance')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','evaluate','checks']);p.add_argument('--data',type=Path);p.add_argument('--source',type=Path);p.add_argument('--output',type=Path);p.add_argument('--models',type=Path);p.add_argument('--kind',choices=['linear','neural']);p.add_argument('--seed',type=int);p.add_argument('--steps',type=int,default=4000);a=p.parse_args()
    if a.action=='prepare':prepare(a.data,a.source,a.output)
    elif a.action=='train':train(a.data,a.output,a.kind,a.seed,a.steps)
    elif a.action=='evaluate':evaluate(a.data,a.models,a.output)
    else:checks()
