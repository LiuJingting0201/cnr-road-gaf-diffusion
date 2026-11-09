"""Independent route-distribution and physical evaluation after candidate closure."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
from g33_evaluate import digest,save,read_pool,extract_record,weighted_signal_distance
from g33_model import CLASSES
from g36_requests import child_nodes
from g38_pool import METHODS
from g38_policy import RoadFeatures


def edge_mass(edges,lengths):
    mass={e:n*lengths[e] for e,n in Counter(edges).items()}
    total=sum(mass.values());assert total>0
    return {e:v/total for e,v in mass.items()}


def overlap(a,b,lengths):
    a,b=Counter(a),Counter(b);edges=set(a)|set(b)
    return sum(min(a[e],b[e])*lengths[e] for e in edges)/sum(max(a[e],b[e])*lengths[e] for e in edges)


def js(a,b):
    edges=sorted(set(a)|set(b));x=np.asarray([a.get(e,0.) for e in edges]);y=np.asarray([b.get(e,0.) for e in edges])
    x=x/x.sum();y=y/y.sum();m=.5*(x+y)
    return float(.5*np.sum(x[x>0]*np.log(x[x>0]/m[x>0]))+.5*np.sum(y[y>0]*np.log(y[y>0]/m[y>0])))


def weighted_native(pools,reference,signal):
    # Equal request mass; conditional candidate mass; physical-time mass within track.
    admitted=[]
    for rid,rows,weights in pools:
        rr=reference[rid]
        valid=[i for i,r in enumerate(rows) if len(r[signal]) and weights[i]>0]
        if len(rr[signal]) and valid:
            w=np.asarray(weights)[valid];w=w/w.sum();admitted.append((rr,[rows[i] for i in valid],w))
    if not admitted:return None,0
    gx=[];gw=[];rx=[];rw=[]
    for rr,rows,weights in admitted:
        rx.append(rr[signal]);rw.append(rr[signal+'_weight']/len(admitted))
        for row,w in zip(rows,weights):gx.append(row[signal]);gw.append(row[signal+'_weight']*w/len(admitted))
    return float(wasserstein_distance(np.concatenate(gx),np.concatenate(rx),np.concatenate(gw),np.concatenate(rw))),len(admitted)


def evaluate(generated,catalog,reference,simulation,network,out):
    out.mkdir(parents=True,exist_ok=False)
    try:
        gen=json.loads((generated/'complete.json').read_text());assert gen['complete'] and gen['reference_opened'] is False
        for n,h in gen['artifacts'].items():assert digest(generated/n)==h
        assert gen['catalog_sha256']==digest(catalog/'routes.json')
        refreceipt,ref=read_pool(reference,'references.npz')
        simreceipt=json.loads((simulation/'simulation-complete.json').read_text())
        assert simreceipt['complete'] and digest(simulation/'vehroute.xml.gz')==simreceipt['files']['vehroute.xml.gz']['sha256']
        # Chain reference -> simulation -> exact prepared manifest used by catalog.
        assert refreceipt['source_sha256']['simulation_receipt']==digest(simulation/'simulation-complete.json')
        assert refreceipt['source_sha256']['requests']==json.loads((catalog/'complete.json').read_text())['request_manifest_sha256']
        pool=np.load(generated/'samples.npz');rows=json.loads((generated/'candidates.json').read_text())['rows']
        selections=json.loads((generated/'selections.json').read_text())['rows'];cat={r['id']:r for r in json.loads((catalog/'routes.json').read_text())['rows']}
        ods={r['id']:r for r in json.loads((catalog/'od.json').read_text())['requests']}
        rr={str(rid):dict(extract_record(ref['txy'][ref['offsets'][i]:ref['offsets'][i+1]]),request_id=str(rid),label=int(ref['labels'][i])) for i,rid in enumerate(ref['ids'])}
        actual={}
        for n in child_nodes(simulation/'vehroute.xml.gz'):
            if n.tag=='vehicle' and n.get('id') in rr:
                history=n.findall('routeDistribution/route') or n.findall('route')
                if history:actual[n.get('id')]=history[-1].get('edges').split()
        records=[dict(extract_record(pool['txy'][pool['offsets'][i]:pool['offsets'][i+1]]),**r) for i,r in enumerate(rows)]
        byrequest=defaultdict(list)
        for r in records:byrequest[r['request_id']].append(r)
        roads=RoadFeatures(network);lengths={vc:{e:roads.edge(vc,e)[3] for e in roads.graph(vc)} for vc in {r['vclass'] for r in ods.values()}}
        route_diagnostics={};results={}
        for c,name in enumerate(CLASSES):
            ids=[rid for rid in byrequest if cat[rid]['label']==c and rid in rr]
            routeids=[rid for rid in ids if rid in actual and all(e in lengths[ods[rid]['vclass']] for e in actual[rid])]
            ceilings=[];exact=0
            for rid in routeids:
                ll=lengths[ods[rid]['vclass']];ceiling=max(overlap(route['edges'],actual[rid],ll) for route in cat[rid]['routes']);ceilings.append(ceiling)
                exact+=any(route['edges']==actual[rid] for route in cat[rid]['routes'])
            route_diagnostics[name]={'requested':sum(r['label']==c for r in cat.values()),'generated':sum(r['label']==c and bool(r['routes']) for r in cat.values()),
                'physical_pairs':len(ids),'route_pairs':len(routeids),'catalog_contains_exact_reference':exact,
                'mean_best_catalog_overlap':float(np.mean(ceilings)) if ceilings else None}
        for method in METHODS:
            selected={s['request_id']:s for s in selections if s['method']==method};groups={}
            for c,name in enumerate(CLASSES):
                ids=[rid for rid,s in selected.items() if s['label']==c and rid in rr]
                chosen=[records[selected[rid]['index']] for rid in ids];refs=[rr[rid] for rid in ids]
                d={'requested':route_diagnostics[name]['requested'],'generated':sum(s['label']==c for s in selected.values()),'paired':len(ids),'physical':{}}
                for signal in ['speed','acceleration']:
                    d['selected_native_'+signal+'_w1']=weighted_signal_distance(chosen,refs,signal)
                    value,n=weighted_native([(rid,byrequest[rid],selected[rid]['weights']) for rid in ids],rr,signal)
                    d['weighted_native_'+signal+'_w1']=value;d['weighted_native_'+signal+'_support']=n
                if chosen:
                    for key in chosen[0]['physical']:
                        pairs=[(x['physical'][key],y['physical'][key]) for x,y in zip(chosen,refs) if x['physical'][key] is not None and y['physical'][key] is not None]
                        d['physical'][key]={'n':len(pairs),'w1':float(wasserstein_distance(*zip(*pairs))) if pairs else None,'mae':float(np.mean([abs(x-y) for x,y in pairs])) if pairs else None}
                gm=Counter();rm=Counter();ovs=[];detours=[];entropy=[];routechanges=0
                rcontrol={s['request_id']:s for s in selections if s['method']=='R'}
                allids=[rid for rid,s in selected.items() if s['label']==c]
                for rid in allids:
                    s=selected[rid];routes=cat[rid]['routes'];weights=s['route_weights'];entropy.append(s['effective_routes'])
                    detours.append(sum(w*r['detour_ratio'] for w,r in zip(weights,routes)))
                    routechanges+=records[s['index']]['route_index']!=records[rcontrol[rid]['index']]['route_index']
                    if rid not in rr or rid not in actual or any(e not in lengths[ods[rid]['vclass']] for e in actual[rid]):continue
                    ll=lengths[ods[rid]['vclass']];rm.update(edge_mass(actual[rid],ll))
                    ov=0.
                    for w,r in zip(weights,routes):
                        gm.update({e:w*v for e,v in edge_mass(r['edges'],ll).items()});ov+=w*overlap(r['edges'],actual[rid],ll)
                    ovs.append(ov)
                d.update(route_pairs=len(ovs),route_edge_usage_js=js(gm,rm) if ovs else None,
                    expected_reference_overlap=float(np.mean(ovs)) if ovs else None,
                    expected_detour=float(np.mean(detours)) if detours else None,
                    effective_route_count=float(np.mean(entropy)) if entropy else None,
                    selected_route_changes_vs_R=routechanges)
                groups[name]=d
            keys=['selected_native_speed_w1','selected_native_acceleration_w1','weighted_native_speed_w1','weighted_native_acceleration_w1','route_edge_usage_js','expected_detour','effective_route_count']
            results[method]={'by_class':groups,'macro':{k:float(np.mean([g[k] for g in groups.values()])) if all(g[k] is not None for g in groups.values()) else None for k in keys}}
        save(out/'complete.json',{'complete':True,'seed':gen['seed'],'generation_sha256':digest(generated/'complete.json'),
            'reference_sha256':digest(reference/'complete.json'),'catalog_coverage':route_diagnostics,'results':results,
            'scope':'New native-demand simulation; one fixed public itinerary per OD; alternative-path overlap is descriptive, not correctness.',
            'primary_weighting':'class macro; equal request, policy candidate probability, physical time within track',
            'native_weighted_support':'candidate weight renormalized only for missing acceleration support; denominators reported',
            'edge_distribution':'equal request; road declared lengths and edge-occurrence counts; endpoint partial-edge lengths not resolved'})
    except BaseException as e:save(out/'failed.json',{'error':repr(e)});raise


def checks():
    lengths={'a':2.,'b':3.};a=edge_mass(['a','a','b'],lengths)
    assert np.isclose(a['a'],4/7) and js(a,a)==0
    assert np.isclose(overlap(['a','a','b'],['a','b'],lengths),5/7)
    assert np.isclose(js({'a':1},{'b':1}),np.log(2))
    records=[{'speed':np.array([1.,2.]),'speed_weight':np.array([.5,.5])}, {'speed':np.array([9.]),'speed_weight':np.array([1.])}]
    value,n=weighted_native([('id',records,[1.,0.])],{'id':records[0]},'speed');assert value==0 and n==1
    print('G38 evaluation checks passed: occurrence weights, zero self-distance, JS disjoint support, zero-weight candidates')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['evaluate','checks'])
    for name in ['generated','catalog','reference','simulation','network','output']:p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.action=='checks':checks()
    else:evaluate(a.generated,a.catalog,a.reference,a.simulation,a.network,a.output)
