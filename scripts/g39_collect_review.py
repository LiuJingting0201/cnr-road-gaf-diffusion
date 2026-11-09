"""Verify completed G39 aggregate evidence; export no individual trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(path.read_text())


def verified(folder):
    receipt=read(folder/'complete.json')
    assert receipt['complete'] and not (folder/'failed.json').exists(),str(folder)
    manifest=receipt.get('artifacts')
    if manifest is not None:
        for name,spec in manifest.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert sha(folder/name)==(spec['sha256'] if isinstance(spec,dict) else spec)
    else:
        for key,name in [('model_sha256','model.pt'),('config_sha256','config.json'),
                         ('stats_sha256','stats.json'),('predictions_sha256','predictions.npz')]:
            if key in receipt:assert sha(folder/name)==receipt[key]
    return receipt


def collect(base,out,all_stages):
    main=base/'g39r-job/g39-data'
    root=verified(main);report=verified(main/'report')
    assert root['evaluation_cells']==112 and root['route_duration_equal_across_arms']
    assert sha(main/'report/complete.json')==root['report_sha256']
    assert report['cells']==112 and report['rows']==560
    arms=('B','S','V','SV','P','SH','RAW');coverage={};diagnostics={}
    for episode in range(390201,390209):
        for seed in (42,43):
            pair=[]
            for arm in arms:
                name=f'{arm}-{seed}';key=f'{episode}/evaluation/{name}/complete.json'
                folder=main/str(episode)/'evaluation'/name
                evaluation=verified(folder);genfolder=main/str(episode)/'generated'/name
                gen=verified(genfolder)
                assert sha(folder/'complete.json')==report['source_sha256'][key]
                assert evaluation['generation_sha256']==sha(genfolder/'complete.json')
                assert gen['model_sha256']==sha(main/'models'/name/'model.pt')
                assert gen['catalog_sha256']==sha(main/str(episode)/'catalog/routes.json')
                assert evaluation['reference_sha256']==sha(main/str(episode)/'reference/complete.json')
                diagnostic=read(folder/'diagnostics.json')
                assert diagnostic['complete'] and diagnostic['generated_sha256']==sha(genfolder/'complete.json')
                assert diagnostic['training_cache_sha256']==sha(main/'raw-cache/complete.json')
                diagnostics[f'{episode}/{name}']=diagnostic
                pair.append(gen['route_duration_pairing_sha256'])
                if name=='B-42':coverage[str(episode)]=evaluation['catalog_coverage']
            assert len(set(pair))==1
    payload={'complete':True,'full_batch':all_stages,'main':root,'coverage':coverage,
             'diagnostics':diagnostics,'receipt_sha256':{'../g39r-job/g39-data':sha(main/'complete.json')},
             'verified_generated_artifacts':112,'individual_records_exported':False}
    if all_stages:
        utility=base/'g39ur-job/g39u-data';ur=verified(utility)
        assert ur['fit_count']==26 and ur['source_G39_sha256']==sha(main/'complete.json')
        for name,expected in ur['results'].items():assert verified(utility/'fits'/name)==expected
        factorial=base/'g39fr-job/g39f-data';fr=verified(factorial)
        assert len(fr['results'])==4 and fr['same_expanded_pool_for_all_methods']
        generated={}
        for key,expected in fr['results'].items():
            episode,seed=key.split('/');folder=factorial/episode/'generated'/seed
            gr=verified(folder);actual=verified(factorial/episode/'evaluation'/seed)
            assert actual==expected and actual['generation_sha256']==sha(folder/'complete.json')
            assert gr['joint_route_duration_max_abs_error']<1e-12
            # Aggregate counts only: do not export fallback request IDs.
            generated[key]={k:gr[k] for k in ['requested','generated','candidates','seconds',
                'joint_route_duration_max_abs_error','network_forwards','reference_opened']}
        payload.update(utility=ur,factorial=fr,factorial_generation=generated)
        payload['receipt_sha256'].update({'../g39ur-job/g39u-data':sha(utility/'complete.json'),
                                         '../g39fr-job/g39f-data':sha(factorial/'complete.json')})
    out.mkdir(parents=True,exist_ok=False)
    for name in ['complete.json','REPORT.md','all-class-metrics.csv']:
        shutil.copy2(main/'report'/name,out/('main-'+name))
    (out/'verified-aggregates.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'full_batch':all_stages,'verified_main_cells':112,
                      'utility_fits':26 if all_stages else None,'factorial_cells':4 if all_stages else None}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--all-stages',action='store_true')
    a=p.parse_args();collect(a.base,a.output,a.all_stages)
