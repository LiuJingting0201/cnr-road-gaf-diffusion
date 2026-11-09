"""Verify closed G40 sources and export small aggregate evidence, not trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
from g33_train import digest,save
from g33_teachers import metrics
from g39_run import verify
from g40_model import ARMS
from g40_requests import EPISODES
from g40_utility import REGIMES


def main(a):
    done=json.loads((a.root/'all-stages-complete.json').read_text());assert done['complete']
    assert not (a.root/'failed.json').exists()
    assert digest(a.root/'complete.json')==done['primary_receipt_sha256']
    assert digest(a.utility/'complete.json')==done['utility_receipt_sha256']
    main=verify(a.root);assert main['evaluation_cells']==128 and main['route_duration_equal_across_arms']
    report=verify(a.root/'report');assert digest(a.root/'report/complete.json')==main['report_sha256']
    models={};priors={};configs={};evaluations={};diagnostics={};coverage={};generations={}
    for arm in ARMS:
        for seed in (42,43):
            name=f'{arm}-{seed}';models[name]=verify(a.root/'models'/name);priors[name]=verify(a.root/'priors'/name)
            configs[name]=json.loads((a.root/'models'/name/'config.json').read_text())
            assert priors[name]['motion_model_sha256']==models[name]['model_sha256']
            assert priors[name]['training_codes_sha256']==models[name]['codes_sha256']
    for e in EPISODES:
        folder=a.root/str(e);ref=verify(folder/'reference')
        catalog_hash=digest(folder/'catalog/routes.json')
        cohort=json.loads((folder/'requests/g40-cohort.json').read_text());assert cohort['training_nonbus_parent_overlap']==0
        for seed in (42,43):
            pairs=[]
            for arm in ARMS:
                name=f'{arm}-{seed}';key=f'{e}/{name}';g=folder/'generated'/name;v=folder/'evaluation'/name
                gen=verify(g)
                assert gen['model_sha256']==models[name]['model_sha256']
                assert gen['prior_sha256']==priors[name]['model_sha256'] and gen['catalog_sha256']==catalog_hash
                result=verify(v,{'generation_sha256':g/'complete.json','reference_sha256':folder/'reference/complete.json'})
                diag=json.loads((v/'diagnostics.json').read_text());assert diag['complete'] and diag['generated_sha256']==digest(g/'complete.json')
                assert not gen['true_future_GAF_opened'] and not gen['reference_opened']
                pairs.append(gen['route_duration_pairing_sha256']);evaluations[key]=result;diagnostics[key]=diag;generations[key]=gen
                if name=='B-42':coverage[str(e)]=result['catalog_coverage']
            assert len(set(pairs))==1
    utility=verify(a.utility);assert utility['fit_count']==30
    cache=verify(a.utility/'cache');assert cache['source_G40_sha256']==digest(a.root/'complete.json')
    with np.load(a.utility/'cache/evaluation.npz',allow_pickle=False) as f:
        expected_y=f['y'];expected_ids=f['ids'];expected_parents=f['source_parent']
    for name,r in utility['results'].items():
        folder=a.utility/'fits'/name;verify(folder)
        with np.load(folder/'predictions.npz',allow_pickle=False) as f:
            assert np.array_equal(f['y'],expected_y)
            assert np.array_equal(f['ids'],expected_ids) and np.array_equal(f['source_parent'],expected_parents)
            computed=metrics(f['y'],f['logits'])
        assert computed==r['pooled'],name
    a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/'verified-aggregates.json',{'complete':True,'all_stages':done,'main':main,'report':report,'models':models,
        'priors':priors,'configs':configs,'evaluations':evaluations,'diagnostics':diagnostics,'coverage':coverage,
        'generations':generations,'utility':utility,'utility_cache':cache,'source_sha256':digest(__file__),
        'utility_evaluation_rows':len(expected_y),'utility_metrics_independently_recomputed':True,
        'raw_trajectories_exported':False,'weights_exported':False})
    save(a.output/'complete.json',{'complete':True,'artifact_sha256':digest(a.output/'verified-aggregates.json'),
        'models':16,'priors':16,'generated_evaluation_cells':128,'utility_fits':30,
        'all_source_artifacts_verified':True,'utility_metrics_independently_recomputed':True})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','utility','output'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
