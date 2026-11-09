"""Regression: provenance model hashes are not local generated artifacts."""
import json
from pathlib import Path
import tempfile
from g39_run import digest,verify


def rejected(call):
    try:call()
    except (AssertionError,FileNotFoundError):return
    raise AssertionError('Corrupt/missing artifact unexpectedly accepted')


def checks():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);model=root/'model';pool=root/'pool'
        model.mkdir();pool.mkdir()
        for name in ('model.pt','config.json','stats.json'):(model/name).write_bytes(name.encode())
        mr={'complete':True,**{k:digest(model/n) for k,n in
            [('model_sha256','model.pt'),('config_sha256','config.json'),('stats_sha256','stats.json')]}}
        (model/'complete.json').write_text(json.dumps(mr));verify(model)
        (pool/'samples.npz').write_bytes(b'samples')
        gr={'complete':True,'model_sha256':mr['model_sha256'],
            'artifacts':{'samples.npz':digest(pool/'samples.npz')}}
        (pool/'complete.json').write_text(json.dumps(gr))
        verify(pool,{'model_sha256':model/'model.pt'})
        assert not (pool/'model.pt').exists()
        rejected(lambda:verify(pool,{'model_sha256':model/'stats.json'}))
        (pool/'samples.npz').write_bytes(b'changed');rejected(lambda:verify(pool))
        (pool/'samples.npz').unlink();rejected(lambda:verify(pool))
        (pool/'samples.npz').write_bytes(b'samples')
        gr['artifacts']['samples.npz']={'sha256':digest(pool/'samples.npz'),'bytes':7}
        (pool/'complete.json').write_text(json.dumps(gr));verify(pool)
        (pool/'failed.json').write_text('{}');rejected(lambda:verify(pool));(pool/'failed.json').unlink()
        (model/'config.json').write_text('tampered');rejected(lambda:verify(model))
        (model/'config.json').write_bytes(b'config.json');(model/'model.pt').unlink();rejected(lambda:verify(model))
        utility=root/'utility';utility.mkdir()
        (utility/'model.pt').write_bytes(b'model');(utility/'predictions.npz').write_bytes(b'predictions')
        ur={'complete':True,'model_sha256':digest(utility/'model.pt'),
            'predictions_sha256':digest(utility/'predictions.npz')}
        (utility/'complete.json').write_text(json.dumps(ur));verify(utility)
        (utility/'predictions.npz').write_bytes(b'changed');rejected(lambda:verify(utility))
    print('G39 receipt checks passed: provenance, both manifest forms, corruption/missing files, failed marker, model and utility integrity')


if __name__=='__main__':checks()
