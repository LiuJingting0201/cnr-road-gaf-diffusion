"""Rebuild the completed utility cache once and require exact array equality."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path('performance-data');original=Path('../g39ur-job/g39u-data/cache')
    candidate=root/'utility-cache-optimized';receipt=root/'utility-cache-equivalence.json'
    assert not receipt.exists() and not candidate.exists()
    assert not (root/'scientific-review.json').exists(),'Keep CPU reconstruction outside GPU benchmark window'
    for p in ['../g39r-job/g39-data','../g39ur-job/g39u-data','../g39fr-job/g39f-data']:
        assert json.loads((Path(p)/'complete.json').read_text())['complete']
        assert not (Path(p)/'failed.json').exists()
    started=time.monotonic();script=Path('utility-candidate/g39_utility.py')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        PYTHONPATH='src:g10-env:original/scripts:../scripts:../vendor/original_cnr/src')
    try:
        with (root/'utility-cache-equivalence.log').open('xb') as log:
            proc=subprocess.Popen(['.venv/bin/python',str(script),'prepare',
                '--root','../g39r-job/g39-data','--data','../g35d-job/g35-data/direct-legs',
                '--output',str(candidate)],env=env,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise
            assert code==0,f'Candidate cache exited{code}'
        old=json.loads((original/'complete.json').read_text());new=json.loads((candidate/'complete.json').read_text())
        assert old['complete'] and new['complete']
        for k in old:
            if k!='artifacts':assert old[k]==new[k],k
        assert set(old['artifacts'])==set(new['artifacts'])
        arrays=0;files={}
        for oldfile in sorted(original.glob('*.npz')):
            newfile=candidate/oldfile.name
            for p,r in [(oldfile,old),(newfile,new)]:
                if p.name in r['artifacts']:assert sha(p)==r['artifacts'][p.name]['sha256']
            with np.load(oldfile,allow_pickle=False) as a,np.load(newfile,allow_pickle=False) as b:
                assert set(a.files)==set(b.files)
                for k in a.files:
                    x,y=a[k],b[k]
                    assert x.dtype==y.dtype and np.array_equal(x,y),(oldfile.name,k)
                    arrays+=1
                files[oldfile.name]={'arrays':len(a.files),'exact':True}
        receipt.write_text(json.dumps({'complete':True,'seconds':time.monotonic()-started,
            'all_arrays_bitwise_equal':True,'checked_arrays':arrays,'files':files,
            'normalizers_and_metadata_equal':True,'candidate_source_sha256':sha(script),
            'original_receipt_sha256':sha(original/'complete.json'),
            'new_receipt_sha256':sha(candidate/'complete.json'),
            'limits':'One completed cache rebuilt; historical preparation was not an instrumented paired benchmark. No speedup ratio claimed.',
            'no_training_or_new_sampling':True},indent=2)+'\n')
        print(receipt.read_text())
    except BaseException as error:
        (root/'utility-cache-equivalence-failed.json').write_text(json.dumps({'complete':False,
            'error':repr(error),'seconds':time.monotonic()-started},indent=2)+'\n');raise


if __name__=='__main__':main()
