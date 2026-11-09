"""Bounded cloud streaming/upload queue. Authentication remains only in memory."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

from archive_release import API,REPO,TAG

ROOT=Path('/tmp/cnr-github-archive-20261109')
PART_BYTES=512*1024**2


def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
    ROOT.mkdir(exist_ok=True);secret=sys.stdin.readline().strip();assert secret
    api=API(secret);api.private();release=api.release();plan=Path('/tmp/cnr-archive-plan.json')
    d=json.loads(plan.read_text());groups=sorted({r['group'] for r in d['files'] if 'duplicate_of' not in r})
    start=time.monotonic();archive=[]
    def upload(path):
        asset=api.upload(release,path);receipt={k:asset[k] for k in ['id','name','size','digest','state']}
        save(ROOT/(path.name+'.receipt.json'),receipt)
        path.unlink() # This queue's verified temporary chunk only, never source data.
        return receipt
    try:
        with (ROOT/'started.json').open('x') as f:json.dump({'started_unix':time.time(),'pid':os.getpid(),'groups':groups,'part_bytes':PART_BYTES,'parallel_uploads':4},f)
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            for group in groups:
                log=(ROOT/(group+'.pack.log')).open('wb')
                proc=subprocess.Popen(['nice','-n','15','python3','/tmp/cnr_archive_pack.py','stream','--plan',str(plan),'--group',group],stdout=subprocess.PIPE,stderr=log)
                pending=[];items=[];number=0;stream_hash=hashlib.sha256();bytes_total=0
                try:
                    while True:
                        if shutil.disk_usage(ROOT).free<16*1024**3:raise RuntimeError('Archive staging free-space guard')
                        part=ROOT/f'{group}.tar.gz.part{number:04d}';written=0
                        with part.open('xb') as f:
                            while written<PART_BYTES:
                                chunk=proc.stdout.read(min(4*1024**2,PART_BYTES-written))
                                if not chunk:break
                                f.write(chunk);written+=len(chunk);stream_hash.update(chunk);bytes_total+=len(chunk)
                        if not written:part.unlink();break
                        pending.append(pool.submit(upload,part));number+=1
                        # At most four uploading chunks plus one producer chunk.
                        if len(pending)>=4:items.append(pending.pop(0).result())
                        save(ROOT/'progress.json',{'group':group,'parts_produced':number,'group_bytes':bytes_total,'uploaded_parts':len(items),'seconds':time.monotonic()-start})
                    assert proc.wait()==0,'Archive producer failed; inspect preserved pack log'
                    items.extend(f.result() for f in pending)
                except BaseException:
                    proc.terminate();raise
                finally:log.close()
                archive.append({'group':group,'compressed_bytes':bytes_total,'stream_sha256':stream_hash.hexdigest(),'parts':sorted(items,key=lambda r:r['name'])})
                save(ROOT/(group+'.complete.json'),archive[-1]);print(json.dumps({'group_complete':group,'bytes':bytes_total,'seconds':time.monotonic()-start}),flush=True)
        export={**d,'archives':archive}
        # Original absolute source paths are provenance; extraction uses only 'path'.
        manifest=ROOT/'SERVER_MANIFEST.json';save(manifest,export);upload(manifest)
        save(ROOT/'complete.json',{'complete':True,'seconds':time.monotonic()-start,'archives':archive,'source_summary':d['summary'],'release_id':release['id'],
            'publication_pending_local_deliverables_and_final_verification':True})
    except BaseException as e:
        save(ROOT/'failed.json',{'error':repr(e),'seconds':time.monotonic()-start});raise


if __name__=='__main__':main()
