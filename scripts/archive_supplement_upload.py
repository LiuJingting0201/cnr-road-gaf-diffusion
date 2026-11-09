"""Privately archive regenerable GAF caches and the closed G38 snapshot."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from archive_release import API
from archive_cloud_upload import save

ROOT=Path('/tmp/cnr-archive-supplements-20261109')
PACKER=Path('/tmp/cnr_archive_pack.py')
G38=Path('/workspace/cnr-experiments/gaf-inloop-v1/g38-job')
PART_BYTES=512*1024**2


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def cache_plan():
    initial=json.loads(Path('/tmp/cnr-archive-plan.json').read_text());rows=[];seen={}
    for r in sorted(initial['excluded'],key=lambda r:r['path']):
        if r['reason']!='regenerable-image-cache':continue
        prefix=max((x for x in initial['roots'] if r['path'].startswith(x+'/')),key=len)
        path=Path(initial['roots'][prefix])/r['path'][len(prefix)+1:]
        before=path.stat();digest=sha(path);after=path.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        row={'path':r['path'],'source':str(path),'group':'cache','bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'mode':before.st_mode&0o777,'sha256':digest}
        key=(before.st_size,digest)
        if key in seen:row['duplicate_of']=seen[key]
        else:seen[key]=row['path']
        rows.append(row)
    assert rows
    result={'schema':'cnr-private-archive-v1','files':rows,'roots':initial['roots'],'excluded':[],'symlinks':[],
            'summary':{'files':len(rows),'logical_bytes':sum(r['bytes'] for r in rows),'unique_bytes':sum(r['bytes'] for r in rows if 'duplicate_of' not in r)},
            'scope':'Supplement includes all regenerable-image-cache entries excluded by the initial cloud plan; byte-level backup only.'}
    path=ROOT/'cache-plan.json';save(path,result);return path


def upload_plan(api,release,plan_path,name,manifest_name):
    plan=json.loads(plan_path.read_text());archives=[]
    def upload(path):
        a=api.upload(release,path);r={k:a[k] for k in ('id','name','size','digest','state')}
        save(ROOT/(path.name+'.receipt.json'),r);path.unlink();return r
    for group in sorted({r['group'] for r in plan['files'] if 'duplicate_of' not in r}):
        h=hashlib.sha256();size=0;parts=[];pending=[];i=0
        with (ROOT/(name+'-'+group+'.pack.log')).open('wb') as log, cf.ThreadPoolExecutor(max_workers=2) as pool:
            proc=subprocess.Popen(['nice','-n','15','python3',str(PACKER),'stream','--plan',str(plan_path),'--group',group],stdout=subprocess.PIPE,stderr=log)
            try:
                while True:
                    assert shutil.disk_usage(ROOT).free>16*1024**3,'Free-space guard'
                    path=ROOT/f'{name}-{group}.tar.gz.part{i:04d}';written=0
                    with path.open('xb') as f:
                        while written<PART_BYTES:
                            b=proc.stdout.read(min(4*1024**2,PART_BYTES-written))
                            if not b:break
                            f.write(b);h.update(b);size+=len(b);written+=len(b)
                    if not written:path.unlink();break
                    pending.append(pool.submit(upload,path));i+=1
                    if len(pending)>=2:parts.append(pending.pop(0).result())
                    save(ROOT/'progress.json',{'archive':name,'group':group,'produced_parts':i,'compressed_bytes':size})
                assert proc.wait()==0,'Snapshot stream failed'
                parts.extend(f.result() for f in pending)
            except BaseException:
                proc.terminate();raise
        archives.append({'group':group,'compressed_bytes':size,'stream_sha256':h.hexdigest(),'parts':sorted(parts,key=lambda r:r['name'])})
    manifest=ROOT/manifest_name;save(manifest,{**plan,'archives':archives});manifest_receipt=upload(manifest)
    result={'archives':archives,'manifest':manifest_receipt,'source_summary':plan['summary']};save(ROOT/(name+'.complete.json'),result);return result


def main():
    ROOT.mkdir(exist_ok=True);api=API(sys.stdin.readline().strip());api.private();release=api.release();start=time.monotonic()
    with (ROOT/'started.json').open('x') as f:json.dump({'pid':os.getpid(),'started_unix':time.time()},f)
    try:
        cache=upload_plan(api,release,cache_plan(),'gaf-cache','CACHE_MANIFEST.json')
        deadline=time.monotonic()+3600
        while not any((G38/'g38-data'/n).exists() for n in ('complete.json','failed.json')):
            if time.monotonic()>deadline:raise RuntimeError('G38 snapshot not closed within bounded wait')
            time.sleep(20)
        time.sleep(5)
        roots=ROOT/'g38-roots.json';save(roots,{'experiments/g38-job':str(G38)})
        plan=ROOT/'g38-plan.json'
        # A failed closed run remains archived. The ordinary planner defers active
        # files until complete; require a successful closure for this supplement.
        assert (G38/'g38-data/complete.json').exists(),'G38 failed: preserve its failure; final snapshot needs review'
        subprocess.run(['python3',str(PACKER),'plan','--roots',str(roots),'--plan',str(plan)],check=True)
        g38=upload_plan(api,release,plan,'g38-final','G38_MANIFEST.json')
        save(ROOT/'complete.json',{'complete':True,'seconds':time.monotonic()-start,'cache':cache,'g38':g38})
    except BaseException as e:save(ROOT/'failed.json',{'error':repr(e),'seconds':time.monotonic()-start});raise


if __name__=='__main__':main()
