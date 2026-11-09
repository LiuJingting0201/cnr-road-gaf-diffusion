"""Preserve old server worktree changes and full history without modifying it."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import archive_supplement_upload as supplement
from archive_release import API
from archive_cloud_upload import save

ROOT=Path('/tmp/cnr-legacy-source-archive-20261109')
SOURCE=Path('/workspace/cnr-road-gaf-diffusion')
PATTERNS=[rb'gh[pousr]_[A-Za-z0-9]{25,}',rb'github_pat_[A-Za-z0-9_]{30,}',rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',rb'AKIA[A-Z0-9]{16}']


def git(*args):return subprocess.check_output(['git','-C',str(SOURCE),*args])


def main():
    ROOT.mkdir(exist_ok=True);api=API(sys.stdin.readline().strip());api.private();release=api.release();start=time.monotonic();supplement.ROOT=ROOT
    with (ROOT/'started.json').open('x') as f:json.dump({'started_unix':time.time()},f)
    try:
        objects=[x.split()[0] for x in git('rev-list','--objects','--all').splitlines()]
        # Batch input is a short list of object IDs, not credentials or content.
        ids=ROOT/'object-ids.txt';ids.write_bytes(b'\n'.join(objects)+b'\n');scanned=0
        with ids.open('rb') as inp:
            proc=subprocess.Popen(['git','-C',str(SOURCE),'cat-file','--batch'],stdin=inp,stdout=subprocess.PIPE)
            for oid in objects:
                header=proc.stdout.readline().split();assert len(header)==3
                size=int(header[2]);remaining=size;tail=b'';hit=False
                while remaining:
                    b=proc.stdout.read(min(4*1024**2,remaining));assert b;remaining-=len(b)
                    if header[1]==b'blob':
                        sample=tail+b;hit|=any(re.search(p,sample) for p in PATTERNS);tail=sample[-256:]
                assert proc.stdout.read(1)==b'\n'
                assert not hit,'Possible credential signature in historical blob '+oid.decode()
                scanned+=header[1]==b'blob'
            assert proc.wait()==0
        bundle=ROOT/'legacy-server-history.bundle';subprocess.run(['git','-C',str(SOURCE),'bundle','create',str(bundle),'--all'],check=True)
        subprocess.run(['git','-C',str(SOURCE),'bundle','verify',str(bundle)],check=True,stdout=subprocess.DEVNULL)
        info=ROOT/'legacy-worktree-state.json';save(info,{'head':git('rev-parse','HEAD').decode().strip(),'status':git('status','--porcelain').decode().splitlines(),
            'refs':git('for-each-ref','--format=%(refname) %(objectname)').decode().splitlines(),'historical_blobs_scanned':scanned,'credential_signature_matches':0,'source_worktree_modified_by_archive':False})
        roots=ROOT/'roots.json';save(roots,{'legacy/source':str(SOURCE)});plan=ROOT/'plan.json'
        subprocess.run(['python3',str(supplement.PACKER),'plan','--roots',str(roots),'--plan',str(plan)],check=True)
        d=json.loads(plan.read_text());rows=[r for r in d['files'] if not r['path'].startswith(('legacy/source/data/','legacy/source/runs/'))]
        s=info.stat();rows.append({'source':str(info),'path':'legacy/source/ARCHIVE_WORKTREE_STATE.json','group':'04-legacy','bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'mode':s.st_mode&0o777,'sha256':supplement.sha(info)})
        seen={}
        for r in rows:
            r.pop('duplicate_of',None);key=(r['bytes'],r['sha256'])
            if key in seen:r['duplicate_of']=seen[key]
            else:seen[key]=r['path']
        d['files']=rows;d['summary']={'files':len(rows),'logical_bytes':sum(r['bytes'] for r in rows),'unique_bytes':sum(r['bytes'] for r in rows if 'duplicate_of' not in r)}
        d['scope']='Full legacy working source/logs and uncommitted changes; data/runs are already in SERVER_MANIFEST. No original worktree modifications.'
        save(plan,d);result=supplement.upload_plan(api,release,plan,'legacy-source','LEGACY_SOURCE_MANIFEST.json')
        a=api.upload(release,bundle);history={k:a[k] for k in ('id','name','size','digest','state')};save(ROOT/'legacy-server-history.bundle.receipt.json',history)
        save(ROOT/'complete.json',{'complete':True,'seconds':time.monotonic()-start,'source':result,'history':history,'historical_blobs_scanned':scanned})
    except BaseException as e:save(ROOT/'failed.json',{'error':repr(e),'seconds':time.monotonic()-start});raise


if __name__=='__main__':main()
