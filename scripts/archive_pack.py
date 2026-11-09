"""Byte-preserving archive plans and streamed tar.gz; never parse scientific targets."""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile

SKIP_DIRS={'.git','.venv','g10-env','__pycache__','.pytest_cache','node_modules','.ssh','.aws','.codex','.agents','github-archive-20261109'}
SECRET_NAMES={'.env','.netrc','.git-credentials','id_rsa','id_ed25519','credentials','credentials.json'}
SECRET_PATTERNS=[rb'gh[pousr]_[A-Za-z0-9]{25,}',rb'github_pat_[A-Za-z0-9_]{30,}',rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',rb'AKIA[A-Z0-9]{16}']


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def group(name):
    if name.startswith('legacy/'):return '04-legacy'
    if name.startswith('local/'):return '00-local-deliverables'
    if 'g35d-job/g35-data/models/' in name or 'g35d-job/g35-data/teachers/' in name or 'g38-job/g38-data/models/' in name:return '00-current-models'
    if any('/'+x+'/' in name for x in ['g33-job','g36-job','g37-job','g38-job']):return '01-recent-experiments'
    if '/g35d-job/' in name or '/g35b-job/g35-data/native/val/' in name:return '02-current-training-data'
    if any('/'+x+'/' in name for x in ['g35-job','g35b-job','g35c-job']):return '05-interrupted-attempts'
    return '03-historical-experiments'


def exclusion(path,relative):
    if path.name in SECRET_NAMES or path.suffix in {'.pem','.key'}:return 'credential-name-excluded'
    if path.name=='.DS_Store' or path.name.endswith('.pyc'):return 'system-cache'
    if path.name.endswith('.partial'):return 'incomplete-transfer; original server artifact separately preserved'
    # These image caches are derived deterministically; source trajectories,
    # encoder parameters, cache indices, receipts and all trained models remain.
    if 'images' in path.name and (path.name.endswith('.npy') or path.name.endswith('.npy.gz')):return 'regenerable-image-cache'
    if 'gaf-cache' in path.parts and (path.suffix=='.npy' or path.name.startswith('spatial-')):return 'regenerable-image-cache'
    if '/g38-job/' in relative:
        job=Path(str(path).split('/g38-job/')[0])/'g38-job'
        if not (job/'g38-data/complete.json').exists():
            inside=path.relative_to(job)
            if inside.parts[0] in {'scripts','docs'}:return None
            if len(inside.parts)>1 and inside.parts[1] in {'models','policy-data','policy-evaluation'}:return None
            return 'active-experiment-snapshot-deferred'
    return None


def plan(roots,out):
    candidates=[];excluded=[];symlinks=[]
    for prefix,root in roots.items():
        root=Path(root).resolve()
        for folder,dirs,files in os.walk(root,followlinks=False):
            kept=[]
            for d in dirs:
                p=Path(folder)/d
                if p.is_symlink():symlinks.append({'path':prefix+'/'+str(p.relative_to(root)),'target':os.readlink(p)});continue
                if d not in SKIP_DIRS:kept.append(d)
            dirs[:]=kept
            for name in files:
                p=Path(folder)/name;relative=prefix+'/'+str(p.relative_to(root))
                if p.is_symlink():symlinks.append({'path':relative,'target':os.readlink(p)});continue
                if name=='.git':continue
                reason=exclusion(p,relative)
                if reason:excluded.append({'path':relative,'bytes':p.stat().st_size,'reason':reason});continue
                if p.stat().st_size<16*1024**2 and p.suffix.lower() in {'.py','.sh','.json','.jsonl','.md','.txt','.log','.toml','.ini','.yaml','.yml'}:
                    b=p.read_bytes()
                    if any(re.search(pattern,b) for pattern in SECRET_PATTERNS):
                        excluded.append({'path':relative,'bytes':len(b),'reason':'possible-credential-content; manual review required'});continue
                candidates.append({'source':str(p),'path':relative,'group':group(relative)})
    rows=[];by_hash={}
    for i,row in enumerate(sorted(candidates,key=lambda x:(x['group'],x['path']))):
        p=Path(row['source']);before=p.stat();h=digest(p);after=p.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Changed during snapshot: '+row['path'])
        row.update(bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=h,mode=before.st_mode & 0o777)
        key=(before.st_size,h)
        if key in by_hash:row['duplicate_of']=by_hash[key]
        else:by_hash[key]=row['path']
        rows.append(row)
        if i%500==0:print(json.dumps({'hashed':i,'total':len(candidates)}),file=sys.stderr,flush=True)
    result={'schema':'cnr-private-archive-v1','roots':roots,'files':rows,'excluded':excluded,'symlinks':symlinks,
        'summary':{'files':len(rows),'logical_bytes':sum(r['bytes'] for r in rows),'unique_bytes':sum(r['bytes'] for r in rows if 'duplicate_of' not in r),
                   'excluded_bytes':sum(r['bytes'] for r in excluded),'excluded_reasons':dict(Counter(r['reason'] for r in excluded))},
        'scientific_data_policy':'Byte-level backup only; no target decoding, fitting, model selection or evaluation.',
        'restore':'Extract unique files, verify every SHA256, then copy duplicate_of targets to alias paths; environment symlinks require reinstall.'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary']),file=sys.stderr)


class CheckedReader:
    def __init__(self,path):self.f=path.open('rb');self.h=hashlib.sha256()
    def read(self,n=-1):
        b=self.f.read(n);self.h.update(b);return b
    def close(self):self.f.close()


def stream(plan_path,group_name):
    d=json.loads(plan_path.read_text());selected=[r for r in d['files'] if r['group']==group_name and 'duplicate_of' not in r]
    assert selected
    with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1,mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tar:
            for row in selected:
                p=Path(row['source']);s=p.stat()
                if s.st_size!=row['bytes'] or s.st_mtime_ns!=row['mtime_ns']:raise RuntimeError('Snapshot changed: '+row['path'])
                info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mode=row['mode'];info.mtime=0
                reader=CheckedReader(p)
                try:tar.addfile(info,reader)
                finally:reader.close()
                if reader.h.hexdigest()!=row['sha256']:raise RuntimeError('Content changed: '+row['path'])
    print(json.dumps({'complete':True,'group':group_name,'files':len(selected)}),file=sys.stderr,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','stream']);p.add_argument('--roots',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--group');a=p.parse_args()
    if a.action=='plan':plan(json.loads(a.roots.read_text()),a.plan)
    else:stream(a.plan,a.group)
