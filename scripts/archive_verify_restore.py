"""Verify downloaded archive parts and safely restore regular files/duplicates."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import tarfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


class Concat:
    def __init__(self,paths):self.paths=iter(paths);self.f=None
    def read(self,n=-1):
        assert n>=0
        pieces=[]
        while n:
            if self.f is None:
                try:self.f=next(self.paths).open('rb')
                except StopIteration:break
            chunk=self.f.read(n)
            if chunk:pieces.append(chunk);n-=len(chunk)
            else:self.f.close();self.f=None
        return b''.join(pieces)
    def close(self):
        if self.f:self.f.close()


def main(a):
    d=json.loads(a.manifest.read_text());records={r['path']:r for r in d['files']};seen=set()
    if a.archive:groups=[{'parts':[{'name':a.archive.name,'digest':'sha256:'+sha(a.archive)}]}];directory=a.archive.parent
    else:groups=d['archives'];directory=a.directory
    if a.output:
        assert not a.output.exists(),'Restore into a fresh directory only';a.output.mkdir(parents=True)
    for group in groups:
        parts=[directory/p['name'] for p in group['parts']];whole=hashlib.sha256()
        for path,item in zip(parts,group['parts']):
            assert 'sha256:'+sha(path)==item['digest'],path
            with path.open('rb') as f:
                for b in iter(lambda:f.read(4*1024**2),b''):whole.update(b)
        if 'stream_sha256' in group:assert whole.hexdigest()==group['stream_sha256']
        reader=Concat(parts)
        try:
            with tarfile.open(fileobj=reader,mode='r|gz') as tar:
                for member in tar:
                    p=PurePosixPath(member.name)
                    assert member.isfile() and not p.is_absolute() and '..' not in p.parts
                    name=str(p);assert name in records and name not in seen
                    expected=records[name];assert 'duplicate_of' not in expected and member.size==expected['bytes']
                    target=a.output.joinpath(*p.parts) if a.output else None
                    if target:target.parent.mkdir(parents=True,exist_ok=True)
                    output=target.open('xb') if target else None;h=hashlib.sha256()
                    try:
                        stream=tar.extractfile(member)
                        for b in iter(lambda:stream.read(4*1024**2),b''):
                            h.update(b)
                            if output:output.write(b)
                    finally:
                        if output:output.close()
                    assert h.hexdigest()==expected['sha256'],name;seen.add(name)
                    if target:target.chmod(expected.get('mode',0o644)&0o777)
        finally:reader.close()
    expected={r['path'] for r in records.values() if 'duplicate_of' not in r}
    assert seen==expected,{'missing':len(expected-seen),'extra':len(seen-expected)}
    if a.output:
        for name,r in records.items():
            if 'duplicate_of' in r:
                p=PurePosixPath(name);canonical=PurePosixPath(r['duplicate_of'])
                assert not p.is_absolute() and '..' not in p.parts and not canonical.is_absolute() and '..' not in canonical.parts
                target=a.output.joinpath(*p.parts);source=a.output.joinpath(*canonical.parts)
                assert sha(source)==r['sha256'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
                target.chmod(r.get('mode',0o644)&0o777)
    print(json.dumps({'verified_unique_files':len(seen),'verified_logical_files':len(records),'restored':bool(a.output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--directory',type=Path);p.add_argument('--archive',type=Path);p.add_argument('--output',type=Path);main(p.parse_args())
