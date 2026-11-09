"""Upload verified private release assets using the normal Git credential helper.

No token is written to disk, printed or passed on a command line. An authenticated
cloud uploader may receive it once on a protected stdin pipe. Release uploads
are kept outside ordinary Git history.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse

REPO='LiuJingting0201/cnr-road-gaf-artifacts'
TAG='archive-2026-11-09'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def token():
    if os.environ.get('CNR_ARCHIVE_TOKEN_STDIN')=='1':
        value=sys.stdin.readline().strip()
        if not value:raise RuntimeError('Missing in-memory upload authentication')
        return value
    r=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
        text=True,capture_output=True,check=True,env=dict(os.environ,GIT_TERMINAL_PROMPT='0'))
    fields=dict(v.split('=',1) for v in r.stdout.splitlines() if '=' in v)
    return fields['password']


class API:
    def __init__(self,secret=None):self.secret=secret if secret is not None else token()
    def request(self,method,path,data=None,host='api.github.com',file=None):
        assert host in {'api.github.com','uploads.github.com'}
        conn=http.client.HTTPSConnection(host,timeout=1200)
        headers={'Authorization':'Bearer '+self.secret,'Accept':'application/vnd.github+json',
                 'X-GitHub-Api-Version':'2022-11-28','User-Agent':'cnr-reproducibility-archive'}
        body=json.dumps(data).encode() if data is not None else None
        if file:
            headers.update({'Content-Type':'application/octet-stream','Content-Length':str(file.stat().st_size)})
            conn.putrequest(method,path)
            for k,v in headers.items():conn.putheader(k,v)
            conn.endheaders()
            with file.open('rb') as f:
                for b in iter(lambda:f.read(1024**2),b''):conn.send(b)
        else:
            headers['Content-Type']='application/json';conn.request(method,path,body=body,headers=headers)
        response=conn.getresponse();status=response.status;raw=response.read();conn.close()
        if not 200<=status<300:
            # Do not log request headers or credential-helper output.
            raise RuntimeError(f'GitHub HTTP {status} for {method} {path.split("?")[0]}: '+raw.decode(errors='replace')[:250])
        return json.loads(raw) if raw else None
    def private(self):
        d=self.request('GET','/repos/'+REPO)
        assert d['private'] is True and d['full_name'].lower()==REPO.lower(),'Private destination required'
    def release(self):
        self.private()
        pages=self.request('GET','/repos/'+REPO+'/releases?per_page=100')
        found=[r for r in pages if r['tag_name']==TAG]
        if found:return found[0]
        return self.request('POST','/repos/'+REPO+'/releases',{'tag_name':TAG,'name':'CNR reproducibility archive — 2026-11-09',
            'draft':True,'body':'Private archive in preparation. Assets are individually SHA256 verified. Read MANIFEST and RESTORE before extraction.'})
    def assets(self,rid):
        result=[];page=1
        while True:
            rows=self.request('GET',f'/repos/{REPO}/releases/{rid}/assets?per_page=100&page={page}');result+=rows
            if len(rows)<100:return result
            page+=1
    def upload(self,release,path):
        path=Path(path);assert path.is_file() and path.stat().st_size<2*1024**3
        checksum=sha(path);expected='sha256:'+checksum
        for attempt in range(3):
            existing=[a for a in self.assets(release['id']) if a['name']==path.name]
            if existing:
                a=existing[0]
                if a['state']=='uploaded':
                    assert a.get('digest')==expected and a['size']==path.stat().st_size,'Asset collision/checksum differs'
                    return a
                # Only a failed, incomplete upload of this exact current name.
                assert a['state']=='starter'
                self.request('DELETE',f'/repos/{REPO}/releases/assets/{a["id"]}')
            try:
                a=self.request('POST',f'/repos/{REPO}/releases/{release["id"]}/assets?name='+urllib.parse.quote(path.name),host='uploads.github.com',file=path)
                assert a['state']=='uploaded' and a['size']==path.stat().st_size and a.get('digest')==expected
                return a
            except (OSError,http.client.HTTPException,RuntimeError):
                if attempt==2:raise
                time.sleep(5*(attempt+1))


def main(a):
    api=API();release=api.release()
    if a.action=='init':print(json.dumps({'id':release['id'],'url':release['html_url'],'draft':release['draft']}));return
    if a.action=='upload':
        a.receipts.mkdir(parents=True,exist_ok=True)
        for path in a.files:
            start=time.monotonic();asset=api.upload(release,path)
            receipt={k:asset[k] for k in ['id','name','size','digest','browser_download_url','state']}
            receipt.update(seconds=time.monotonic()-start,private_repository=REPO)
            (a.receipts/(path.name+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
            print(json.dumps(receipt),flush=True)
    elif a.action=='publish':
        manifest=json.loads(a.manifest.read_text());assets={r['name']:r for r in api.assets(release['id'])}
        for item in manifest['assets']:
            assert assets[item['name']]['digest']==item['digest'] and assets[item['name']]['size']==item['size']
        result=api.request('PATCH',f'/repos/{REPO}/releases/{release["id"]}',{'draft':False,'body':a.body.read_text()})
        print(json.dumps({'url':result['html_url'],'draft':result['draft'],'verified_assets':len(manifest['assets'])}))
    else:
        print(json.dumps({'release':release['html_url'],'draft':release['draft'],'assets':api.assets(release['id'])},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['init','upload','publish','list']);p.add_argument('files',nargs='*',type=Path)
    p.add_argument('--receipts',type=Path,default=Path('upload-receipts'));p.add_argument('--manifest',type=Path);p.add_argument('--body',type=Path)
    main(p.parse_args())
