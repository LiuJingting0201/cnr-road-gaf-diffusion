"""Download specified private Release assets and verify GitHub SHA256 receipts."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import tempfile
import urllib.parse
from archive_release import API,REPO,sha


def download(api,asset,directory):
    assert Path(asset['name']).name==asset['name'] and asset['state']=='uploaded'
    directory.mkdir(parents=True,exist_ok=True);target=directory/asset['name']
    if target.exists():
        assert target.stat().st_size==asset['size'] and 'sha256:'+sha(target)==asset['digest']
        return {'name':target.name,'size':asset['size'],'digest':asset['digest'],'already_verified':True}
    assert shutil.disk_usage(directory).free>asset['size']+1024**3
    c=http.client.HTTPSConnection('api.github.com',timeout=300)
    c.request('GET',f'/repos/{REPO}/releases/assets/{asset["id"]}',headers={'Authorization':'Bearer '+api.secret,'Accept':'application/octet-stream','X-GitHub-Api-Version':'2022-11-28','User-Agent':'cnr-archive-restore'})
    r=c.getresponse()
    if r.status in (301,302,303,307,308):
        u=urllib.parse.urlsplit(r.getheader('Location'));r.read();c.close()
        assert u.scheme=='https' and u.hostname and u.hostname.endswith('.githubusercontent.com')
        c=http.client.HTTPSConnection(u.hostname,timeout=300)
        # The signed GitHub asset URL needs no credential. Never forward auth.
        c.request('GET',urllib.parse.urlunsplit(('','',u.path,u.query,'')),headers={'User-Agent':'cnr-archive-restore'});r=c.getresponse()
    assert r.status==200,f'Asset download HTTP {r.status}'
    h=hashlib.sha256();size=0
    with tempfile.NamedTemporaryFile(prefix=target.name+'.',suffix='.partial',dir=directory,delete=False) as f:
        partial=Path(f.name)
        while True:
            b=r.read(4*1024**2)
            if not b:break
            f.write(b);h.update(b);size+=len(b)
    c.close();assert size==asset['size'] and 'sha256:'+h.hexdigest()==asset['digest'],'Preserved partial for inspection'
    partial.rename(target)
    return {'name':target.name,'size':size,'digest':asset['digest'],'download_verified':True,'credential_forwarded_to_redirect':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--asset',action='append',required=True);p.add_argument('--directory',required=True,type=Path);a=p.parse_args()
    api=API();api.private();release=api.release();assets={x['name']:x for x in api.assets(release['id'])}
    for name in a.asset:print(json.dumps(download(api,assets[name],a.directory)),flush=True)


if __name__=='__main__':main()
