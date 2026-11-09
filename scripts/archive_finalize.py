"""Verify all planned private assets and publish the completed archive Release."""
import argparse
import base64
import json
from pathlib import Path
from archive_release import API,REPO,TAG,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--staging',type=Path,required=True);a=p.parse_args();root=a.staging
    remote=json.loads((root/'REMOTE_CLOSURE.json').read_text());local=json.loads((root/'local-upload-complete.json').read_text())
    recovery=json.loads((root/'CLOUD_RESTORE_CHECKS.json').read_text());assert local['complete'] and recovery['restored']
    expected=[]
    for name,value in remote.items():
        if name=='coverage':continue
        assert value['complete'];expected.extend(value['receipts'])
    expected.extend(local['assets'])
    for file in (root/'final-upload-receipts').glob('*.json'):expected.append(json.loads(file.read_text()))
    source=Path(__file__).resolve().parent.parent
    for name in ('Road_Conditioned_Diffusion_Thesis_v19.pdf','PoliTo_Thesis_v19_Overleaf.zip'):
        file=source/'deliverables/thesis-v19'/name;expected.append({'name':name,'size':file.stat().st_size,'digest':'sha256:'+sha(file)})
    model=json.loads((root/'download-check.json').read_text());expected.append({'name':model['asset_name'],'size':model['bytes'],'digest':'sha256:'+model['sha256']})
    api=API();api.private();release=api.release()
    for name in ('RESTORE_COMPLETE.md','CLOUD_RESTORE_CHECKS.json'):
        asset=api.upload(release,root/name);expected.append({k:asset[k] for k in ('name','id','size','digest','state')})
    actual={x['name']:x for x in api.assets(release['id'])};wanted={}
    for row in expected:
        if row['name'] in wanted:assert wanted[row['name']]['digest']==row['digest']
        wanted[row['name']]=row
    assert set(actual)-{'ARCHIVE_INDEX.json'}==set(wanted),'Missing or unexpected Release assets'
    for name,r in wanted.items():
        x=actual[name];assert x['state']=='uploaded' and (x['size'],x['digest'])==(r['size'],r['digest']),name
    rows=[{k:actual[name][k] for k in ('id','name','size','digest','state')} for name in sorted(wanted)]
    lm=json.loads((root/'LOCAL_MANIFEST.json').read_text())
    coverage={**remote['coverage'],'local_file_paths':lm['summary']['files'],'local_logical_bytes':lm['summary']['logical_bytes']}
    coverage['total_file_paths']=coverage['cloud_file_paths']+coverage['local_file_paths'];coverage['total_logical_bytes']=coverage['cloud_logical_bytes']+coverage['local_logical_bytes']
    index={'schema':'cnr-private-release-index-v1','archive_date':'2026-11-09','complete':True,'repository':REPO,'release_tag':TAG,'coverage':coverage,
        'assets':rows,'verified_asset_bytes_excluding_index':sum(x['size'] for x in rows),'read_first':'RESTORE_COMPLETE.md',
        'limits':['Not a bootable machine image; Python/SUMO environments require installation.','Credentials and incomplete local transfer fragments excluded.','Original source files were not deleted.','No scientific target decoding or new evaluation performed by archive tooling.'],
        'history':['full-source-history.bundle: all30refs at initial local0eff678snapshot','legacy-server-history.bundle: independently verified old-server Git history; uncommitted files in legacy source snapshot'],
        'recovery_checks':['Actual local1455file restore and full bundle mirror clone/fsck','Actual GitHub model download with SHA256 match','Actual cloud current-models download and full restore; see CLOUD_RESTORE_CHECKS.json']}
    index_path=root/'ARCHIVE_INDEX.json';index_path.write_text(json.dumps(index,indent=2)+'\n');index_asset=api.upload(release,index_path)
    body=f'''Comprehensive private reproducibility archive, verified2026-11-09.

Start with **RESTORE_COMPLETE.md** (supersedes the earlier RESTORE.md), then ARCHIVE_INDEX.json. Includes {coverage['total_file_paths']:,} archived file paths, {coverage['total_logical_bytes']/10**9:.2f}GB logical source content, full local/server Git history, uncommitted legacy code, all model/data/experiment snapshots, GAF caches and closed G38 results. Original data and weights remain private.

All {len(rows)+1} assets were checked against exact sizes and SHA256 digests. Full assets total {(sum(x['size'] for x in rows)+index_asset['size'])/2**30:.2f}GiB including index. Model archives and local deliverables were actually restored; see recovery receipts. Installed environments and credentials are excluded. Source files were preserved.

Index SHA256: `{index_asset['digest']}`.

The public [source repository](https://github.com/LiuJingting0201/cnr-road-gaf-diffusion) contains code, full aggregate results, current thesis PDF and Overleaf package. Archive completion is not a claim that GAF universally improves generation.
'''
    result=api.request('PATCH',f'/repos/{REPO}/releases/{release["id"]}',{'draft':False,'body':body})
    assert result['draft'] is False
    readme=f'''# CNR private reproducibility archive

[Open the completed2026-11-09 archive]({result['html_url']}). All {len(rows)+1} Release attachments are size/SHA256 verified, covering {coverage['total_file_paths']:,} file paths and {coverage['total_logical_bytes']/10**9:.2f}GB logical project content. Downloads total {(sum(x['size'] for x in rows)+index_asset['size'])/2**30:.2f}GiB.

Read **RESTORE_COMPLETE.md**, then **ARCHIVE_INDEX.json**. The earlier RESTORE.md remains as a historical preparation note; the final instructions include GAF cache, G38 and legacy-source supplements.

Contents include original user CSVs, SUMO recordings, prepared data, all model checkpoints, generated candidates, full metrics/configurations/logs/failures, GAF image caches, thesis drafts/Overleaf packages, complete local and server Git bundles, and old-server uncommitted source modifications. Closed G38 generation/evaluation is included. Environments/installers, credentials and partial transfer fragments are excluded; this is not a bootable server image.

The current thesis and source/results remain in the [public code repository](https://github.com/LiuJingting0201/cnr-road-gaf-diffusion). Original data and weights are private here. Original source files have not been deleted.

Final index SHA256: `{index_asset['digest']}`.
'''
    old=api.request('GET','/repos/'+REPO+'/contents/README.md')
    commit=api.request('PUT','/repos/'+REPO+'/contents/README.md',{'message':'Publish verified complete archive and recovery index','sha':old['sha'],'content':base64.b64encode(readme.encode()).decode()})
    proof={'complete':True,'private_repository':REPO,'release_url':result['html_url'],'release_id':release['id'],'draft':False,'verified_assets':len(rows)+1,
        'uploaded_bytes':sum(x['size'] for x in rows)+index_asset['size'],'index_sha256':index_asset['digest'],'coverage':coverage,'private_readme_commit':commit['commit']['sha']}
    (root/'PUBLISHED.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
