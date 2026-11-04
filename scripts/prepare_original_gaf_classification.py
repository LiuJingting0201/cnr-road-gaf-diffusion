"""Original full-vehicle Hilbert/squeeze/RGB, participant labels; no guidance training."""
import argparse,hashlib,io,json,lzma,subprocess,time
from pathlib import Path
import numpy as np
from cnr.original_gaf import trajectory_sequence,fit_channels,encode_frozen
p=argparse.ArgumentParser();p.add_argument('--output',default='data/processed/original-hilbert224');p.add_argument('--eps',type=float,default=.0037);args=p.parse_args()
source=Path('data/canonical/most.npz.xz');out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
beg=time.monotonic();z=np.load(io.BytesIO(lzma.decompress(source.read_bytes())),allow_pickle=False);raw_meta=json.loads(str(z['metadata']))
xy=np.cumsum(z['delta_xy'],axis=0,dtype=np.int64)/raw_meta['coordinate_scale'];offsets=np.r_[0,np.cumsum(z['counts'])]
protocol=json.loads(Path('data/processed/most64/dataset.json').read_text());splits=protocol['split_by_id']
lo=np.full(2,np.inf);hi=np.full(2,-np.inf)
for j,ident in enumerate(z['ids']):
 if splits[str(ident)]=='train':
  x=xy[offsets[j]:offsets[j+1]];lo=np.minimum(lo,x.min(0));hi=np.maximum(hi,x.max(0))
arrays={s:{'sequences':[],'labels':[],'ids':[]} for s in ['train','val']};rejected=[];clipped={s:0 for s in arrays}
for j,ident in enumerate(z['ids']):
 split=splits[str(ident)]
 if split not in arrays:continue
 x=xy[offsets[j]:offsets[j+1]]
 try:h,n=trajectory_sequence(x,lo,hi,eps=args.eps)
 except ValueError as ex:rejected.append({'id':str(ident),'split':split,'reason':str(ex)});continue
 clipped[split]+=n;arrays[split]['sequences'].append(h);arrays[split]['labels'].append(z['labels'][j]);arrays[split]['ids'].append(str(ident))
 if j%250==0:print(json.dumps({'vehicle':j,'elapsed_s':time.monotonic()-beg}),flush=True)
clo,chi=fit_channels(np.array(arrays['train']['sequences']));summary={}
for split,a in arrays.items():
 seq=np.array(a['sequences']);labels=np.array(a['labels'],dtype=np.int64);images,n=encode_frozen(seq,clo,chi)
 np.savez_compressed(out/f'{split}.npz',images=images,labels=labels,ids=np.array(a['ids']),sequences=seq)
 summary[split]={'vehicles':len(seq),'class_counts':{name:int(np.sum(labels==c)) for c,name in enumerate(raw_meta['classes'])},'coordinate_clipped_points':clipped[split],'channel_clipped_values':n}
meta={'source':raw_meta,'source_package_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_commit':'4773ad65bce88d38b8b108ef5e4773fd99883c94','git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'summary':summary,'xy_min':lo.tolist(),'xy_max':hi.tolist(),'channel_min':clo.tolist(),'channel_max':chi.tolist(),'hilbert_eps':args.eps,'length':224,'mtf_bins':8,'scope':'Full vehicle trajectory, original squeeze and mirror duplication, RGB GASF/GADF/MTF; newly trained CNN, not original classifier reproduction. Test neither encoded nor evaluated.','adaptations':['Canonical time ordering verified by packer','local SUMO x/y used in original normalized coordinate slots, not claimed GPS','train-only coordinate and channel extrema frozen for deployment; clip out of bounds and count','keep previous vehicle split; no claim to original group classification accuracy'],'elapsed_s':time.monotonic()-beg}
(out/'dataset.json').write_text(json.dumps(meta,indent=2));(out/'rejections.json').write_text(json.dumps(rejected,indent=2));print(json.dumps(meta,indent=2),flush=True)
