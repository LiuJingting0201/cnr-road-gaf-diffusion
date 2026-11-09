"""Fixed-budget continuation: GAF encoders trained through trajectory denoising."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from g33_train import digest, save
from g33_model import pack_motion
from g39_train import load_train, split_batch
from g39_targets import derangement
from g39_fast_inputs import MotionInputCache
from g40_model import ARMS, ConditionedMotion, CodeEncoders, views, predict
from g40_data import source_states, load_prefix, class_centers


def prepare_model(args, rows, device):
    root = args.base_models/f'B-{args.seed}'; receipt = json.loads((root/'complete.json').read_text())
    assert receipt['complete'] and digest(root/'model.pt') == receipt['model_sha256']
    old = torch.load(root/'model.pt', map_location='cpu', weights_only=False)
    assert old['alignment_arm'] == 'B' and old['seed'] == args.seed and old['width'] == 64
    model = ConditionedMotion(base=64); model.load_base(old['state_dict'])
    kind = 'raw' if args.arm == 'J-RAW' else 'gaf'
    states, source_hashes = source_states(args.teachers, args.raw_teachers, kind)
    encoders = CodeEncoders(args.arm, states)
    inputs = {}; prefixes = {}
    if args.arm not in ('B','C'):
        for v in views(args.arm):
            inputs[v], prefixes[v] = load_prefix(args.prefixes/(kind+'-'+v), rows, device)
            assert prefixes[v]['source_model_sha256'] == source_hashes[v]
    centers = class_centers(args.teachers, rows, device) if args.arm == 'C' else None
    return model.to(device), encoders.to(device), inputs, centers, old, source_hashes, prefixes


def train(args):
    out = args.output; out.mkdir(parents=True, exist_ok=False); start = time.monotonic()
    try:
        torch.set_num_threads(2); torch.manual_seed(args.seed); np.random.seed(args.seed)
        torch.backends.cudnn.benchmark = False; device = torch.device('cuda')
        rows, clocks, targets, stats = load_train(args.data, 'native_0_25s')
        model, encoders, inputs, centers, old, teacher_hashes, prefix_receipts = prepare_model(args, rows, device)
        assert stats == old['stats']
        initial_hash = hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in model.state_dict().values())).hexdigest()
        cache = MotionInputCache(rows, clocks, targets, stats, device) if args.input_pipeline == 'cached' else None
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        encoder_ema = copy.deepcopy(encoders).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW([
            {'params': model.parameters(), 'lr': 2e-4},
            {'params': [p for p in encoders.parameters() if p.requires_grad], 'lr': 5e-5}], weight_decay=1e-4)
        permutation = np.arange(len(rows['ids']))
        if args.arm == 'J-SH': permutation, _ = derangement(rows['labels'], rows['length'], rows['duration'])
        assert np.array_equal(rows['labels'][permutation], rows['labels'])
        schedule = torch.cumprod(1-torch.linspace(.0001,.02,1000,device=device), 0)
        config = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        config.update(training_rows=len(rows['ids']), training_parents=len(set(rows['parent_ids'])),
            class_counts=np.bincount(rows['labels'],minlength=5).tolist(), initial_motion_sha256=initial_hash,
            data_sha256=digest(args.data), base_model_sha256=digest(args.base_models/f'B-{args.seed}'/'model.pt'),
            teacher_model_sha256=teacher_hashes, prefix_receipt_sha256={v:digest(args.prefixes/(encoders.kind+'-'+v)/'complete.json') for v in inputs},
            classification_weight=.1, denoiser_lr=.0002, encoder_lr=.00005, ema=.001,
            bottleneck_dimensions=[32,32], batchnorm_running_stats_frozen=True,
            trained_encoder_layers='last residual stage, classifier head and bottleneck for joint arms',
            no_validation_for_training_or_selection=True, fixed_final_checkpoint=True,
            prior_not_fitted_yet=True, source_sha256={n:digest(Path(__file__).with_name(n)) for n in ('g40_train.py','g40_model.py','g40_data.py')},
            parameters={'motion':sum(p.numel() for p in model.parameters()),
                        'encoder_trainable':sum(p.numel() for p in encoders.parameters() if p.requires_grad)},
            shuffle_same_parent_pairs=int(np.sum(rows['parent_ids']==rows['parent_ids'][permutation])) if args.arm=='J-SH' else None)
        save(out/'config.json', config); save(out/'stats.json',stats)
        rng = np.random.default_rng(args.seed+40000)
        groups = [np.flatnonzero(rows['labels']==c) for c in range(5)]; lengths = np.array([len(t) for t in targets])
        token_count = 0; model.train(); encoders.train()
        with (out/'training.jsonl').open('w',buffering=1) as log:
            for step in range(1,args.steps+1):
                if time.monotonic()-start > 4*3600: raise TimeoutError('Four-hour fit limit')
                labels = rng.integers(0,5,64); indices = np.array([rng.choice(groups[c]) for c in labels])
                chunks = split_batch(indices,targets,32768,32); optimizer.zero_grad(set_to_none=True); metrics=[]
                for chunk in chunks:
                    ix = np.array(chunk); style_ix = torch.as_tensor(permutation[ix],device=device)
                    batch = cache.pack(ix) if cache is not None else pack_motion([targets[i] for i in ix],[clocks[i] for i in ix],
                        rows['labels'][ix],rows['paths'][ix],rows['length'][ix],rows['duration'][ix],stats,device)
                    code, logits = encoders({v:x[style_ix] for v,x in inputs.items()}, batch['labels'], centers)
                    t = torch.randint(1000,(len(ix),),device=device); noise = torch.randn_like(batch['x'])*batch['mask']
                    alpha = schedule[t,None,None]
                    noisy = (alpha.sqrt()*batch['x']+(1-alpha).sqrt()*noise)*batch['mask']
                    prediction = predict(model,batch,noisy,t,None if args.arm=='B' else code)
                    denoising = ((prediction-noise).square()*batch['time_weights']).sum()/64
                    cls = denoising.new_tensor(0.)
                    if args.arm.startswith('J-'):
                        cls = torch.stack([F.cross_entropy(y,batch['labels'],reduction='sum') for y in logits.values()]).mean()/64
                    loss = denoising+.1*cls; loss.backward()
                    metrics.append(torch.stack([loss.detach(),denoising.detach(),cls.detach()]))
                    token_count += int(lengths[ix].sum())
                parameters = [*model.parameters(),*[p for p in encoders.parameters() if p.requires_grad]]
                norm = torch.nn.utils.clip_grad_norm_(parameters,1.)
                values = torch.cat([torch.stack(metrics).sum(0,dtype=torch.float64),norm.detach().double().reshape(1)]).cpu().numpy()
                if not np.isfinite(values).all(): raise ValueError('Nonfinite loss/gradient')
                optimizer.step()
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),.001)
                    torch._foreach_lerp_(list(encoder_ema.parameters()),list(encoders.parameters()),.001)
                if step % 1000 == 0 or step == args.steps:
                    torch.save({'step':step,'motion':model.state_dict(),'encoder':encoders.state_dict(),
                        'ema_motion':ema.state_dict(),'ema_encoder':encoder_ema.state_dict(),'optimizer':optimizer.state_dict(),
                        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                        'numpy_rng':rng.bit_generator.state,'valid_tokens_seen':token_count},out/'checkpoint-latest.tmp')
                    (out/'checkpoint-latest.tmp').replace(out/'checkpoint-latest.pt')
                if step==1 or step%100==0:
                    record={'step':step,'loss':float(values[0]),'denoising_loss':float(values[1]),'classification_loss':float(values[2]),
                        'gradient_norm':float(values[3]),'elapsed_s':time.monotonic()-start,'microbatches':len(chunks),
                        'valid_tokens_seen':token_count,'gpu_max_allocated_bytes':torch.cuda.max_memory_allocated()}
                    log.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
        model_path=out/'model.pt'
        torch.save({'motion':ema.state_dict(),'encoder':encoder_ema.state_dict(),'arm':args.arm,'seed':args.seed,
                    'width':64,'stats':stats,'classes':old['classes'],'steps':args.steps},model_path)
        # Only train-side codes; SH preserves the wrong-trip conditioning used during fitting.
        codes=[]; logits_saved={v:[] for v in views(args.arm)}
        with torch.no_grad():
            for a in range(0,len(rows['ids']),64):
                ix=np.arange(a,min(a+64,len(rows['ids']))); pix=torch.as_tensor(permutation[ix],device=device)
                code, logits=encoder_ema({v:x[pix] for v,x in inputs.items()},torch.as_tensor(rows['labels'][ix],device=device),centers)
                codes.append(code.cpu().numpy())
                for v,y in logits.items():logits_saved[v].append(y.cpu().numpy())
        values=np.concatenate(codes);assert values.shape==(len(rows['ids']),64) and np.isfinite(values).all()
        np.savez_compressed(out/'train-codes.npz',**{k:rows[k] for k in ('ids','parent_ids','labels','paths','length','duration')},
            code=values,paired_input_id=rows['ids'][permutation],**{v+'_logits':np.concatenate(z) for v,z in logits_saved.items()})
        save(out/'complete.json',{'complete':True,'arm':args.arm,'seed':args.seed,'steps':args.steps,
            'elapsed_s':time.monotonic()-start,'training_rows':len(rows['ids']),'valid_tokens_seen':token_count,
            'model_sha256':digest(model_path),'config_sha256':digest(out/'config.json'),'stats_sha256':digest(out/'stats.json'),
            'codes_sha256':digest(out/'train-codes.npz'),'prior_fitted':False,'generation_evaluated':False,
            'artifacts':{name:digest(out/name) for name in ('model.pt','config.json','stats.json','train-codes.npz')},
            'final_cuda_rng_sha256':hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
            'final_batch_rng_sha256':hashlib.sha256(json.dumps(rng.bit_generator.state,sort_keys=True).encode()).hexdigest(),
            'classes':old['classes'],'code_generation_source':'training rows only; SH uses the fixed paired wrong-trip image',
            'code_std_quantiles':np.quantile(values.std(0),[0,.1,.5,.9,1]).tolist()})
    except BaseException as error:
        save(out/'failed.json',{'error':repr(error),'seconds':time.monotonic()-start});raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('data','base-models','teachers','raw-teachers','prefixes','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=(42,43),required=True)
    p.add_argument('--steps',type=int,default=5000);p.add_argument('--input-pipeline',choices=('original','cached'),default='original')
    train(p.parse_args())
