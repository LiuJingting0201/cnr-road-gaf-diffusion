"""Validate matching requests and summarize the 24 five-class DiffTraj fits."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

MODES = ('bicycle', 'bus', 'motorcycle', 'passenger', 'taxi')
SEEDS = (42, 43)
PAPER = ('density_js', 'od_js', 'length_js', 'hotspot_f1')
MOTION = ('speed_w1_mps', 'slow_time_abs_error_pp', 'slow_entries_abs_error_per_10min')
FIELDS = (*PAPER, *MOTION)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def motion_values(record):
    err = record['signed_error']
    return dict(speed_w1_mps=record['speed_w1_mps'],
                slow_time_abs_error_pp=abs(err['slow_time_percent']),
                slow_entries_abs_error_per_10min=abs(err['slow_entries_per_10min']))


def equivalent(x, y):
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(equivalent(x[k], y[k]) for k in x)
    if isinstance(x, list):
        return isinstance(y, list) and len(x) == len(y) and all(equivalent(a, b) for a,b in zip(x,y))
    if isinstance(x, (int, float)):
        return math.isclose(x, y, abs_tol=1.e-10, rel_tol=1.e-12)
    return x == y


def aggregate(root):
    runs, reference, protocol, swaps, pooled = {}, {}, None, {}, {}
    rows = {}
    for seed in SEEDS:
        for arm in 'ABCD':
            for mode in (MODES if arm in 'AB' else ('all',)):
                job = f'{mode}-{arm}-{seed}'
                filename = root / 'runs' / job / 'eval' / 'metrics.json'
                d = json.loads(filename.read_text())
                check((d['arm'],d['seed'],d['mode'],d['epoch'],d['split'],d['phase']) ==
                      (arm,seed,mode,50,'validation','validation'), f'Unexpected fit {job}')
                eligible = MODES if mode == 'all' else (mode,)
                check(set(eligible)==set(d['by_class']) and d['requests']==600*len(eligible),
                      f'Incorrect validation population {job}')
                p = {k:d[k] for k in ('length_bins_m','low_m','high_m','pool_noise_sha256','eps_seed','sampling','metric_protocol','motion_protocol')}
                if protocol is None:
                    protocol = p
                check(equivalent(protocol,p),f'Grid / bins / noise / method differs for {job}')
                runs[job] = dict(file=str(filename.relative_to(root)), request_sha256=d['request_sha256'],
                                 noise_sha256=d['noise_sha256'], requests=d['requests'])
                for k in eligible:
                    b, m = d['by_class'][k], d['motion_by_class'][k]
                    check(b['requests']==m['requests']==600, f'Incorrect row count {job} {k}')
                    rec = {i:b[i] for i in ('request_sha256','noise_sha256','reference_median_length_m','reference_occupied_od')}
                    rec['motion_observed'] = m['observed']
                    rec['observed_500_vs_500'] = d['disjoint_observed_vs_observed'][k]
                    check(rec['observed_500_vs_500']['requests']==500, f'Invalid calibration sample {k}')
                    check(equivalent(reference.setdefault(k,rec),rec), f'Different requests, noise or observations {job} {k}')
                    if mode!='all':
                        check(all(b[i]==d[i] for i in ('request_sha256','noise_sha256')), f'Invalid single-class identity {job}')
                    values = dict(**{i:b[i] for i in PAPER}, **motion_values(m))
                    check(all(math.isfinite(v) for v in values.values()),f'Nonfinite score in {job}')
                    rows[(seed,arm,k)] = dict(**values, motion_observed=m['observed'],
                                              motion_generated=m['generated'], motion_signed_error=m['signed_error'])
                if mode=='all':
                    pooled[job] = dict(**{i:d[i] for i in PAPER}, **motion_values(d['motion']))
                    sw = d['label_swap']
                    check(sw['applicable'], f'Missing label-swap protocol {job}')
                    swaps[job] = sw
                else:
                    check(not d['label_swap']['applicable'], f'Unexpected swap in single-class model {job}')
    check(len(runs)==24 and len(rows)==40 and len(swaps)==4,'Incomplete experiment matrix')
    pooled_ids = {(runs[j]['request_sha256'],runs[j]['noise_sha256']) for j in pooled}
    check(len(pooled_ids)==1, 'Pooled model requests/noise differ')
    common = next(iter(swaps.values()))
    for k, sw in swaps.items():
        fields = ('pairs','observed_pair_ids_sha256','matching_rule','eligible_requests_per_side','covered_fraction')
        if common['pairs']:
            fields = (*fields,'observed_bus_minus_bicycle')
        check(all(equivalent(common[f], sw[f]) for f in fields),f'Unmatched swap population {k}')
    macro = {f'{arm}-{s}': {v: mean(rows[s,arm,m][v] for m in MODES) for v in FIELDS}
             for s in SEEDS for arm in 'ABCD'}
    effects=[]
    for s in SEEDS:
        for baseline, extended in [('A','B'),('C','D')]:
            for m in MODES:
                b, e = rows[s,baseline,m], rows[s,extended,m]
                effects.append(dict(seed=s,mode=m,comparison=f'{extended} minus {baseline}',
                                    difference={v:e[v]-b[v] for v in FIELDS}))
    return dict(fits=len(runs), validation_requests_per_mode=600,
                eligibility='A and B train separately per mode; C and D train on all five modes with labels.',
                reference_by_mode=reference, protocol=protocol, fit_records=runs, pooled_metrics=pooled,
                macro_mode_metrics=macro, per_mode={f'{m}-{arm}-{s}':v for (s,arm,m),v in rows.items()},
                within_training_population_effects=effects, bus_bicycle_label_swap=swaps)


def report(result):
    def header(first):
        return [f'| {first} | Seed | Density JS | OD JS | Length JS | Hotspot F1 | Speed W1 (m/s) | Slow time error (pp) | Slow entries error (/10 min) |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def row(label, seed, v):
        return f'| {label} | {seed} | ' + ' | '.join(f'{v[k]:.4f}' for k in FIELDS) + ' |'
    lines=['# Validation results for the five-class DiffTraj experiments','',
           'Each vehicle class contributes 600 validation requests. A and B fit separate models per class. C and D use one shared model with vehicle labels. B and D add speed supervision. Request and initial-noise fingerprints match for every comparison within each vehicle class. The two slow-motion errors are absolute differences of trip means.','',
           '## Equal-weight mean over the five classes','',*header('Arm')]
    for arm in 'ABCD':
        for s in SEEDS:
            lines.append(row(arm,s,result['macro_mode_metrics'][f'{arm}-{s}']))
    lines+=['','## Shared models evaluated on all 3,000 requests','',*header('Arm')]
    for arm in 'CD':
        for s in SEEDS:
            lines.append(row(arm,s,result['pooled_metrics'][f'all-{arm}-{s}']))
    lines+=['','## Results by vehicle class','',*header('Class and arm')]
    for m in MODES:
        for arm in 'ABCD':
            for s in SEEDS:
                lines.append(row(f'{m}, {arm}', s,result['per_mode'][f'{m}-{arm}-{s}']))
    lines+=['','## Matched bus–bicycle label contrasts','',
            'Each observed pair contains a bus and a bicycle with similar requested trip attributes. Each paired request is generated twice in the shared model with identical initial noise. Signed contrasts are bus minus bicycle.','',
            '| Arm and seed | Matched pairs | Observed slow-time contrast (pp) | Generated (pp) | Observed slow-entry contrast (/10 min) | Generated (/10 min) |',
            '|---|---:|---:|---:|---:|---:|']
    for j,d in result['bus_bicycle_label_swap'].items():
        if d['pairs']:
            o,g=d['observed_bus_minus_bicycle'], d['generated_bus_minus_bicycle']
            values=[o['slow_time_percent'],g['slow_time_percent'],o['slow_entries_per_10min'],g['slow_entries_per_10min']]
            lines.append(f"| {j} | {d['pairs']} | "+' | '.join(f'{v:.3f}' for v in values)+' |')
        else:
            lines.append(f'| {j} | 0 | — | — | — | — |')
    lines+=['','## Independent groups of observed trips','',
            'For each mode the 1,000 available validation trips are shuffled deterministically and split into two disjoint groups of 500. This describes empirical variation at that sample size. Evaluations above compare 600 observations against 600 samples generated for the same requests.','',
            '| Class | Density JS | OD JS | Length JS | Hotspot F1 |','|---|---:|---:|---:|---:|']
    for m in MODES:
        d=result['reference_by_mode'][m]['observed_500_vs_500']
        lines.append(f'| {m} | '+' | '.join(f'{d[k]:.4f}' for k in PAPER)+' |')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    root=parser.parse_args().root
    result=aggregate(root)
    for name,body in [('validation-tables.md',report(result)),
                      ('validation-summary.json',json.dumps(result,indent=2,allow_nan=False)+'\n')]:
        path=root/name
        temp=path.with_suffix(path.suffix+'.tmp')
        temp.write_text(body)
        temp.replace(path)
    print('Validated 24 fits, 40 per-class scores and four paired label-swap records.',flush=True)


if __name__=='__main__':
    main()
