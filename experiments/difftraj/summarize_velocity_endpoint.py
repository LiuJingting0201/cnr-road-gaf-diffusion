"""Verify the matched endpoint-loss experiment and write its validation tables."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def summarize(base, new):
    seeds=(42,43)
    fits = {}
    for s in seeds:
        old_cfg = read(base/f'velocity-{s}-config.json')
        new_cfg = read(new/f'config-{s}.json')
        assert new_cfg.pop('integrated_endpoint_weight') == .1 and new_cfg == old_cfg
        old, model = (read(folder/f'velocity-{s}.json') for folder in (base,new))
        for k in ('noise_sha256','head_sha256','request_sha256','source_manifest_sha256',
                  'prepared_manifest_sha256','requests','unique_draws','class_confusion','classes','updates','epoch','seed'):
            assert old[k] == model[k], k
        assert old['updates']==9760 and old['requests']==3000
        fits[s] = old,model
    lines=['# Integrated endpoint loss on five-class SUMO trips', '',
           'Validation uses 3,000 common requests, 600 per class, one fixed draw per request with the supplied label. '
           'Across-class metrics weight all classes equally. Columns give the original velocity generator and the '
           'same generator trained with the extra integrated endpoint loss. The weight is 0.1 for both seeds. '
           '[Loss definition and comparison protocol](PROTOCOL.md).','',
           '| Measure | Original 42 | Endpoint 42 | Original 43 | Endpoint 43 |',
           '|---|---:|---:|---:|---:|']
    def table_row(title,values,fmt):
        lines.append('| '+' | '.join([title]+[format(x,fmt) for x in values])+' |')
    for title,keys,fmt in [
            ('Requested destination hit',('endpoints','end_cell_hit'),'.1%'),
            ('Both requested cells hit',('endpoints','joint_cell_hit'),'.1%'),
            ('Speed W1 (m/s)',('speed_w1_mps',),'.3f'),
            ('Acceleration W1 (m/s²)',('accel_w1_mps2',),'.3f'),
            ('Density JS',('paper','density_js'),'.4f'),
            ('OD JS',('paper','od_js'),'.4f'),
            ('Length JS',('paper','length_js'),'.4f'),
            ('Hotspot F1',('paper','hotspot_f1'),'.3f')]:
        values=[]
        for s in seeds:
            for fit in fits[s]:
                val=fit['macro']['provided']
                for key in keys: val=val[key]
                values.append(val)
        table_row(title,values,fmt)
    geo=[read(base/'direction-debug.json'),read(new/'geometry.json')]
    for label,key,fmt in [('Median paired end error (m)','median_end_error_m','.1f'),
                           ('Median net bearing error (degrees)','median_net_angle_deg','.2f'),
                           ('Median absolute net displacement length error (m)','median_abs_net_length_error_m','.1f')]:
        table_row(label,[g[f'velocity-{s}']['all'][key] for s in seeds for g in geo],fmt)
    lines+=['','## Holding request and noise fixed, switch bicycle to bus','',
            'Observed reference compares 53 matched SUMO trip pairs whose requested mean speeds differ by at most 0.5 m/s. '
            'The generated change switches the label at each identical header and noise. Entries count intervals crossing '
            'below 1 m/s and remaining below for at least two intervals. Time share is in percentage points.','',
            '| Bus minus bicycle | SUMO | Original 42 | Endpoint 42 | Original 43 | Endpoint 43 |',
            '|---|---:|---:|---:|---:|---:|']
    pair=[read(p/'balance.json')['pairs']['bus_minus_bicycle'] for p in (base,new)]
    assert pair[0]['matched_ids_sha256']==pair[1]['matched_ids_sha256'] and pair[1]['after']['pairs']==53
    for feat,label in [('slow_time_percent','Low-speed share (pp)'),('slow_entries_per_10min','Low-speed entries / 10 min'),
                       ('p90_speed_mps','90th-percentile speed (m/s)'),('mean_abs_acceleration_mps2','Mean absolute acceleration (m/s²)')]:
        items=[p['features'][feat] for p in pair]
        assert items[0]['observed']==items[1]['observed']
        table_row(label,[items[0]['observed']['mean']]+[v['generated'][f'velocity-{s}']['mean'] for s in seeds for v in items],'.3f')
    lines+=['','## By vehicle class','',
            '| Class | Seed | Destination hit, original → endpoint | Median end error (m) | Speed W1 (m/s) |',
            '|---|---:|---:|---:|---:|']
    for cls in fits[42][0]['classes']:
        for s in seeds:
            arms=[f['results']['provided'][cls] for f in fits[s]]
            def comparison(get,fmt):
                return ' → '.join(format(get(v),fmt) for v in arms)
            lines.append('| '+' | '.join([cls,str(s),comparison(lambda x:x['endpoints']['end_cell_hit'],'.1%'),
                                       comparison(lambda x:x['endpoints']['end_error_m']['median'],'.0f'),
                                       comparison(lambda x:x['speed_w1_mps'],'.3f')])+' |')
    lines+=['','The same first validation request per class is plotted at both training seeds. The dotted box is the requested destination cell. '
            'Colours give speed on a shared scale within each request. Distance annotations compare the generated endpoint to the paired observation.','',
            '![Same-request maps, seed 42](map-42.png)','','![Same-request maps, seed 43](map-43.png)','']
    (new/'RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path)
    p.add_argument('result',type=Path)
    args=p.parse_args()
    summarize(args.baseline,args.result)
