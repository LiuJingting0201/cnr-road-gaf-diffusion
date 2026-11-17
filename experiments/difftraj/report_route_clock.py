"""Check fixed-route receipts and write the release comparison using final positions."""
import argparse
import json
from pathlib import Path


def get(x,keys):
    for k in keys.split('/'):
        x=x[k]
    return x


def compare(folder, previous):
    load=lambda p:json.loads(p.read_text())
    a,b=(load(folder/f'metrics-{s}.json') for s in (42,43))
    common=load(folder/'common-routes-42/metrics-43.json')
    prior=load(previous)
    for result in [a,b,common]:
        assert result['requests_sha256']==a['requests_sha256'] and result['noise_sha256']==a['noise_sha256']
        assert result['pairs']['matched_ids_sha256']==a['pairs']['matched_ids_sha256']
        assert result['n']==3000 and result['pairs']['after']['pairs']==53
        assert result['arms']['baseline']['macro']['endpoints']==result['arms']['conditioned']['macro']['endpoints']
    assert a['route_sha256']==common['route_sha256'] and a['route_sha256']!=b['route_sha256']
    for result,old in zip([a,b],prior):
        assert result['seed']==old['seed'] and result['pairs']['matched_ids_sha256']==old['matched_ids']
        for metric in ['speed_w1_mps','accel_w1_mps2']:
            assert abs(result['arms']['baseline']['macro'][metric]-old['macro'][metric])<1e-6
    lines=['# Can route-conditioned speed diffusion improve the final trip?', '',
           'Two seeded velocity fits received the length and distance-ordered turn angles of routes sampled by frozen coordinate models. '
           'The tables compare timing on the same sampled route, with the same request and noise. All velocity and acceleration measurements use the final positions after retiming. '
           'Each cell gives the original → route-conditioned clock. The third column reuses the exact set of coordinate routes in the first column.','',
           '| Measure | Seed 42, routes 42 | Seed 43, routes 43 | Seed 43, routes 42 |','|---|---:|---:|---:|']
    rows=[('Speed W1 (m/s)','macro/speed_w1_mps','.4f'),
          ('Acceleration W1 (m/s²)','macro/accel_w1_mps2','.4f'),
          ('Requested destination hit','macro/endpoints/end_cell_hit','.1%'),
          ('Both requested cells hit','macro/endpoints/joint_cell_hit','.1%'),
          ('Mean within-trip speed W1 (m/s)','mean_paired_trip_speed_w1_mps','.4f'),
          ('Class-average absolute mean-speed bias (m/s)','macro_speed_w1_mean_bias_lower_bound_mps','.4f'),
          ('Median absolute speed-clock rescaling','median_absolute_length_rescaling','.1%'),
          ('Density JS','macro/paper/density_js','.4f'),('OD JS','macro/paper/od_js','.4f'),
          ('Length JS','macro/paper/length_js','.4f'),('Hotspot F1','macro/paper/hotspot_f1','.3f')]
    def format_row(label,key,fmt):
        return '| '+' | '.join([label]+[' → '.join(format(get(d['arms'][m],key),fmt) for m in ('baseline','conditioned')) for d in (a,b,common)])+' |'
    for label,key,fmt in rows:
        lines.append(format_row(label,key,fmt))
    lines+=['','## Speed error by vehicle', '',
            '| W1 (m/s) | Seed 42, routes 42 | Seed 43, routes 43 | Seed 43, routes 42 |','|---|---:|---:|---:|']
    for cls in a['arms']['baseline']['per_class']:
        lines.append(format_row(cls,'per_class/'+cls+'/speed_w1_mps','.3f'))
    lines+=['','## Change only the vehicle label, keeping each route fixed','',
            'The simulated reference uses 53 similar bus–bicycle trip pairs with requested mean-speed gaps at most 0.5 m/s. Both generated labels are applied to both requests, each at its own unchanged route and noise. '
            'Earlier exploratory label swaps permitted a different route when changing the label; the original clock was re-scored here under the fixed-route rule.','',
            '| Bus minus bicycle (matched SUMO difference) | Seed 42, routes 42 | Seed 43, routes 43 | Seed 43, routes 42 |','|---|---:|---:|---:|']
    for k,label in [('slow_time_percent','Low-speed time (pp)'),('slow_entries_per_10min','Low-speed entries / 10 min'),('p90_speed_mps','90th-percentile speed (m/s)')]:
        lines.append(format_row(f"{label}; SUMO {a['observed_gap'][k]:+.2f}",'label_effect/'+k,'+.2f'))
    lines+=['','[Training and matching protocol](PROTOCOL.md)','','![Paths and speed curves for the first request of each vehicle class, seed 42](map-42.png)','',
            '![Same comparison for seed 43](map-43.png)','','[Seed 43 on seed-42 routes](common-routes-42/map-43.png)','']
    (folder/'RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('result',type=Path)
    p.add_argument('previous',type=Path)
    a=p.parse_args()
    compare(a.result,a.previous)
