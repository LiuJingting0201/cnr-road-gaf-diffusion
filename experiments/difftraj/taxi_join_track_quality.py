"""Join the endpoint and road checks by cohort, draw order and source row."""
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'docs/results'
destination = ROOT / 'taxi-porto-track-quality'
road = ROOT / 'taxi-road-quality'


def collect(city, source):
    rows = {}
    for row in csv.DictReader(source):
        if city == 'SUMO':
            arm, index = row['arm'], row['row_index']
            whole, linked, central, central_linked = ('intervals', 'linked', 'central_intervals', 'central_linked')
        else:
            arm, index = row['cohort'], row['dataset_row']
            whole, linked, central, central_linked = ('intervals', 'linked_intervals', 'interior_intervals', 'interior_linked')
        key = (int(row['request_order']), int(index), 'observed' if arm in ('real','observed') else arm)
        assert key not in rows
        rows[key] = dict(intervals=int(row[whole]), linked_intervals=int(row[linked]),
                         central_intervals=int(row[central]), central_linked_intervals=int(row[central_linked]),
                         whole_connected=int(row['walk']), whole_speed_compatible=int(row['passed']))
    assert len(rows) == 3000
    return rows


def main():
    with (road/'sumo/rows.csv').open(newline='') as handle:
        sumo = collect('SUMO', handle)
    with gzip.open(road/'porto/porto-road-requests.csv.gz','rt',newline='') as handle:
        porto = collect('Porto', handle)
    city_results = {'SUMO':sumo, 'Porto':porto}
    totals = {}
    used = set()
    merged = []
    with (destination/'endpoints/per-trip.csv').open(newline='') as handle:
        for record in csv.DictReader(handle):
            city, seed = record['city'], record['seed']
            index = int(record['draw_index']), int(record['pack_index'])
            key = (city,seed,*index)
            assert key not in used
            used.add(key)
            generated = city_results[city][(*index, seed)]
            observed = city_results[city][(*index, 'observed')]
            assert generated['intervals'] == observed['intervals']
            assert generated['central_intervals'] == observed['central_intervals']
            merged.append({**record, **generated, **{'observed_'+k:v for k,v in observed.items()}})
            counts = totals.setdefault(city+'-'+seed, Counter())
            within = max(float(record[k]) for k in ('origin_error_m','destination_error_m')) <= 500
            walk = generated['whole_connected']
            counts.update(n=1, walk=walk, both_requested_cells_and_walk=int(record['both_requested_cells_hit'])*walk,
                          both_within_500m_and_walk=int(within)*walk,
                          observed_walk=observed['whole_connected'],
                          generated_and_observed_walk=walk*observed['whole_connected'])
    assert len(merged) == 4000 and all(v['n']==1000 for v in totals.values())
    with (destination/'paired-trips.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(merged)
    (destination/'paired-counts.json').write_text(json.dumps(totals,indent=2)+'\n')


if __name__ == '__main__':
    main()
