import hashlib
import json
from pathlib import Path

EXPECTED = {
    'midimost_with_class2.csv': '2dddc627721dfd37a4ee084c20e87beaf2dfbf7a91f87db03c9628dd835afdf8',
    'vehicle_positions_TS_New.csv': 'e033d58142a3320daa129ae9f69e6c71726dc735e1ef8f50a30345d56f4cc40e',
}
report = {}
for name, expected in EXPECTED.items():
    h = hashlib.sha256()
    with (Path('data/raw')/name).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    actual = h.hexdigest()
    report[name] = {'sha256': actual, 'matches': actual == expected}
print(json.dumps(report, indent=2))
if not all(x['matches'] for x in report.values()):
    raise SystemExit('Raw data hash mismatch; upload may be incomplete')
