#!/usr/bin/env bash
set -eu
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
export PYTHONPATH=$PWD/routing
test -d routing/valhalla || "$py" -m pip install --no-deps --no-index --target "$PYTHONPATH" ./pyvalhalla-3.9.1-*.whl
for city in Porto Beijing; do
    mkdir -p "tiles-$city" "build-$city"
    "$py" - "$city" <<'PY'
from pathlib import Path
import json,sys
from valhalla import get_config
c=sys.argv[1]
conf=get_config(tile_extract='',tile_dir='tiles-'+c)
conf['mjolnir']['concurrency']=6
conf['service_limits']['trace']['max_shape']=100000
conf['service_limits']['trace']['max_distance']=50000000
Path(c+'.json').write_text(json.dumps(conf))
PY
    (cd "build-$city" && "$PYTHONPATH/valhalla/bin/valhalla_build_tiles" -c "../$city.json" "../maps/$city.pbf" > build.log 2>&1)
    touch "$city.ready"
done
