"""Collect timestamped GTT positions for one day. Run in a private data directory."""
import fcntl
import json
import sqlite3
import time
import urllib.request
from google.transit import gtfs_realtime_pb2

url = 'https://percorsieorari.gtt.to.it/das_gtfsrt/vehicle_position.aspx'
lock = open('collector.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
db = sqlite3.connect('gtt.sqlite')
db.execute('create table if not exists positions(vehicle text, trip text, service_date text, route text, t integer, lat real, lon real, speed real, primary key(vehicle,trip,service_date,t))')
db.execute('create table if not exists polls(fetched integer, feed integer, entities integer, added integer, error text)')
stop = time.monotonic() + 86400
while time.monotonic() < stop:
    now = int(time.time())
    feed = gtfs_realtime_pb2.FeedMessage()
    added = 0
    error = None
    try:
        with urllib.request.urlopen(url, timeout=25) as response:
            feed.ParseFromString(response.read())
        if not feed.HasField('header') or not feed.IsInitialized():
            raise ValueError('Incomplete GTFS feed')
        for entity in feed.entity:
            v = entity.vehicle
            p = v.position
            if not (v.timestamp and v.vehicle.id and v.trip.trip_id and p.HasField('latitude') and p.HasField('longitude')):
                continue
            if not (44.7 <= p.latitude <= 45.4 and 7.2 <= p.longitude <= 8.2):
                continue
            added += db.execute('insert or ignore into positions values(?,?,?,?,?,?,?,?)', (v.vehicle.id,v.trip.trip_id,v.trip.start_date,v.trip.route_id,v.timestamp,p.latitude,p.longitude,p.speed if p.HasField('speed') else None)).rowcount
    except Exception as exc:
        error = repr(exc)
    row = (now, feed.header.timestamp,len(feed.entity),added,error)
    db.execute('insert into polls values(?,?,?,?,?)',row)
    db.commit()
    print(json.dumps(row),flush=True)
    time.sleep(60)
