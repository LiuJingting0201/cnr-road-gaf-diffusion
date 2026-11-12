"""Join a private GTT position snapshot to its service calendar and scheduled terminals."""
import csv,gzip,io,itertools,json,math,sys,zipfile,statistics
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo

fixes,static,out=sys.argv[1:]
out=Path(out);out.mkdir(parents=True,exist_ok=True)
z=zipfile.ZipFile(static)
def table(name):
    with z.open(name+'.txt') as f:
        yield from csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig'))
def index(name,key):
    return {r[key]:r for r in table(name)}
def secs(stamp):
    h,m,s=map(int,stamp.split(':'));return h*3600+m*60+s
def distance(a,b):
    lat1,lon1,lat2,lon2=map(math.radians,(*a,*b))
    h=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 12742000*math.asin(min(1,math.sqrt(h)))
def quant(a):
    a=sorted(a)
    return [round(a[round(p*(len(a)-1))],2) for p in [.1,.5,.9]] if a else []
routes=index('routes','route_id');trips=index('trips','trip_id');stops=index('stops','stop_id')
fleet=index('vehicles','vehicle_id'); categories=index('vehicle_categories','vehicle_category_id')
calendar=index('calendar','service_id')
exceptions={(r['service_id'],r['date']):r['exception_type'] for r in table('calendar_dates')}
def active(sid,day):
    if (sid,day) in exceptions:return exceptions[sid,day]=='1'
    r=calendar.get(sid)
    return bool(r and r['start_date']<=day<=r['end_date'] and r[datetime.strptime(day,'%Y%m%d').strftime('%A').lower()]=='1')
groups=defaultdict(list); speedfields=0
with gzip.open(fixes,'rt') as f:
    for v,tid,day,route,t,lat,lon,speed in csv.reader(f):
        groups[v,tid,day].append((int(t),(float(lat),float(lon)),route));speedfields+=bool(speed)
seen={tid for v,tid,day in groups};ends={}
for r in table('stop_times'):
    tid=r['trip_id']
    if tid not in seen:continue
    seq=int(r['stop_sequence'])
    first,last=ends.get(tid,(r,r))
    ends[tid]=(r if seq<int(first['stop_sequence']) else first,r if seq>int(last['stop_sequence']) else last)
summary=Counter(); bymode=defaultdict(Counter);dtlist=[];speeds=[];dur=[];vehicles=defaultdict(set);rr=defaultdict(set)
with gzip.open(out/'segments.csv.gz','wt',newline='') as f:
    w=csv.writer(f);w.writerow(['unix_seconds','vehicle','trip','service_date','route','segment','mode','lat','lon','interval_speed_mps','terminal_covered','fleet_category','class_check'])
    for (v,tid,day),rows in groups.items():
        rows.sort(); trip=trips.get(tid)
        summary['points']+=len(rows);summary['groups']+=1
        if not trip or not active(trip['service_id'],day) or any(r[2] and r[2]!=trip['route_id'] for r in rows):continue
        route=trip['route_id'];mode={'3':'bus','0':'tram'}.get(routes[route]['route_type'],'other')
        summary['matched_groups']+=1;bymode[mode]['matched_groups']+=1
        parts=[[]]
        for t,pos,_ in rows:
            velocity=None
            if parts[-1]:
                prev,prevpos,_=parts[-1][-1];dt=t-prev;dist=distance(pos,prevpos)
                dtlist.append(dt)
                if dt<=0 or dt>180 or dist>35*dt:
                    summary['break_gap' if dt<=0 or dt>180 else 'break_speed']+=1;parts.append([])
                else:velocity=dist/dt
            parts[-1].append((t,pos,velocity))
        for num,part in enumerate(parts):
            duration=part[-1][0]-part[0][0]
            metres=sum(distance(p[1],q[1]) for p,q in zip(part,part[1:]))
            if len(part)<10 or duration<600 or metres<500 or mode=='other':continue
            cat_id=fleet.get(v,{}).get('vehicle_category_id','')
            category=categories.get(cat_id, categories.get(cat_id+v[-1],{})).get('vehicle_category_name','')
            label=category.upper()
            rail='TRAM' in label or 'T.P.I.R.' in label
            check='unknown' if not label else ('conflict' if (mode=='tram')!=rail else 'consistent')
            summary['class_'+check]+=1
            ok=False
            if tid in ends:
                a,b=ends[tid]
                def stoppos(row):
                    st=stops[row['stop_id']];return float(st['stop_lat']),float(st['stop_lon'])
                start,end=secs(a['departure_time']),secs(b['arrival_time'])
                midnight=datetime.strptime(day,'%Y%m%d').replace(tzinfo=ZoneInfo('Europe/Rome')).timestamp()
                ok=(distance(part[0][1],stoppos(a))<=200 and distance(part[-1][1],stoppos(b))<=200 and end>start and duration>=.7*(end-start) and abs(part[0][0]-midnight-start)<=900 and abs(part[-1][0]-midnight-end)<=900)
            bymode[mode]['segments']+=1;bymode[mode]['points']+=len(part);bymode[mode]['terminal_covered']+=int(ok)
            vehicles[mode].add(v);rr[mode].add(route);dur.append(duration)
            for t,pos,velocity in part:
                if velocity is not None:speeds.append(velocity)
                w.writerow((t,v,tid,day,route,num,mode,*pos,'' if velocity is None else round(velocity,3),int(ok),category,check))
for m in bymode:bymode[m].update(vehicles=len(vehicles[m]),routes=len(rr[m]))
report=dict(feed=next(table('feed_info'))['feed_version'],service_dates=sorted({k[2] for k in groups}),counts=summary,modes=bymode,reported_speed_points=speedfields,interfix_seconds_p10_50_90=quant(dtlist),segment_duration_seconds_p10_50_90=quant(dur),interval_speed_mps_p10_50_90=quant(speeds))
(out/'coverage.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
