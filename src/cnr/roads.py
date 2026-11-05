"""Exact nearest-centerline diagnostics, not a topological map matcher."""
import xml.etree.ElementTree as ET
import numpy as np
import shapely
from shapely import STRtree


# class2 does not encode SUMO vClass. Treat these as explicit hypotheses.
CLASS_TO_VCLASSES = {
    'pedestrian': ('pedestrian',), 'bus': ('bus',),
    'motorcycle': ('motorcycle',), 'private_car': ('passenger',),
    'truck': ('truck',), 'delivery': ('delivery',),
    'bycicle': ('bicycle',), 'taxi_or_uber': ('taxi', 'passenger'),
}


def allows(attributes, vclass):
    if 'allow' in attributes:
        allowed = attributes['allow'].split()
        return 'all' in allowed or vclass in allowed
    denied = attributes.get('disallow', '').split()
    return 'all' not in denied and vclass not in denied


class RoadNetwork:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.location = root.find('location').attrib
        self.lanes, self.shapes = [], []
        for edge in root.findall('edge'):
            for lane in edge.findall('lane'):
                coords = [tuple(map(float, p.split(',')[:2])) for p in lane.get('shape', '').split()]
                if len(coords) < 2:
                    continue
                self.lanes.append(dict(lane.attrib, edge=edge.get('id'), function=edge.get('function', 'normal')))
                self.shapes.append(shapely.LineString(coords))
        self.shapes = np.array(self.shapes, dtype=object)
        if not len(self.shapes):
            raise ValueError('No lane shapes')
        self.connections = [dict(c.attrib) for c in root.findall('connection')]
        self.trees = {None: STRtree(self.shapes)}

    def distances(self, xy, label=None):
        if label not in self.trees:
            vc = CLASS_TO_VCLASSES[label]
            keep = [i for i, a in enumerate(self.lanes) if any(allows(a, c) for c in vc)]
            if not keep:
                raise ValueError(f'No permitted lane for {label}')
            self.trees[label] = STRtree(self.shapes[keep])
        tree = self.trees[label]
        points = shapely.points(np.asarray(xy).reshape(-1, 2))
        nearest = tree.nearest(points)
        return np.asarray(shapely.distance(points, tree.geometries[nearest]))


def distance_summary(d):
    return {'points': len(d), 'median_m': float(np.median(d)),
            'p95_m': float(np.quantile(d, .95)), 'max_m': float(np.max(d)),
            'within_2m': float(np.mean(d <= 2)), 'within_5m': float(np.mean(d <= 5))}
