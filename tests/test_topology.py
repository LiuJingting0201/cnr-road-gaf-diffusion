import numpy as np
from cnr.topology import LaneConsistency


def network(tmp_path):
    p=tmp_path/'net.xml'
    p.write_text('''<net><location netOffset="0,0"/>
    <edge id="a"><lane id="a_0" index="0" allow="passenger" shape="0,0 10,0"/></edge>
    <edge id=":via" function="internal"><lane id=":via_0" index="0" allow="passenger" shape="10,0 15,0 15,5"/></edge>
    <edge id="b"><lane id="b_0" index="0" allow="passenger" shape="15,5 15,15"/></edge>
    <edge id="overpass"><lane id="overpass_0" index="0" allow="passenger" shape="5,-5 5,5"/></edge>
    <connection from="a" fromLane="0" to="b" toLane="0" via=":via_0"/>
    <connection from=":via" fromLane="0" to="b" toLane="0"/>
    </net>''')
    return LaneConsistency(p,radius=.2)


def test_preserves_internal_lane_and_direction(tmp_path):
    m=network(tmp_path);a=m.ids['a_0'];b=m.ids['b_0']
    assert m.successors[a]=={m.ids[':via_0']}
    assert m.route_distance(a,9,b,1,'private_car')==12
    assert np.isinf(m.route_distance(b,1,a,9,'private_car'))
    good=m.match([[7,0],[9,0],[11,0],[15,0],[15,3],[15,6]],'private_car')
    assert good['graph_consistent']
    bad=m.match([[9,0],[7,0],[3,0]],'private_car')
    assert bad['geometry_covered'] and not bad['graph_consistent']


def test_crossing_without_connection_is_not_a_junction(tmp_path):
    m=network(tmp_path)
    r=m.match([[2,0],[4,0],[5,0],[5,2],[5,4]],'private_car')
    assert r['geometry_covered'] and r['minimum_incompatible_transitions']>0
    assert m.match([[0,0],[1,0]],'pedestrian')['status']=='unsupported_pedestrian'


def test_offroad_and_ambiguous_crossing(tmp_path):
    m=network(tmp_path)
    assert m.match([[2,0],[5,0],[8,0]],'private_car')['graph_consistent']
    r=m.match([[2,0],[2,3],[8,0]],'private_car')
    assert r['missing_candidate_points']==1 and not r['graph_consistent']


def test_multiple_positions_on_one_bent_lane(tmp_path):
    p=tmp_path/'bend.xml'
    p.write_text('<net><location netOffset="0,0"/><edge id="a">'
                 '<lane id="a_0" index="0" allow="passenger" shape="0,0 10,0 10,10"/>'
                 '</edge></net>')
    m=LaneConsistency(p,radius=.75)
    x=np.array([[9.6,.6],[9.4,.4],[9.8,1.2]])
    candidates,_=m.candidates(x,'private_car')
    assert len(candidates[0])==2 and len(candidates[1])==2
    # Individually nearest projections move 10.6 -> 9.4 backwards; the full
    # candidate set admits a forward sequence without relaxing any tolerance.
    assert m.match(x,'private_car')['graph_consistent']
    assert not m.match(x[::-1],'private_car')['graph_consistent']
