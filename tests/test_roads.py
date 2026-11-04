import numpy as np
from cnr.roads import RoadNetwork, allows


def test_permission_defaults_and_overrides():
    assert allows({}, 'bus')
    assert not allows({'allow': 'pedestrian bicycle'}, 'passenger')
    assert allows({'disallow': 'passenger bus'}, 'pedestrian')
    assert not allows({'disallow': 'all'}, 'bus')
    assert allows({'allow': 'all'}, 'bus')


def test_exact_distance_with_permissions(tmp_path):
    f = tmp_path/'network.xml'
    f.write_text('<net><location netOffset="0,0"/><edge id="a">'
                 '<lane id="a_0" allow="passenger" shape="0,0 10,0"/>'
                 '<lane id="a_1" allow="pedestrian" shape="0,4 10,4"/>'
                 '</edge></net>')
    road = RoadNetwork(f)
    np.testing.assert_allclose(road.distances([[5,1]]), [1])
    np.testing.assert_allclose(road.distances([[5,1]], 'pedestrian'), [3])
