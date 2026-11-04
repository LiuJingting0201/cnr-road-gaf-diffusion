import hashlib,json
from pathlib import Path
import numpy as np
from cnr.original_gaf import trajectory_sequence,fit_channels,encode_frozen,encode_hilbert_sequences,raw_channels
from cnr_trajectory.encoding.gaf import encode_trajectory_frame
import pandas as pd
import pytest


def test_vendored_sources_are_unchanged():
    base=Path('vendor/original_cnr');p=json.loads((base/'PROVENANCE.json').read_text())
    for row in p['files']:assert hashlib.sha256((base/row['path']).read_bytes()).hexdigest()==row['sha256']


def test_full_frame_parity_against_unmodified_original_pipeline():
    # Integer grid steps avoid the intentional out-of-bounds adapter extension.
    xy=np.random.default_rng(4).uniform(0,1,(3,40,2));flat=xy.reshape(-1,2)
    frame=pd.DataFrame({'vehicle_id':np.repeat(['car0','car1','car2'],40),'type':np.repeat(['bus','car','bike'],40),'lon':flat[:,0],'lat':flat[:,1]})
    ref=encode_trajectory_frame(frame)
    lo=flat.min(0);hi=flat.max(0)
    sequences=np.stack([trajectory_sequence(x,lo,hi)[0] for x in xy])
    np.testing.assert_array_equal(sequences,ref.sequences)
    clo,chi=fit_channels(sequences,batch=1);images,_=encode_frozen(sequences,clo,chi,batch=1)
    np.testing.assert_array_equal(images,ref.images)


def test_frozen_encoder_is_batch_independent():
    seq=np.random.default_rng(5).random((3,224));lo,hi=fit_channels(seq)
    whole,_=encode_frozen(seq,lo,hi)
    for i in range(3):np.testing.assert_array_equal(whole[i:i+1],encode_frozen(seq[i:i+1],lo,hi)[0])


def test_original_failure_not_silently_repaired():
    from cnr_trajectory.encoding.sequence import squeeze,duplicate
    x=list(range(225));assert len(duplicate(squeeze(x,224),224))==225


def test_hilbert_is_locally_constant():
    from cnr_trajectory.encoding.hilbert import hilbert_index
    assert hilbert_index(.231,.271,.0037)==hilbert_index(.23100001,.27100001,.0037)
