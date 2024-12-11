import os
import numpy  as np
import tables as tb
import time   as tm

from pytest import fixture

from numpy.testing import assert_allclose

from . dst_io              import load_dst
from .. types.ic_types     import xy
from .. evm.event_model    import Cluster
from .. evm.event_model    import Hit
from .. evm.event_model    import HitCollection
from .  hits_io            import hits_writer
from .  hits_io            import load_hits
from .  hits_io            import load_hits_skipping_NN
from .. types.ic_types     import NN

@fixture(scope='session')
def TlMC_hits(ICDATADIR):
    hits_file_name = "dst_NEXT_v1_00_05_Tl_ACTIVE_100_0_7bar_DST_10.h5"
    hits_file_name = os.path.join(ICDATADIR, hits_file_name)
    hits = load_hits(hits_file_name)
    return hits


@fixture(scope='session')
def TlMC_hits_skipping_NN(ICDATADIR):
    hits_file_name = "dst_NEXT_v1_00_05_Tl_ACTIVE_100_0_7bar_DST_10.h5"
    hits_file_name = os.path.join(ICDATADIR, hits_file_name)
    hits = load_hits_skipping_NN(hits_file_name)
    return hits

@fixture(scope='session')
def hits_toy_data(ICDATADIR):
    npeak  = np.array   ([0]*25 + [1]*30 + [2]*35 + [3]*10)
    nsipm  = np.arange  (1000, 1100)
    x      = np.linspace( 150,  250, 100)
    y      = np.linspace(-280, -180, 100)
    xrms   = np.linspace(   1,   80, 100)
    yrms   = np.linspace(   2,   40, 100)
    z      = np.linspace(   0,  515, 100)
    q      = np.linspace( 1e3,  1e3, 100)
    e      = np.linspace( 2e3,  1e4, 100)
    x_peak = np.array([(x * e).sum() / e.sum()] * 100)
    y_peak = np.array([(y * e).sum() / e.sum()] * 100)

    hits_filename = os.path.join(ICDATADIR, "toy_hits.h5")
    return hits_filename, (npeak, nsipm, x, y, xrms, yrms, z, q, e, x_peak, y_peak)



def test_load_hits_load_events(TlMC_hits):
    hits = TlMC_hits
    assert np.equal(list(hits.keys()),
                    [5000000, 5000003, 5000004, 5000007, 5000010]).all()

def test_load_hits_double_ratio_e_q_equals_one(TlMC_hits):
    hits = TlMC_hits
    Ein = []
    Qin = []
    Emax = []
    Qmax = []
    for event, hitc in hits.items():
        E = []
        Q = []

        for hit in hitc.hits:
            if(hit.Q != NN):
                E.append(hit.E)
                Q.append(hit.Q)

        Emax.append(np.max(E))
        Qmax.append(np.max(Q))
        E   .pop   (np.argmax(E))
        Q   .pop   (np.argmax(Q))
        Ein .extend(E)
        Qin .extend(Q)


    r1 = np.mean(Emax)/np.mean(Qmax)
    r2 = np.mean(Ein)/np.mean(Qin)
    r = r1/r2
    np.isclose(r, 1, rtol=0.1)

def test_load_hits_double_ratio_e_q_equals_one_skipping_NN(TlMC_hits_skipping_NN):
    hits = TlMC_hits_skipping_NN
    Ein = []
    Qin = []
    Emax = []
    Qmax = []
    for event, hitc in hits.items():
        E = []
        Q = []

        for hit in hitc.hits:
            E.append(hit.E)
            Q.append(hit.Q)

        Emax.append(np.max(E))
        Qmax.append(np.max(Q))
        E   .pop   (np.argmax(E))
        Q   .pop   (np.argmax(Q))
        Ein .extend(E)
        Qin .extend(Q)


    r1 = np.mean(Emax)/np.mean(Qmax)
    r2 = np.mean(Ein)/np.mean(Qin)
    r = r1/r2
    np.isclose(r, 1, rtol=0.1)

def test_hits_writer(config_tmpdir, hits_toy_data):
    output_file = os.path.join(config_tmpdir, "test_hits.h5")

    _, (npeak, nsipm, x, y, xstd, ystd, z, q, e, x_peak, y_peak) = hits_toy_data

    with tb.open_file(output_file, 'w') as h5out:
        write = hits_writer(h5out)
        hits = HitCollection(-1, -1)
        for i in range(len(x)):
            c = Cluster(q[i], xy(x[i], y[i]), xy(xstd[i], ystd[i]), nsipm[i])
            h = Hit(npeak[i], c, z[i], e[i], xy(x_peak[i], y_peak[i]))
            hits.hits.append(h)
        write(hits)

    dst = load_dst(output_file, group = "RECO", node = "Events")
    assert_allclose(npeak, dst.npeak.values)
    assert_allclose(x_peak , dst.Xpeak .values)
    assert_allclose(y_peak , dst.Ypeak .values)
    assert_allclose(nsipm, dst.nsipm.values)
    assert_allclose(x    , dst.X    .values)
    assert_allclose(y    , dst.Y    .values)
    assert_allclose(np.sqrt(xstd), dst.Xrms .values)
    assert_allclose(np.sqrt(ystd), dst.Yrms .values)
    assert_allclose(z    , dst.Z    .values)
    assert_allclose(q    , dst.Q    .values)
    assert_allclose(e    , dst.E    .values)


def test_hit_time_is_in_second(ICDATADIR):
    output_file = os.path.join(ICDATADIR, "hits_1hit_perSiPM_30pes_6817_trigger2_v0.9.9_20190111_krth1600.0.h5")
    the_hits = load_hits(output_file)

    for evt, hit_coll in the_hits.items():
        evt_time = hit_coll.time
        year = int(tm.strftime("%Y", tm.localtime(evt_time)))

        assert 2008 < year < 2050
