import os
import tables as tb
import numpy  as np

from numpy.testing import assert_allclose

from pytest import fixture

from .. evm.event_model import Voxel
from .. evm.event_model import VoxelCollection
from .  voxels_io       import true_voxels_writer
from .  voxels_io       import load_voxels

@fixture(scope='session')
def voxels_toy_data(ICDATADIR):
    event = np.zeros(100)
    X     = np.linspace( 150,  250, 100)
    Y     = np.linspace(-280, -180, 100)
    Z     = np.linspace(   0,  100, 100)
    E     = np.linspace( 1e3,  1e3, 100)
    size  = np.reshape(np.repeat([10,10,10],100),(100,3))

    voxels_filename = os.path.join(ICDATADIR, "toy_voxels.h5")
    return voxels_filename, (event, X, Y, Z, E, size)


def test_true_voxels_writer(config_tmpdir, voxels_toy_data):

    voxels_filename, (event, X, Y, Z, E, size) = voxels_toy_data
    output_file = os.path.join(config_tmpdir, 'toy_voxels.h5')

    with tb.open_file(output_file, 'w') as h5out:
        write = true_voxels_writer(h5out)
        voxels = VoxelCollection([])
        for xv, yv, zv, ev, sv in zip(X, Y, Z, E, size):
            v = Voxel(xv, yv, zv, ev, sv)
            voxels.voxels.append(v)
        write(event[0],voxels.voxels)

    with tb.open_file(output_file) as vdst:
        assert_allclose(event, vdst.root.TrueVoxels.Voxels[:]['event'])
        assert_allclose(X,     vdst.root.TrueVoxels.Voxels[:]['X'])
        assert_allclose(Y,     vdst.root.TrueVoxels.Voxels[:]['Y'])
        assert_allclose(Z,     vdst.root.TrueVoxels.Voxels[:]['Z'])
        assert_allclose(E,     vdst.root.TrueVoxels.Voxels[:]['E'])
        assert_allclose(size,  vdst.root.TrueVoxels.Voxels[:]['size'])

def test_load_voxels(config_tmpdir, voxels_toy_data):

    voxels_filename, (event, X, Y, Z, E, size) = voxels_toy_data

    voxels_dict = load_voxels(voxels_filename)
    vX =    [voxel.X    for voxel in voxels_dict[0].voxels]
    vY =    [voxel.Y    for voxel in voxels_dict[0].voxels]
    vZ =    [voxel.Z    for voxel in voxels_dict[0].voxels]
    vE =    [voxel.E    for voxel in voxels_dict[0].voxels]
    vsize = [voxel.size for voxel in voxels_dict[0].voxels]

    assert np.allclose(X, vX)
    assert np.allclose(Y, vY)
    assert np.allclose(Z, vZ)
    assert np.allclose(E, vE)
    assert np.allclose(size, vsize)
