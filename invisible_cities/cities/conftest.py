import os
import pytest
import shutil

import numpy  as np
import tables as tb

@pytest.fixture(scope="session")
def Th228_hits(ICDATADIR):
    filename = "228Th_10evt_hits.h5"
    filename = os.path.join(ICDATADIR, filename)
    return filename

@pytest.fixture(scope="session")
def Th228_tracks(ICDATADIR):
    filename = "228Th_10evt_tracks.h5"
    filename = os.path.join(ICDATADIR, filename)
    return filename

@pytest.fixture(scope="session")
def Th228_deco(ICDATADIR):
    filename = "228Th_10evt_deco.h5"
    filename = os.path.join(ICDATADIR, filename)
    return filename


@pytest.fixture(scope="session")
def Th228_deco_separate(ICDATADIR):
    filename = "228Th_10evt_deco_separate.h5"
    filename = os.path.join(ICDATADIR, filename)
    return filename


@pytest.fixture(scope="session")
def Th228_hits_missing(Th228_hits, config_tmpdir):
    """Copy input file and remove the hits from the first event"""
    outpath = os.path.basename(Th228_hits).replace(".h5", "_missing_hits.h5")
    outpath = os.path.join(config_tmpdir, outpath)
    shutil.copy(Th228_hits, outpath)
    with tb.open_file(outpath, "r+") as file:
        first_evt = file.root.Run.events[0][0]
        evt_rows  = [row[0] == first_evt for row in file.root.RECO.Events]
        n_delete  = sum(evt_rows)
        file.root.RECO.Events.remove_rows(0, n_delete)
    return outpath

@pytest.fixture(scope="session")
def next100_mc_krmap(ICDATADIR):
    filename = "map_NEXT100_MC.h5"
    filename = os.path.join(ICDATADIR, filename)
    return filename


@pytest.fixture(scope='session')
def Kr_pmaps_run4628_filename(ICDATADIR):
    filename = os.path.join(ICDATADIR, "Kr_pmaps_run4628.h5")
    return filename

