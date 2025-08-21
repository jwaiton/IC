"""
-----------------------------------------------------------------------
                               Thekla
-----------------------------------------------------------------------
Thekla, the city forever under construction.

This city is a rework of Esmeralda, reading Sophronia hits and applying
the new relevant tooling before extracting topology information.

Work in progress, as it shall be forever (ironic but fitting).
"""

import numpy  as np
import tables as tb
import pandas as pd

from os   .path  import expandvars
from scipy.stats import multivariate_normal
from numpy       import nan_to_num

from .  components import city
from .  components import collect
from .  components import copy_mc_info
from .  components import print_every
from .  components import hits_corrector
from .  components import hits_thresholder
from .  components import hits_and_kdst_from_files
from .  components import compute_and_write_tracks_info
from .  components import identity

from .. types.symbols import HitEnergy

from .. core.configure         import EventRangeType
from .. core.configure         import OneOrManyFiles
from .. core                import tbl_functions        as tbl
from .. evm                 import event_model          as evm
from .. dataflow            import dataflow             as fl

from .. dataflow.dataflow      import push
from .. dataflow.dataflow      import pipe

from .. reco.hits_functions    import cut_over_Q
from .. reco.hits_functions    import drop_isolated

from ..  io.run_and_event_io import run_and_event_writer
from ..  io.hits_io          import hits_writer
from ..  io.hits_io          import hits_from_df
from ..  io.hits_io          import hitc_from_df
from ..  io.kdst_io          import kdst_from_df_writer

from typing import List
from typing import Optional

# Temporary. The removal of the event model will fix this.
def hitc_to_df_(hitc):
    columns = "event time npeak Xpeak Ypeak nsipm X Y Xrms Yrms Z Q E Qc Ec track_id Ep".split()
    columns = {col:[] for col in columns}

    for hit in hitc.hits:
        columns["event"   ].append(hitc.event)
        columns["time"    ].append(hitc.time)
        columns["npeak"   ].append(hit .npeak)
        columns["Xpeak"   ].append(hit .Xpeak)
        columns["Ypeak"   ].append(hit .Ypeak)
        columns["nsipm"   ].append(hit .nsipm)
        columns["X"       ].append(hit .X)
        columns["Y"       ].append(hit .Y)
        columns["Xrms"    ].append(hit .Xrms)
        columns["Yrms"    ].append(hit .Yrms)
        columns["Z"       ].append(hit .Z)
        columns["Q"       ].append(hit .Q)
        columns["E"       ].append(hit .E)
        columns["Qc"      ].append(hit .Qc)
        columns["Ec"      ].append(hit .Ec)
        columns["track_id"].append(hit .track_id)
        columns["Ep"      ].append(hit .Ep)
    return pd.DataFrame(columns)


@city
def thekla(    files_in         : OneOrManyFiles
             , file_out         : str
             , compression      : str
             , event_range      : EventRangeType
             , print_mod        : int
             , detector_db      : str
             , run_number       : int
             , threshold        : float
             , drop_distance    : List[float]
             , drop_minimum     : int
             , energy_type      : HitEnergy
             , paolina_params   : dict
             , corrections      : Optional[dict] = None
             ):
    """
    The city corrects Sophronia hits energy and extracts topology information
    ----------
    Parameters
    ----------
    files_in    : str, filepath
         Input file
    file_out    : str, filepath
         Output file
    compression : str
         Default  'ZLIB4'
    event_range : int /'all_events'
         Number of events from files_in to process
    print_mod   : int
         How frequently to print events
    detector_db : str
         detector database name
    run_number  : int
         Has to be negative for MC runs

    threshold     : float
        Threshold to be applied to all SiPMs (PEs)
    drop_distance : list[float]
        Distance to check if a SiPM has active neighbours
    drop_minimum  : int
        Minimum number of hits to classify a cluster

    INSERT THE PAOLINA PARAMS IN HERE BUT AS ISAURA I GUESS?

    corrections : dict
        filename : str
            Path to the file holding the correction maps
        apply_temp : bool
            Whether to apply temporal corrections
        norm_strat : NormStrategy
            Normalization strategy
        norm_value : float, optional
            Normalization value in case of `norm_strat = NormStrategy.custom`

    ----------
    Input
    ----------
    Sophronia output
    ----------
    Output
    ----------
    RECO    : hits table
    MC info : (if run number <=0)
    """

    # mapping functionals
    hitc_to_df     = fl.map(hitc_to_df_, item="hits") # <- reusing keys naughty
    df_to_hitc     = fl.map(hitc_from_df, item="hits") # <- reusing keys naughty
    correct_hits   = fl.map(hits_corrector(**corrections) if corrections is not None else identity
                            , item="hits")

    cut_sensors    = fl.map(cut_over_Q   (threshold, ['E', 'Ec']), item = 'hits')
    drop_sensors   = fl.map(drop_isolated(drop_distance, ['E', 'Ec'], drop_minimum), item = 'hits')

    # spy components
    event_count_in            = fl.spy_count()
    event_count_post_cuts     = fl.spy_count()
    event_count_post_topology = fl.count()
    filter_out_none = fl.filter(lambda x: x is not None, args = "kdst")
    event_number_collector = collect()
    collect_evts = "event_number", fl.fork( event_number_collector.sink
                                          , event_count_post_topology.sink)

    with tb.open_file(file_out, 'w', filters=tbl.filters(compression)) as h5out:
        write_event_info = fl.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))
        write_hits       = fl.sink(hits_writer(h5out),          args="hits")
        write_kdst_table = fl.sink( kdst_from_df_writer(h5out), args =  "kdst"      )
        compute_tracks   = compute_and_write_tracks_info(paolina_params, h5out, hit_type=energy_type)
        # it should write the reco'd hits and also the topological information
        result = push(source = hits_and_kdst_from_files(files_in, 'RECO', 'Events'),
                      pipe   = pipe(fl.slice(*event_range, close_all=True)  ,
                                    print_every(print_mod)                  ,
                                    event_count_in.spy                      ,
                                    correct_hits                            ,
                                    hitc_to_df                              ,
                                    cut_sensors                             ,
                                    drop_sensors                            ,
                                    df_to_hitc                              ,
                                    event_count_post_cuts.spy               ,
                                    fl.fork(compute_tracks                      ,
                                            write_hits                          ,
                                            (filter_out_none, write_kdst_table) ,
                                            write_event_info                    ,
                                            collect_evts
                                            )
                                    ),
                      result = dict(events_in   = event_count_in        .future,
                                    events_out  = event_count_post_cuts .future,
                                    evtnum_list = event_number_collector.future))

                                
        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)  
                                
        return result
                      
                      
                      
