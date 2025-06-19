import numpy  as np
import pandas as pd

from scipy.spatial.distance import cdist
from itertools   import compress
from copy        import deepcopy
from typing      import List
from .. evm  import event_model as evm
from .. types.ic_types      import NN
from .. types.ic_types      import xy

from typing import Optional
from typing import List
from typing import Callable

def split_energy(total_e, clusters):
    if len(clusters) == 1:
        return [total_e]
    qs = np.array([c.Q for c in clusters])
    return total_e * qs / np.sum(qs)

def merge_NN_hits(hits : List[evm.Hit], same_peak : bool = True) -> List[evm.Hit]:
    """ Returns a list of the hits where the  energies of NN hits are distributed to the closest hits such that the added energy is proportional to
    the hit energy. If all the hits were NN the function returns empty list. """
    nn_hits     = [h for h in hits if h.Q==NN]
    non_nn_hits = [deepcopy(h) for h in hits if h.Q!=NN]
    passed = len(non_nn_hits)>0
    if not passed:
        return []
    hits_to_correct=[]
    for nn_h in nn_hits:
        peak_num = nn_h.npeak
        if same_peak:
            hits_to_merge = [h for h in non_nn_hits if h.npeak==peak_num]
        else:
            hits_to_merge = non_nn_hits
        try:
            z_closest  = min(abs(h.Z-nn_h.Z) for h in hits_to_merge)
        except ValueError:
            continue
        h_closest = [h for h in hits_to_merge if np.isclose(abs(h.Z-nn_h.Z), z_closest)]

        total_raw_energy = sum(h.E  for h in h_closest)
        total_cor_energy = sum(h.Ec for h in h_closest)
        for h in h_closest:
            hits_to_correct.append([h, nn_h.E * h.E / total_raw_energy, nn_h.Ec * h.Ec / total_cor_energy])

    for h, raw_e, cor_e in hits_to_correct:
        h.E  += raw_e
        h.Ec += cor_e

    return non_nn_hits

def threshold_hits(hits : List[evm.Hit], th : float, on_corrected : bool=False) -> List[evm.Hit]:
    """Returns list of the hits which charge is above the threshold. The energy of the hits below the threshold is distributed among the hits in the same time slice. """
    if th==0:
        return hits
    else:
        new_hits=[]
        for z_slice in np.unique([x.Z for x in hits]):
            slice_hits  = [x for x in hits if x.Z == z_slice]
            raw_es      = np.array([x.E  for x in slice_hits])
            cor_es      = np.array([x.Ec for x in slice_hits])
            raw_e_slice = np.   sum(raw_es)
            cor_e_slice = np.nansum(cor_es) + np.finfo(np.float64).eps

            if on_corrected:
                mask_thresh = np.array([x.Qc >= th for x in slice_hits])
            else:
                mask_thresh = np.array([x.Q  >= th for x in slice_hits])
            if sum(mask_thresh) < 1:
                hit = evm.Hit( slice_hits[0].npeak
                             , evm.Cluster(NN, xy(0,0), xy(0,0), 0)
                             , z_slice
                             , raw_e_slice
                             , xy(slice_hits[0].Xpeak, slice_hits[0].Ypeak)
                             , s2_energy_c = cor_e_slice)
                new_hits.append(hit)
                continue
            hits_pass_th = list(compress(deepcopy(slice_hits), mask_thresh))

            raw_es_new = split_energy(raw_e_slice, hits_pass_th)
            cor_es_new = split_energy(cor_e_slice, hits_pass_th)
            for hit, raw_e, cor_e in zip(hits_pass_th, raw_es_new, cor_es_new):
                hit.E  = raw_e
                hit.Ec = cor_e
                new_hits.append(hit)
        return new_hits


def cut_and_redistribute_df(cut_condition : str,
                            variables     : List[str]=[]) -> Callable:
    '''
    Apply a cut condition to a dataframe and redistribute the cut out values
    of a given variable.

    Parameters
    ----------
    df      : dataframe to be cut

    Initialization parameters:
        cut_condition : String with the cut condition (example "Q > 10")
        variables     : List with variables to be redistributed.

    Returns
    ----------
    pass_df : dataframe after applying the cut and redistribution.
    '''
    def cut_and_redistribute(df : pd.DataFrame) -> pd.DataFrame:
        pass_df = df.query(cut_condition).copy()
        if not len(pass_df): return pass_df

        with np.errstate(divide='ignore'):
            columns  =      pass_df.loc[:, variables]
            columns *= np.divide(df.loc[:, variables].sum().values, columns.sum())
            pass_df.loc[:, variables] = columns

        return pass_df

    return cut_and_redistribute


def cut_over_Q(q_cut, redist_var):
    '''
    Apply a cut over the SiPM charge condition to hits and redistribute the
    energy variables.

    Parameters
    ----------
    q_cut      : Charge value over which to cut.
    redist_var : List with variables to be redistributed.

    Returns
    ----------
    cut_over_Q : Function that will cut the dataframe and redistribute
    values.
    '''
    cut = cut_and_redistribute_df(f"Q > {q_cut}", redist_var)

    def cut_over_Q(df):  # df shall be an event cdst
        cdst = df.groupby(['event', 'npeak']).apply(cut).reset_index(drop=True)

        return cdst

    return cut_over_Q


def drop_isolated( distance   : List[float],
                   redist_var : Optional[List] = [],
                   nhits      : Optional[int] = 3):
    """
    Master function deciding whether to drop isolated
    hits or clusters, dependent on list provided.

    # Comment -> I think this should be changed, the logic
    for how you choose clusters or sensors is silly.

    Parameters
    ----------
    distance   : Sensor pitch in 2 or 3 dimensions
    redist_var : List with variables to be redistributed.
    nhits      : Number of hits 
    Returns
    ----------
    drop_isolated_sensors : Function that will drop the isolated sensors.
    """
    
    # distance is XY -> N
    if   len(distance) == 2:
        drop = drop_isolated_sensors(distance, redist_var)
    elif len(distance) == 3:
        drop = drop_isolated_clusters(distance, nhits, redist_var)
    else:
        raise ValueError(f"Invalid drop_dist parameter: expected 2 or 3 entries, but got {len(distance)}.")


    def drop_isolated(df): # df shall be an event cdst
        df = df.groupby(['event', 'npeak']).apply(drop).reset_index(drop=True)

        return df

    return drop_isolated


def drop_isolated_sensors(distance  : List[float]=[10., 10.],
                          variables : List[str  ]=[        ]) -> Callable:
    """
    Drops rogue/isolated hits (SiPMs) from a groupedby dataframe.

    Parameters
    ----------
    df      : GroupBy ('event' and 'npeak') dataframe

    Initialization parameters:
        distance  : Distance to check for other sensors. Usually equal to sensor pitch.
        variables : List with variables to be redistributed.

    Returns
    -------
    pass_df : hits after removing isolated hits
    """
    dist = np.sqrt(distance[0] ** 2 + distance[1] ** 2)

    def drop_isolated_sensors(df : pd.DataFrame) -> pd.DataFrame:
        x       = df.X.values
        y       = df.Y.values
        xy      = np.column_stack((x,y))
        dr2     = cdist(xy, xy) # compute all square distances

        if not np.any(dr2>0):
            return df.iloc[:0] # Empty dataframe

        closest = np.apply_along_axis(lambda d: d[d > 0].min(), 1, dr2) # find closest that it's not itself
        mask_xy = closest <= dist # take those with at least one neighbour
        pass_df = df.loc[mask_xy, :].copy()

        with np.errstate(divide='ignore'):
            columns  = pass_df.loc[:, variables]
            columns *= np.divide(df.loc[:, variables].sum().values, columns.sum())
            pass_df.loc[:, variables] = columns

        return pass_df

    return drop_isolated_sensors


def drop_isolated_clusters(distance   :  List[float]= [10., 10., 1.],
                           nhits      :  int        = 3,
                           variables  :  List[str  ]= [      ]) -> Callable:
    '''
    Drops isolated clusters of hits (SiPMs).

    Parameters
    ----------
    df       : Groupby ('event' and 'npeak') dataframe

    Initialisation parameters:
        distance  : Distance to check for other sensors, equal to sensor pitch and z rebinning.
        nhits     : Number of hits to classify a cluster.
        variables : List of variables to be redistributed (generally the energies)
    '''


    def drop_event(df):
        # normalise distances
        x = df.X.values / distance[0]
        y = df.Y.values / distance[1]
        z = df.Z.values / distance[2]

        xyz = np.column_stack((x, y, z))
        dr3 = cdist(xyz, xyz)
        # normalised, so distance square root of the dimensions
        dist = np.sqrt(3)

        # If there aren't any clusters, return empty df
        if not np.any(dr3>0):
            return df.iloc[:0] 
        
        # create mask for clusters by determining how many hits are within range.
        closest = np.apply_along_axis(lambda d: len(d[d < dist]), 1, dr3)
        mask_xyz = closest > nhits

        # expand mask to include neighbors of neighbors etc to avoid removing edges
        expanded_mask = mask_xyz.copy()
        for _ in range(len(df)):
            # find neighbors of currently included hits and combine
            new_neighbors = np.any(dr3[:, expanded_mask] < dist, axis=1)
            new_mask = expanded_mask | new_neighbors
            # if no new neighbors are added break
            if np.array_equal(new_mask, expanded_mask):
                break
            expanded_mask = new_mask
        mask_xyz = expanded_mask

        pass_df = df.loc[mask_xyz, :].copy()
        # reweighting
        with np.errstate(divide='ignore'):
            columns = pass_df.loc[:, variables]
            columns *= np.divide(df.loc[:,variables].sum().values,columns.sum())
            pass_df.loc[:, variables] = columns

        return pass_df
    
    return drop_event