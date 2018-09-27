import numpy as np
import scipy.spatial.distance as ssdistance

def calc_distance_sqerror(arr, dist_func_name='cosine'):
    """
    If you get an error, you might have cells with all nan.
    """
    dist_func = getattr(ssdistance, dist_func_name)
    distance = np.zeros((arr.shape[1], arr.shape[1]))
    num_cells = arr.shape[1]
    for num1 in xrange(num_cells):
        ts1 = arr[:, num1]
        ts1 = ts1[~np.isnan(ts1)]
        for num2 in xrange(num_cells):
            ts2 = arr[:, num2]
            ts2 = ts2[~np.isnan(ts2)]
            mts1, mts2 = fill_short_series(ts1, ts2)
            min_frame = min(len(ts1), len(ts2))
            distance[num1, num2] = dist_func(mts1, mts2) / min_frame
    return distance

def fill_short_series(ts1, ts2):
    """make two time series the same length by repeating for shorter ones"""
    TILE = 100 # FIXME. Should be computationally determined.
    diff_len = len(ts1) - len(ts2)
    if diff_len < 0:
        ts1 = np.tile(ts1, TILE)[:len(ts1)-diff_len]
    elif diff_len > 0:
        ts2 = np.tile(ts2, TILE)[:len(ts2)+diff_len]
    return ts1, ts2

def apply_clusters(clusters, arr):
    re_arr = np.zeros([arr.shape[0], arr.shape[1]])
    
    ind = 0 
    
    for i in range(0,len(clusters)):
        for j in clusters[i][1]:
            re_arr[ind,:] = arr[j, :]
            ind += 1
            
    return re_arr

def kshape_clusters(arr, cluster_num, ax=1):
    from kshape.core import kshape, zscore
    clusters = kshape(zscore(arr, ax), cluster_num)
    re_arr = apply_clusters(clusters, arr)
    return re_arr
    