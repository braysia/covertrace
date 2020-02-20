import numpy as np
from skimage.morphology import watershed

def watershed_peak_labels(traces, peak_prob, slope_prob, steps=50, min_seed_prob=0.8, min_peak_prob=0.5, min_seed_length=2):

    total_prob = peak_prob + slope_prob
    labels = np.zeros(traces.shape)
    seed_labels = np.zeros(traces.shape)

    for n, t in enumerate(traces):
        seed_idxs = _constant_thres_seg(peak_prob[n], min_length=min_seed_length, min_prob=min_seed_prob)
        seeds = _idxs_to_labels(t, seed_idxs)

        if np.sum(seeds) > 0:
            labels[n] = _agglom_watershed_peak_finder(t, seeds, total_prob[n], steps=steps, min_peak_prob=min_peak_prob)
            seed_labels[n] = seeds

    return labels.astype(int), seed_labels.astype(int)

def _constant_thres_seg(result, min_prob=0.8, min_length=8, max_gap=2):
    '''
    1D only currently
    min_prob is the minimum value for a point to be considered
    min_length is the minimum length of consecutive points to be kept
    max_gap is how many points can be missing from a peak
    '''
    candidates = np.where(result>min_prob)[0]
    diffs = np.ediff1d(candidates, to_begin=1)
    bounds = np.where(diffs>max_gap)[0] #find where which stretchs of points are separated by more than max_gap
    peak_idxs = [p for p in np.split(candidates, bounds) if len(p)>=min_length]

    return peak_idxs

def _agglom_watershed_peak_finder(trace, seeds, total_prob, steps=50, min_peak_prob=0.5):
    '''
    '''    
    if (np.sum(seeds) > 0):
        perclist_trace = np.linspace(np.nanmin(trace), np.nanmax(trace), steps)
        prev_t = perclist_trace[-1]
        cand_all = np.where(total_prob >= min_peak_prob)[0]
        
        cand_step = []
        for _t in reversed(perclist_trace):
            seed_idxs = _labels_to_idxs(seeds)
            cand_step.append(np.where(np.logical_and(trace > _t, trace <= prev_t))[0])
            cand_t = np.hstack(cand_step)
            cand = np.intersect1d(cand_all, cand_t)
            cand = np.union1d(seed_idxs, cand)
            prev_t = _t
            
            cand_mask = _idxs_to_mask(trace, cand)
            seeds = watershed(trace, markers=seeds, mask=cand_mask, watershed_line=True, compactness=5)
        
    return seeds

def _detect_peak_tracts(trace, labels, max_gap=12):
    '''
    should return peak values in tract and (y1, y2) of the height of the base
    tracts are individual arrays in list of tracts returned
    '''
    if np.sum(labels) > 0:
        peak_idxs = np.where(labels>0)[0]
        labels_no_zeros = np.array([labels[p] for p in peak_idxs])
        diffs = np.ediff1d(peak_idxs, to_begin=1)
        bounds = np.where(diffs > max_gap)[0]
        tracts = [np.unique(t) for t in np.split(labels_no_zeros, bounds)]
    else:
        tracts = []
        
    return tracts

def _idxs_to_labels(trace, seed_idxs):
    labels = np.zeros(trace.shape)
    
    for n, s in enumerate(seed_idxs):
        labels[s] = n + 1
    
    return labels

def _labels_to_idxs(labels):
    return np.where(labels>0)[0]
    
def _idxs_to_mask(trace, idx):
    mask = np.zeros_like(trace, dtype=bool)
    mask[idx] = True
    return mask

def _labels_to_peak_idxs(labels):
    peak_idxs = []
    for l in np.unique(labels):
        if l > 0:
            peak_idxs.append(np.where(labels==l)[0])
    return peak_idxs

def _edge_pts_to_idxs(edge_pts):
    e0, e1 = edge_pts
    return np.arange(e0[0], e1[0], 1)