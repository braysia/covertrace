import numpy as np
import math
from peakutils import baseline

# STILL TO BE COMPLETED
# import new peak_width function from notebook
# make user facing functions for actually getting the data out

def normalize_by_baseline(trace, deg=1):
    '''
    should estimate baseline of degree deg
    Normalize trace by the mean of that baseline
    Returns fold activation over baseline
    '''
    base = baseline(traces, deg=deg)
    return trace / np.mean(base)

def _peak_base_pts(trace, peak_idx):
    '''
    returns two points that define the base of the peak
    '''
    base_pts = (peak_idx[0], trace[peak_idx[0]]), (peak_idx[-1], trace[peak_idx[-1]])
    base_pts = _adjust_edge_base_height(trace, base_pts, dist=4) #adjust base height for edges of traces
    return base_pts

def _plateau_pts(trace, plateau_idx):
    '''
    seed should be determined using constant thres as with the watershed algorithm
    this will take those seeds and return the left and right points for each plateau in seed.
    '''
    return (plateau_idx[0], trace[plateau_idx[0]]), (plateau_idx[-1], trace[plateau_idx[-1]])

def _slope_pts(trace, peak_idx, plateau_idx):
    '''
    need to decide how to get slope idx. Should probably be plateau - base on each side
    '''
    left_slope = (peak_idx[0], trace[peak_idx[0]]), (plateau_idx[0], trace[plateau_idx[0]])
    right_slope = (peak_idx[-1], trace[peak_idx[-1]]), (plateau_idx[-1], trace[plateau_idx[-1]])
    return left_slope, right_slope

def _mean_diff(trace, idx):
    return np.mean(np.diff(trace[idx]))

def _median_diff(trace, idx):
    return np.median(np.diff(trace[idx]))

def _avg_slope(trace, idx):
    return (trace[idx[-1]] - trace[idx[0]]) / len(idx)

def _peak_base(trace, peak_idx, base_pts=None):
    '''
    first pass, just returns the values at the left and right most indices
    peak_idx should always be sorted
    '''
    if base_pts is None:
        ((x0, y0), (x1, y1)) = _peak_base_pts(trace, peak_idx)
    else:
        ((x0, y0), (x1, y1)) = base_pts
    theta = math.degrees(math.atan2((y1 - y0), (x1 - x0)))
    return y0, y1, theta

def _adjust_edge_base_height(trace, base_pts, dist=4):
    '''
    base_pts should be list or tuple of two points at the base: [(x1, y1), (x2, y2)]
    will return new points with y1 or y2 adjusted. (shouldn't ever be both)
    '''
    left_height = base_pts[0][1]
    right_height = base_pts[1][1]
    
    left_edge = 0
    right_edge = len(trace) - 1 #because of 0 indexing

    if abs(base_pts[0][0] - left_edge) <= dist:
        left_height = right_height
    elif abs(base_pts[1][0] - right_edge) <= dist:
        right_height = left_height
        
    return (base_pts[0][0], left_height), (base_pts[1][0], right_height)

def _tract_adjusted_peak_prominence(trace, labels, tracts, bi_directional=False):
    '''
    if bidirectional is True, base can be raised or lowered
    if bidirectional is False, base can only be lowered. i.e. use the original peak base if it is lower
    returns list of prominences of peak in function
    '''
    prominences = []
    for n, t in enumerate(tracts):
        if len(t) > 1:
            left_base = np.where(labels==t[0])[0][0]
            right_base = np.where(labels==t[-1])[0][-1]
            left_height, right_height, _ = _peak_base(trace, [left_base, right_base])
            tract_base_height = np.mean([left_height, right_height])
      
            for p in t:
                peak_idx = np.where(labels==p)[0]
                base = (left_height, right_height)
                
                if not bi_directional:
                    ((x1, y1), (x2, y2)) = _peak_base_pts(trace, peak_idx)
                    old_base = (y1, y2)
                    if np.mean(old_base) < tract_base_height:
                        base = old_base

                prominences.append(_peak_prominence(trace, peak_idx, peak_base=base))
        else:
            peak_idx = np.where(labels==t[0])[0]
            prominences.append(_peak_prominence(trace, peak_idx))
    
    return prominences

def _peak_asymmetry(trace, peak_idx, amp_idx=None):
    '''
    Returns number in [0, 1] that is how far from the left edge the amplitude is located
    0 at left edge, 0.5 in middle, 1 at right edge
    '''
    if amp_idx is None:
        amp_idx, _ = _peak_amplitude(trace, peak_idx)
        
    insert = np.searchsorted(peak_idx, amp_idx)
    return insert / len(peak_idx)

def _peak_amplitude(trace, peak_idx):
    '''
    first attempt, simply returns np max in the range of the peak
    I probably should later include something that will check to make sure its not an extraneous point
    This already returns x, y - and I believe it should be a tuple
    '''
    mask = np.zeros_like(trace, dtype=bool)
    mask[peak_idx] = True
    pv = np.where(mask==True, trace, 0)
 
    return np.nanargmax(pv), pv[np.nanargmax(pv)]

def _peak_prominence(trace, peak_idx, peak_base=None, peak_amp=None):
    '''
    This does not return prominence as defined in topology
    Instead it returns prominence as the difference between the peak amplitude and the peak base
    '''

    if peak_base is None:
        left_base, right_base, theta = _peak_base(trace, peak_idx)
        peak_base = [left_base, right_base]
        
    base_height = np.mean(peak_base)
        
    if peak_amp is None:
        amp_idx, amp = _peak_amplitude(trace, peak_idx)

    return amp - base_height, base_height

def _peak_width(trace, peak_idx, height=0.5, prominence=None, true_height=None):
    '''
    should return difference in peak_idx at height (currently just returns indices)
    needs to have added support for multiple crossings?
    '''
    mask = np.zeros_like(trace, dtype=bool)
    mask[peak_idx] = True
    pv = np.where(mask==True, trace, 0)
    
    amp_idx, amp = _peak_amplitude(trace, peak_idx)
    
    if prominence is None:
        prominence, base = _peak_prominence(trace, peak_idx)
    
    if true_height is None:
        target_height = (prominence * height) + base
    else:
        target_height = true_height
    
    i = amp_idx
    left_crosses = [] # could be used to find all widths
    while i < peak_idx[-1] and target_height < pv[i]:
        i -= 1
        left_cross = i
        if pv[i] < target_height:
            left_cross += (target_height - pv[i]) / (pv[i+1] - pv[i])
        
    i = amp_idx
    right_crosses = [] # could be used to find all widths
    while i < peak_idx[-1] and target_height < pv[i]:
        i += 1
        right_cross = i
        if pv[i] < target_height:
            right_cross -= (target_height - pv[i]) / (pv[i-1] - pv[i])   
            
    return left_cross, right_cross, target_height