import numpy as np
import math
from peakutils import baseline
from scipy.integrate import simps
from data_processing import nan_helper

def normalize_by_baseline(trace, deg=1):
    '''
    should estimate baseline of degree deg
    Normalize trace by the mean of that baseline
    Returns fold activation over baseline
    '''
    if np.isnan(trace).any():
        nans, z = nan_helper(trace)
        trace[nans] = np.interp(z(nans), z(~nans), trace[~nans])
        
    base = baseline(trace, deg=deg)
    return trace / np.mean(base)

def _detect_peak_tracts(trace, labels, max_gap=12):
    '''
    this would be faster if I re-wrote it to use peak_idx instead of labels
    should return peak values in tract
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

def _peak_base_pts(trace, peak_idx, adjust_edge=True, dist=4):
    '''
    returns two points that define the base of the peak
    '''
    base_pts = (peak_idx[0], trace[peak_idx[0]]), (peak_idx[-1], trace[peak_idx[-1]])
    if adjust_edge:
        base_pts = _adjust_edge_base_height(trace, base_pts, dist=dist) #adjust base height for edges of traces
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
    returns the line from left to right always
    '''
    left_slope = (peak_idx[0], trace[peak_idx[0]]), (plateau_idx[0], trace[plateau_idx[0]])
    right_slope = (plateau_idx[-1], trace[plateau_idx[-1]]), (peak_idx[-1], trace[peak_idx[-1]])
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

def _tract_adjusted_peak_prominence(trace, labels, tracts, peak_bases=None, peak_amplitudes=None, peak_base_pts=None, bi_directional=False):
    '''
    if bidirectional is True, base can be raised or lowered
    if bidirectional is False, base can only be lowered. i.e. use the original peak base if it is lower
    returns list of prominences of peak in function
    if peak characteristics are not provided, default values will be used to calculate.
    '''

    prominences = []
    flat_tracts = [peak for tract in tracts for peak in tract]
    for n, t in enumerate(tracts):
        if len(t) > 1:
            if peak_bases is not None:
                left_peak = flat_tracts.index(t[0])
                right_peak = flat_tracts.index(t[-1])
                left_height = peak_bases[left_peak][0]
                right_height = peak_bases[right_peak][1]
            else:
                left_base = np.where(labels==t[0])[0][0]
                right_base = np.where(labels==t[-1])[0][-1]
                
                left_height, right_height, _ = _peak_base(trace, [left_base, right_base])
            
            tract_base_height = np.mean([left_height, right_height])
            base = (left_height, right_height)
      
            for p in t:
                peak_num = flat_tracts.index(p)
                peak_idx = np.where(labels==p)[0]
                
                if peak_amplitudes is not None:
                    peak_amp = peak_amplitudes[peak_num]
                else:
                    peak_amp = _peak_amplitude(trace, peak_idx)
                
                if not bi_directional:
                    if peak_base_pts is not None:
                        ((x1, y1), (x2, y2)) = peak_base_pts[peak_num]
                    else:
                        ((x1, y1), (x2, y2)) = _peak_base_pts(trace, peak_idx)
                    
                    old_base = (y1, y2)
                    if np.mean(old_base) < tract_base_height:
                        base = old_base


                prominences.append(_peak_prominence(trace, peak_idx, peak_base=base, peak_amp=peak_amp))
        else:
            peak_idx = np.where(labels==t[0])[0]
            i = flat_tracts.index(t[0])

            if peak_bases is not None:
                base = peak_bases[i]
            else:
                base = _peak_base(trace, peak_idx)

            if peak_amplitudes is not None:
                peak_amp = peak_amplitudes[i]
            else:
                peak_amp = _peak_amplitude(trace, peak_idx)

            prominences.append(_peak_prominence(trace, peak_idx, peak_base=base, peak_amp=peak_amp))
    
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

def _peak_asymmetry_by_plateau(trace, peak_idx, plateau_idx):
    '''
    measures asymmetry from the center of the plateau
    '''

    plateau_left, plateau_right = _plateau_pts(trace, plateau_idx)
    plateau_mid = (plateau_left[0] + plateau_right[0]) / 2

    insert = np.searchsorted(peak_idx, plateau_mid)
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

def _area_under_curve(trace, idx):
    '''
    returns area under curve for the indices provided
    '''
    return simps(trace[idx])

def _peak_prominence(trace, peak_idx, peak_base=None, peak_amp=None):
    '''
    This does not return prominence as defined in topology
    Instead it returns prominence as the difference between the peak amplitude and the peak base
    '''

    if peak_base is None:
        left_base, right_base, theta = _peak_base(trace, peak_idx)
        peak_base = [left_base, right_base]
    else:
        peak_base = [peak_base[0], peak_base[1]]
        
    base_height = np.mean(peak_base)
        
    if peak_amp is None:
        amp_idx, amp = _peak_amplitude(trace, peak_idx)
    else:
        amp_idx, amp = peak_amp

    return amp - base_height, base_height

def _time_above_thres(trace, peak_idx, rel_thres=0.5, prominence=None, abs_thres=None):
    '''
    returns number of data points that trace was above thres
    '''
    
    mask = np.zeros_like(trace, dtype=bool)
    mask[peak_idx] = True
    pv = np.where(mask==True, trace, 0)

    if prominence is None:
        prominence, base = _peak_prominence(trace, peak_idx)
    else:
        prominence, base = prominence

    if abs_thres is None:
        target_thres = (prominence * rel_height) + base
    else:
        target_thres = abs_thres

    return np.where(pv>=target_thres)[0].shape[0]

def _get_crosses_at_height(trace, peak_idx, rel_height=0.5, abs_height=None, 
        tracts=None, estimate='linear', return_widest=True, amplitude=None, prominence=None, slope_pts=None):
    '''
    any nans present in trace are linearly interpolated to avoid nans in final results
    '''

    mask = np.zeros_like(trace, dtype=bool)
    mask[peak_idx] = True
    pv = np.where(mask==True, trace, 0)

    if np.isnan(trace).any():
        nans, z = nan_helper(trace)
        trace[nans] = np.interp(z(nans), z(~nans), trace[~nans])

    if (estimate == 'linear') and (slope_pts is None):
        raise ValueError('Slope pts must be provided if using linear estimation of peak width')
    elif (estimate == 'gauss') and (tracts is None):
        raise ValueError('Tract information of peaks must be provided if using gaussian estimation of peak width')

    if amplitude is None:
        amp_idx, amp = _peak_amplitude(trace, peak_idx)
    else:
        amp_idx, amp = amplitude
    
    if abs_height is None:

        if prominence is None:
            prominence, base = _peak_prominence(trace, peak_idx)
        else:
            prominence, base = prominence

        target_height = (prominence * rel_height) + base
    else:
        target_height = abs_height

    if target_height > np.max(pv):
        #height above peak, should return nans so that width becomes nan
        return [(np.nan, np.nan), (np.nan, np.nan)]
    
    all_cross_pts = np.where(np.diff(np.sign(trace - target_height)))[0]
    peak_cross_pts = np.array([a for a in all_cross_pts if a in peak_idx])

    #crosses is pt before the crossing on both the way up and the way down
    #get directionality of cross and interpolate the point
    direction = np.array([np.sign(target_height - trace[c]) for c in peak_cross_pts])
    crosses = [c + (abs(target_height - trace[c]) / (trace[c+1] - trace[c])) for c in peak_cross_pts]

    try:
        first_up_cross = np.where(direction>=0)[0][0]
        last_down_cross = np.where(direction<0)[0][-1]
        first_down_cross = np.where(direction<0)[0][0]

        if return_widest:
            crosses = (crosses[first_up_cross], crosses[last_down_cross])
        else:
            crosses = (crosses[first_up_cross], crosses[first_down_cross])

    except IndexError: # really dirty - I should figure out exactly which exceptions might get called 
        # this means that it couldn't find one up and one down cross for each
        # the crosses will have to be recalculated. 
        if estimate == 'gauss':
            # TODO FINISH GAUSS
            # Need to find a way to fit a gaussian mixture
            pass
        elif estimate == 'linear':
            left_slope_pts = slope_pts[0]
            right_slope_pts = slope_pts[1]

            if len(np.where(direction>=0)[0]) > 0: #has an up cross, needs down (right) cross
                left_cross = crosses[np.where(direction>=0)[0][0]]
                right_cross = _linear_interp_x(right_slope_pts[0], right_slope_pts[1], target_height)

            elif len(np.where(direction<0)[0]) > 0: # has a down cross, needs an up (left) cross
                left_cross = _linear_interp_x(left_slope_pts[0], left_slope_pts[1], target_height)

                if return_widest:
                    right_cross = crosses[np.where(direction<0)[0][-1]]
                else:
                    right_cross = crosses[np.where(direction<0)[0][0]]

            else: # has no crosses, estimate both
                left_cross = _linear_interp_x(left_slope_pts[0], left_slope_pts[1], target_height)
                right_cross = _linear_interp_x(right_slope_pts[0], right_slope_pts[1], target_height)

            crosses = (left_cross, right_cross)

        elif estimate == 'base':
            crosses = (peak_idx[0], peak_idx[-1])
        elif (estimate == None) or (estimate == 'None'):
            # values are below peak, and user decides not to estimate. Width is nan
            return [(np.nan, np.nan), (np.nan, np.nan)]
        else:
            raise ValueError('Estimate must be one of gauss, linear, or base.')

    if len(list(zip(crosses, [target_height for c in crosses]))) >= 1: ## CHECK: should this len == 1 always?
        return list(zip(crosses, [target_height for c in crosses]))

def _width_at_pts(test_pts):
    return [t[-1][0] - t[0][0] for t in test_pts]

def _linear_interp_x(pt1, pt2, test_pt):
    '''
    pt1 and pt2 must be tuple of (x, y)
    test_pt is y value where x is to be estimated
    '''
    return pt1[0] + ((test_pt - pt1[1]) * (pt2[0] - pt1[0])) / (pt2[1] - pt1[1])

def _gaussian_fits(trace, labels, tracts, peak_amplitudes=None):
    '''
    TODO: write this function
    '''
    pass