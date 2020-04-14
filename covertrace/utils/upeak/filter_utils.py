import numpy as np
from data_utils import _peak_asymmetry_by_plateau, _peak_amplitude, _peak_prominence, _detect_peak_tracts, _tract_adjusted_peak_prominence
import scipy.stats as stats

def clean_peaks(traces, labels, seeds, length_thres=None, assym_thres=None, linear_thres=None, amplitude_thres=None, prominence_thres=None):
    '''
    This uses a 2D matrix of traces (cells x timepoints)
    labels and seeds must match dimensions
    '''

    #messy - should think of a better way to do this, maybe a decorator
    args = locals()
    del args['traces']
    del args['labels']
    del args['seeds']
    func_dict = {'length_thres':_filter_peaks_by_length, 'assym_thres':_filter_peaks_by_assymmetry, 'linear_thres':_filter_peaks_by_linreg, 'amplitude_thres':_filter_peaks_by_amplitude, 'prominence_thres': _filter_peaks_by_prominence}
    cleaning_functions = [(func_dict[a], v) for a, v in args.items() if v is not None]
    
    cleaned_labels = labels.copy()
    cleaned_seeds = seeds.copy()
    for n, t in enumerate(traces):
        l = labels[n]
        s = seeds[n]

        if np.sum(l) > 0:  
            for func, param in cleaning_functions:
                if param == True:
                    # uses default specified in the function
                    l = func(t, l, s)
                else:
                    # use the value the user supplied
                    l = func(t, l, s, param)

        cleaned_labels[n] = l
        cleaned_seeds[n] = np.where(seeds[n]==l, seeds[n], 0)

    return cleaned_labels, cleaned_seeds

def _filter_peaks_by_length(trace, labels, seeds, min_length=8):
    #trace is a dummy variable to keep things consistent
    peaks, counts = np.unique(labels, return_counts=True)
    
    for n, p in enumerate(peaks):
        if (p > 0) and (counts[n] < min_length):
            labels = np.where(labels != p, labels, 0)
    return labels

def _filter_peaks_by_linreg(trace, labels, seeds, thres=0.8):
    '''
    will do a linear regression with the points that are part of the peak
    returns labels that have r^2 less than thres
    '''
    peaks = np.unique(labels)
    for n, p in enumerate(peaks):
        if p > 0:
            peak_idxs = np.where(labels==p)[0]
            peak_values = trace[peak_idxs]
            slope, intercept, rvalue, pvalue, st = stats.linregress(peak_idxs, peak_values)
            
            if rvalue ** 2 >= thres:
                labels = np.where(labels != p, labels, 0)
    return labels

def _filter_peaks_by_assymmetry(trace, labels, seeds, thres=0.1):
    '''
    Should remove peaks based on relative position of peaks to base
    Alternatively, could be done with height of amplitdue relative to ending height of peak
    i.e. if the left or right base is to close to amplitude, it's not a complete peak
    thres is how far in fraction of peak length can it be from the amp be from the edge of the peak
    '''
    peaks = np.unique(labels)
    for n, p in enumerate(peaks):
        if p > 0:
            peak_idx = np.where(labels==p)[0]
            plateau_idx = np.where(seeds==p)[0]

            assym = _peak_asymmetry_by_plateau(trace, peak_idx, plateau_idx)

            if abs(1-assym) <= thres:
                labels = np.where(labels != p, labels, 0)
    return labels

def _filter_peaks_by_amplitude(trace, labels, seeds, thres=1.5):
    '''
    remove all peaks where the max value does not reach thres
    '''
    peaks = np.unique(labels)
    for n, p in enumerate(peaks):
        if p > 0:
            peak_idx = np.where(labels==p)[0]
            amp_location, amp = _peak_amplitude(trace, peak_idx)
            
            if amp < thres:
                labels = np.where(labels != p, labels, 0)
    return labels

def _filter_peaks_by_prominence(trace, labels, seeds, thres=0.5):
    '''
    Values are calculated using defaults. It's possible user would have specified different parameters.
    The only way to include those would be to include these functions as part of the Peak class.
    May be something useful to do in future. Would allow filtering after paramaterization.
    '''
    peaks = np.unique(labels)
    tracts = _detect_peak_tracts(trace, labels)
    prominences = _tract_adjusted_peak_prominence(trace, labels, tracts)

    for n, p in enumerate(peaks): #could loop over tracts too.
        if p > 0:
            if prominences[n-1][0] < thres:
                labels = np.where(labels != p, labels, 0)
    return labels

def _filter_peaks_by_height_asymmetry(trace, labels, seeds, thres=0.2):
    '''
    TODO right function that will remove a peak if one end of the base is to close to the amplitude in terms of height
    '''
    pass