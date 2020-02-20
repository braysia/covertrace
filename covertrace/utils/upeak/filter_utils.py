import numpy as np
from data_utils import _peak_asymmetry, _peak_amplitude, _peak_prominence
import scipy.stats as stats

def clean_peaks(trace, labels, seeds, length_thres=None, assym_thres=None, linear_thres=None, amplitude_thres=None, prominence_thres=None):
    '''
    This should be set up to take in a 2D matrix.
    '''

    #messy af - should think of a better way to do this, maybe a decorator
    args = locals()
    del args['trace']
    del args['labels']
    del args['seeds']
    func_dict = {'length_thres':_filter_peaks_by_length, 'assym_thres':_filter_peaks_by_assymmetry, 'linear_thres':_filter_peaks_by_linreg, 'amplitude_thres':_filter_peaks_by_amplitude, 'prominence_thres': _filter_peaks_by_prominence}
    cleaning_functions = [(func_dict[a], v) for a, v in args.items() if v is not None]
    
    cleaned_labels = labels.copy()
    cleaned_seeds = seeds.copy()
    for n, t in enumerate(traces):
        l = labels[n]

        if np.sum(l) > 0:  
            for func, param in cleaning_functions:
                l = func(t, l, param)

        cleaned_labels[n] = l
        cleaned_seeds[n] = np.where(seeds[n]==l, seeds[n], 0)

    return cleaned_labels, cleaned_seeds

def _filter_peaks_by_length(trace, labels, min_length=8):
    #trace is a dummy variable to keep things consistent
    peaks, counts = np.unique(labels, return_counts=True)
    
    for n, p in enumerate(peaks):
        if (p > 0) and (counts[n] < min_length):
            labels = np.where(labels != p, labels, 0)
    return labels

def _filter_peaks_by_linreg(trace, labels, thres=0.8):
    '''
    will do a linear regression with the points that are part of the peak
    returns labels that passed the test
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

def _filter_peaks_by_assymmetry(trace, labels, thres=0.1):
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
            assym = _peak_asymmetry(trace, peak_idx)
            
            if abs(1-assym) <= thres:
                labels = np.where(labels != p, labels, 0)
    return labels

def _filter_peaks_by_amplitude(trace, labels, thres=1.5):
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

def _filter_peaks_by_prominence(trace, labels, thres=0.5):
    '''
    remove all peaks that have prominence less than thres
    '''
    peaks = np.unique(labels)
    for n, p in enumerate(peaks):
        if p > 0:
            peak_idx = np.where(labels==p)[0]
            amp_location, amp = _peak_amplitude(trace, peak_idx)
            
            if amp < thres:
                labels = np.where(labels != p, labels, 0)
    return labels

def _renumber_labels(labels):
    '''
    not sure that this is needed
    '''
    pass