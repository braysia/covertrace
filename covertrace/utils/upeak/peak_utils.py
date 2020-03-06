import numpy as np
from skimage.morphology import watershed
from collections import OrderedDict
import data_utils as du
from filter_utils import clean_peaks
from data_processing import nan_helper_2d

class Peaks(OrderedDict):
    
    def add_site(self, name, traces, peak_labels, seed_labels):
        '''
        Adds site to Peaks class
        
        keyword arguments:
        name -- name of site (if another site of that name exists, it will be overwritten)
        traces -- array containing the trace data that will be used when extracting peak info
        peak_labels -- labels of the peaks
        seed_labels -- labels of the seeds used for watershed peak finding (peak plateaus)
        '''
        
        self[name] = peak_site(traces, peak_labels, seed_labels)

    def filter_peaks(self, length_thres=None, assym_thres=None, linear_thres=None, amplitude_thres=None, prominence_thres=None):
        '''
        Will filter peaks based on the parameters provided. If a parameter is None, that cleaning function will not be called.
        If you want to use default values for a cleaning function, set a parameter to True.

        # TODO: This should be changed to pass the site to the filter function, and use the prominence values provided
        '''
        for key in self.keys():
            clean_labels, clean_seeds = clean_peaks(self[key].traces, self[key].peak_labels, self[key].seed_labels,
                length_thres=length_thres, assym_thres=assym_thres, linear_thres=linear_thres, amplitude_thres=amplitude_thres, prominence_thres=prominence_thres)

            self[key] = peak_site(self[key].traces, clean_labels, clean_seeds)

    def interp_nans(self):
        '''
        Use linear interpolation to fill in nans in the trace data
        '''
        for key in self.keys():
            self[key]._interp_nans()

    def traces(self):
        return [self[key].traces for key in self.keys()]

    def normalize_traces(self, method='base'):
        '''
        keyword args:
        method - currently only option is base. Calculates linear base of each trace and normalizes values to the mean of the baseline

        TODO: add more normalization options
        '''
        for key in self.keys():
            self[key]._normalize_traces(method=method)
        
    def amplitude(self):
        for key in self.keys():
            self[key]._get_amplitude()
        return [self[key].amplitude for key in self.keys()]
            
    def base(self, adjust_edge=True, dist=4):
        '''
        adjust_edge will make the base flat for peaks that extend to the edge of the trace.
        dist is the distance from the edge that this adjustment will apply
        '''
        for key in self.keys():
            self[key]._get_base(adjust_edge=adjust_edge, dist=dist)
        return [self[key].base for key in self.keys()]

    def asymmetry(self, method='plateau'):
        '''
        plateau method: finds middle of peak plateau and measures asymmetry from there
        amplitude method: finds point of highest amplitude and measures asymmetry from there
        '''
        for key in self.keys():
            self[key]._get_asymmetry(method=method)
        return [self[key].asymmetry for key in self.keys()]

    def base_pts(self, adjust_edge=True, dist=4):
        '''
        adjust_edge will make the base flat for peaks that extend to the edge of the trace.
        dist is the distance from the edge that this adjustment will apply
        '''
        for key in self.keys():
            self[key]._get_peak_base_pts(adjust_edge=adjust_edge, dist=dist)
        return [self[key].base_pts for key in self.keys()]

    def prominence(self, adjust_tracts=True, bi_directional=False, max_gap=12):
        '''
        prominence is defined in this case as the height of the peak above the base of the peak
        if adjust_tracts is true, the base will be defined as the base of a tract of peaks
        if bi_directional is true, the base can be made higher or lower based on tracts, if false, only can be lower
        max_gap is the distance between peaks used for detecting tracts
        '''
        for key in self.keys():
            self[key]._get_prominence(adjust_tracts=adjust_tracts, bi_directional=bi_directional, max_gap=max_gap)
        return [self[key].prominence for key in self.keys()]

    def peak_area_under_curve(self):
        for key in self.keys():
            self[key]._get_auc(area='peak')
        return [self[key].peak_auc for key in self.keys()]

    def total_area_under_curve(self):
        for key in self.keys():
            self[key]._get_auc(area='total')
        return [self[key].total_auc for key in self.keys()]

    def width(self, rel_height=0.5, abs_height=None, estimate='linear', return_widest=True):
        '''
        rel_height is the height relative to the PROMINENCE of the peak at which to measure width
        if abs_height is a value, it will take precedence over rel_height. Finds width at that value
        estimate can be linear or base. if the height of the peak at which you want the width is below the base of the peak
            then it can estimate by fitting the slopes of the peak to a line, or by using just the width of the base
        if return_widest is true it will return the widest found width, otherwise returns the narrowest
        '''
        for key in self.keys():
            self[key]._get_width(rel_height=rel_height, abs_height=abs_height, estimate=estimate, return_widest=return_widest)

        if abs_height is None:
            attr = 'width_rel_{:.2f}'.format(rel_height).replace('.', '_')
        else:
            attr = 'width_abs_{:.2f}'.format(abs_height).replace('.', '_')

        return [getattr(self[key], attr) for key in self.keys()]

    def cross_pts(self, rel_height=0.5, abs_height=None, estimate='linear', return_widest=True):
        '''
        rel_height is the height relative to the PROMINENCE of the peak at which to measure width
        if abs_height is a value, it will take precedence over rel_height. Finds width at that value
        estimate can be linear or base. if the height of the peak at which you want the width is below the base of the peak
            then it can estimate by fitting the slopes of the peak to a line, or by using just the width of the base
        if return_widest is true it will return the widest found width, otherwise returns the narrowest
        '''
        for key in self.keys():
            self[key]._get_cross_pts(rel_height=rel_height, abs_height=abs_height, estimate=estimate, return_widest=return_widest)

        if abs_height is None:
            attr = 'cross_rel_{:.2f}'.format(rel_height).replace('.', '_')
        else:
            attr = 'cross_abs_{:.2f}'.format(abs_height).replace('.', '_')

        return [getattr(self[key], attr) for key in self.keys()]

    def plateau_width(self):
        for key in self.keys():
            self[key]._get_plateau_width()
        return [self[key].plateau_width for key in self.keys()]

    def slope_pts(self):
        for key in self.keys():
            self[key]._get_slope_pts()
        return [self[key].slope_pts for key in self.keys()]

    def tracts(self, max_gap=12):
        for key in self.keys():
            self[key]._get_tracts()
        return [self[key].tracts for key in self.keys()]

    def del_attr(self, attr):
        '''
        removes an attribute
        useful if you want to recalculate something
        '''
        for key in self.keys():
            self[key]._clear_attr(attr)

    def add_attr(self, attr_name, attr_value):
        '''
        can be used to save arbitrary info in the Peaks class
        for example, to save an attribute before deleting it
        '''
        for key in self.keys()
            setattr(self[key], attr_name, attr_value)

    def get_attr(self, attr_name):
        '''
        can be used to get data from an attribute that is not listed above
        '''
        return [getattr(self[key], attr_name) for key in self.keys()]
    
    def _set_keys2attr(self):
        for key in self.keys():
            setattr(self, key, self[key])

class peak_site():
    def __init__(self, traces, peak_labels, seed_labels):
        self.traces = np.array(traces)
        self.peak_labels = np.array(peak_labels)
        self.seed_labels = np.array(seed_labels)
        self._peak_idxs = _labels_to_peak_idxs(self.peak_labels)
        self._plateau_idxs = _labels_to_peak_idxs(self.seed_labels)
        self._peak_masks = _labels_to_mask(self.peak_labels)
        self._plateau_masks = _labels_to_mask(self.seed_labels)

    def _clear_attr(self, attr):
        if hasattr(self, attr):
            delattr(self, attr)
        else:
            print('Attribute {0} not found'.format(attr))
    
    def _get_amplitude(self):
        
        if not hasattr(self, 'amplitude'):
            self.amplitude = []
            for n in range(0, self.traces.shape[0]):
                self.amplitude.append([du._peak_amplitude(self.traces[n], p) for p in self._peak_idxs[n]])
        
        return self.amplitude
    
    def _get_asymmetry(self, method='plateau'):
    
        if not hasattr(self, 'asymmetry'):
            if (not hasattr(self, 'amplitude')) and (method == 'amplitude'):
                self._get_amplitude()
            
            self.asymmetry = []
            for n in range(0, self.traces.shape[0]):
                if method == 'plateau':
                    self.asymmetry.append([du._peak_asymmetry(self.traces[n], p, a[0]) for (p, a) in zip(self._peak_idxs[n], self.amplitude[n])])
                elif method == 'amplitude':
                    self.asymmetry.append([du._peak_asymmetry_by_plateau(self.traces[n], p, pl) for (p, pl) in zip(self._peak_idxs[n], self._plateau_idxs)])
            
        return self.asymmetry

    def _get_auc(self, area='total'):

        if area == 'peak':
            if not hasattr(self, 'peak_auc'):
                self.peak_auc = []
                for n in range(0, self.traces.shape[0]):
                    self.peak_auc.append([du._area_under_curve(self.traces[n], pi) for pi in self._peak_idxs[n]])

            return self.peak_auc

        elif area == 'total':
            if not hasattr(self, 'total_auc'):
                self.total_auc = []
                for n in range(0, self.traces.shape[0]):
                    self.total_auc.append(du._area_under_curve(self.traces[n], range(0, self.traces.shape[1])))

            return self.total_auc
    
    def _get_peak_base_pts(self, adjust_edge=True, dist=4):
        
        if not hasattr(self, 'base_pts'):
            self.base_pts = []
            for n in range(0, self.traces.shape[0]):
                self.base_pts.append([du._peak_base_pts(self.traces[n], p, adjust_edge, dist) for p in self._peak_idxs[n]])
        
        return self.base_pts
    
    def _get_base(self, adjust_edge=True, dist=4):
        
        if not hasattr(self, 'base'):
            if not hasattr(self, 'base_pts'):
                self.base_pts = self._get_peak_base_pts(adjust_edge=adjust_edge, dist=dist)
                
            self.base = []
            for n in range(0, self.traces.shape[0]):
                self.base.append([du._peak_base(self.traces[n], p, bp) for (p, bp) in zip(self._peak_idxs[n], self.base_pts[n])])
            
        return self.base

    def _get_tracts(self, max_gap=12):

        if not hasattr(self, 'tracts'):
            self.tracts = []
            for n in range(0, self.traces.shape[0]):
                self.tracts.append(_detect_peak_tracts(self.traces[n], self.peak_labels[n], max_gap=max_gap))

        return self.tracts

    def _get_prominence(self, adjust_tracts=True, bi_directional=False, max_gap=10):
        '''
        bidirectional and max_gap parameter only used if adjust_tracts is True
        '''

        if not hasattr(self, 'prominence'):
            self.prominence = []

            # will be used to calculate prominence below
            if not hasattr(self, 'amplitude'):
                self._get_amplitude()
            if not hasattr(self, 'base'):
                self._get_base()

            if adjust_tracts:
                if not hasattr(self, 'tracts'):
                    self.tracts = []
                    for n in range(0, self.traces.shape[0]):
                        self.tracts.append(_detect_peak_tracts(self.traces[n], self.peak_labels[n]))
                        self.prominence.append(du._tract_adjusted_peak_prominence(self.traces[n], self.peak_labels[n], self.tracts[n],
                            self.base[n], self.amplitude[n], self.base_pts[n], bi_directional))

            else:
                for n in range(0, self.traces.shape[0]):
                    self.prominence.append([du._peak_prominence(self.traces[n], p, b, a) for (p, b, a) in zip(self._peak_idxs[n], self.base[n], self.amplitude[n])])

        return self.prominence

    def _get_width(self, rel_height=0.5, abs_height=None, estimate='linear', return_widest=True):
        '''
        needs to be transferred from the ipynb
        if abs_height is not None, then rel_height is ignored
        '''
        if abs_height is None:
            width_attr = 'width_rel_{:.2f}'.format(rel_height).replace('.','_')
            cross_attr = 'cross_rel_{:.2f}'.format(rel_height).replace('.','_')
        else:
            width_attr = 'width_abs_{:.2f}'.format(abs_height).replace('.','_')
            cross_attr = 'cross_abs_{:.2f}'.format(abs_height).replace('.','_')

        if not hasattr(self, width_attr):
            setattr(self, width_attr, [])
            
            if not hasattr(self, cross_attr):
                setattr(self, cross_attr, self._get_cross_pts(rel_height=rel_height, abs_height=abs_height, estimate=estimate, return_widest=return_widest))
            setattr(self, width_attr, [du._width_at_pts(c) for c in getattr(self, cross_attr)])

        return getattr(self, width_attr)

    def _get_cross_pts(self, rel_height=0.5, abs_height=None, estimate='linear', return_widest=True):

        if abs_height is None:
            width_attr = 'width_rel_{:.2f}'.format(rel_height).replace('.','_')
            cross_attr = 'cross_rel_{:.2f}'.format(rel_height).replace('.','_')
            if not hasattr(self, 'prominence'):
                self._get_prominence()
        else:
            width_attr = 'width_abs_{:.2f}'.format(abs_height).replace('.','_')
            cross_attr = 'cross_abs_{:.2f}'.format(abs_height).replace('.','_')

        if (estimate == 'linear') and (not hasattr(self, 'slope_pts')):
            self._get_slope_pts()
        if (estimate == 'gauss') and (not hasattr(self, 'tracts')):
            self._get_tracts()

        if not hasattr(self, cross_attr):
            setattr(self, cross_attr, [])
            for n in range(0, self.traces.shape[0]):
                curr_state = getattr(self, cross_attr)
                curr_state.append([du._get_crosses_at_height(self.traces[n], pi, rel_height, abs_height, self.tracts[n], estimate, return_widest,
                    a, p, sl) for (pi, a, p, sl) in zip(self._peak_idxs[n], self.amplitude[n], self.prominence[n], self.slope_pts[n])])
                setattr(self, cross_attr, curr_state)

        return getattr(self, cross_attr)

    def _get_plateau_width(self):

        if not hasattr(self, 'plateau_width'):
            self.plateau_width = []
            self.plateau_pts = []
            for n in range(0, self.traces.shape[0]):
                self.plateau_pts.append([du._plateau_pts(self.traces[n], p) for p in self._plateau_idxs[n]])
                self.plateau_width.append([x1 - x0 for ((x0, y0), (x1, y1)) in self.plateau_pts[n]])

        return self.plateau_width

    def _get_slope_pts(self):

        if not hasattr(self, 'slope_pts'):
            self.slope_pts = []
            for n in range(0, self.traces.shape[0]):
                self.slope_pts.append([du._slope_pts(self.traces[n], pe, pl) for (pe, pl) in zip(self._peak_idxs[n], self._plateau_idxs[n])])

        return self.slope_pts

    def _get_gaussians(self):
        # TODO: Need to figure out how to estimate gaussian mixture
        if not hasattr(self, 'gaussians'):
            if not hasattr(self, 'tracts'):
                self._get_tracts()

            self.gaussians = []

        return self.gaussians

    def _interp_nans(self):
        self.traces = nan_helper_2d(self.traces)

    def _normalize_traces(self, method):
        if method == 'base':
            for n in range(0, self.traces.shape[0]):
                self.traces[n] = du.normalize_by_baseline(self.traces[n])
        else:
            raise ValueError('Unknown normalization method {0}'.format(method))

def watershed_peak(traces, slope_prob, plateau_prob, steps=50, min_seed_prob=0.8, min_peak_prob=0.5, min_seed_length=2):
    '''
    Used to segment peaks based on predictions from the CNN model
    traces - trace data for entire site
    slope_prob - probability array of slope from model (normally output feature 1) 
    plateau_prob - proability array of plateau from model (normally output feature 2)
    steps - number of watershed steps to do (more is likely more accurate, but takes longer)
    min_seed_prob - minimum probability of plateau that can be accepted to segment a peak
    min_peak_prob - minimum probability to include a point as a peak
    min_seed_length - if any plateau is shorter than this length the peak will not be segmented
    '''

    labels = np.zeros(traces.shape)
    seed_labels = np.zeros(traces.shape)

    total_prob = slope_prob + plateau_prob

    for n, t in enumerate(traces):
        seed_idxs = _constant_thres_seg(plateau_prob[n], min_length=min_seed_length, min_prob=min_seed_prob)
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

def _labels_to_mask(labels):
    
    if np.ndim(labels) == 2:
        mask_stack = []
        for lab in labels:
            mask_list = []
            for l in np.unique(lab):
                if l > 0:
                    mask_list.append(np.where(lab==l, True, False))
            mask_stack.append(mask_list)
        return mask_stack
    elif np.ndim(labels) == 1:
        mask_list = []
        for l in np.unique(labels):
            if l > 0:
                mask_list.append(np.where(labels==l, True, False))
        return mask_list

def _labels_to_peak_idxs(labels):

    if np.ndim(labels) == 2:
        peak_stack = []
        for lab in labels:
            peak_idxs = []
            for l in np.unique(lab):
                if l > 0:
                    peak_idxs.append(np.where(lab==l)[0])
            peak_stack.append(peak_idxs)
        return peak_stack
    elif np.ndim(labels) == 1:
        peak_idxs = []
        for l in np.unique(labels):
                if l > 0:
                    peak_idxs.append(np.where(labels==l)[0])    
        return peak_idxs

def _edge_pts_to_idxs(edge_pts):
    e0, e1 = edge_pts
    return np.arange(e0[0], e1[0], 1)