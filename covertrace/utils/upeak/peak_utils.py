import numpy as np
from skimage.segmentation import watershed
from collections import OrderedDict
import data_utils as du
from filter_utils import clean_peaks
from data_processing import nan_helper_2d
from upeak_predict import predict_peaks

class Peaks(OrderedDict):

    def add_site(self, name, traces, predictions=None, weights=None, model=None, normalize=False, steps=25, 
            min_seed_prob=0.8, min_peak_prob=0.5, min_seed_length=2):
        '''
        Run UPeak on traces and save predictions and labels as a new site class.
        Either weights or predictions can be supplied

        #TODO Finish docstring
        #TODO Add type casting for traces
        #NOTE _set_keys2attr() after every cycle is slow, but allows for checking shape during building of peaks class
        
        keyword arguments:
        name -- name of site (if another site of that name exists, it will be overwritten)
        traces -- array containing the trace data that will be used when extracting peak info
        '''

        if predictions is None:
            if weights is None:
                raise ValueError('Predictions or weights must be supplied.')
            else:
                predictions = predict_peaks(traces, model=model, weights=weights, normalize=normalize)
        
        no_nan_traces = nan_helper_2d(traces) #nans in peaks can interfere with segmentation, so linear interpolation
        peak_labels, seed_labels = watershed_peak(no_nan_traces, predictions[:, :, 1], predictions[:, :, 2], steps=steps, min_seed_prob=min_seed_prob,
            min_peak_prob=min_peak_prob, min_seed_length=min_seed_length)

        self[name] = peak_site(traces, peak_labels, seed_labels, predictions)
        self._set_keys2attr()

    def reload_sites(self):
        '''
        Reloads all of the sites in Peaks class. Removes all attributes.
        '''
        for site, data in self.items():
            self[site] = peak_site(data.raw_traces, data.peak_labels, data.seed_labels, data.predictions)
        self._set_keys2attr()

    def add_site_labels(self, name, traces, peak_labels, seed_labels):
        '''
        Depracated in favor of add_site
        Adds site to Peaks class using already generated labels
        
        keyword arguments:
        name -- name of site (if another site of that name exists, it will be overwritten)
        traces -- array containing the trace data that will be used when extracting peak info
        peak_labels -- labels of the peaks
        seed_labels -- labels of the seeds used for watershed peak finding (peak plateaus)
        '''
        
        self[name] = peak_site(traces, peak_labels, seed_labels)
        self._set_keys2attr()

    def filter_peaks(self, length_thres=None, asym_thres=None, linear_thres=None, amp_thres=None, prom_thres=None, height_asym_thres=None):
        '''
        Will filter peaks based on the parameters provided. If a parameter is None, that cleaning function will not be called.
        If you want to use default values for a cleaning function, set a parameter to True.

        # TODO: This should be changed to pass the site to the filter function, and use the prominence values provided
        '''
        for key in self.keys():
            clean_labels, clean_seeds = clean_peaks(self[key].traces, self[key].peak_labels, self[key].seed_labels,
                length_thres=length_thres, asym_thres=asym_thres, linear_thres=linear_thres, amp_thres=amp_thres,
                prom_thres=prom_thres, height_asym_thres=height_asym_thres)

            self[key] = peak_site(self[key].traces, clean_labels, clean_seeds, self[key].predictions)

        self._set_keys2attr()

    def interp_nans(self):
        '''
        Use linear interpolation to fill in nans in the trace data
        '''
        for key in self.keys():
            self[key]._interp_nans()

    def traces(self):
        '''
        Returns value of traces used for calculating all peak_attributes.
        If normalize_traces has been called, these traces are normalized.
        Otherwise. traces are input traces and self.traces = self.raw_traces
        '''
        return [self[key].traces for key in self.keys()]

    def raw_traces(self):
        '''
        Returns original traces that were provided when Peaks class was made.
        '''
        return [self[key].raw_traces for key in self.keys()]

    def predictions(self):
        '''
        Returns predictions generated by UPeak model
        '''
        return [self[key].predictions for key in self.keys()]

    def peak_counts(self):
        '''
        Returns number of detected peaks for each site
        '''
        return [self[key].peak_counts for key in self.keys()]

    def normalize_traces(self, method='base'):
        '''
        Normalizes traces using method provided. 
        Initial traces are saved in self.raw_traces and remain unchanged. Normalized traces are in self.traces
        If nans were present in traces, those are interpolated

        keyword args:
        method - currently only option is base. Calculates linear base of each trace and normalizes values to the mean of the baseline

        TODO: add more normalization options. potentially something like flatten, which subtracts the baseline?
        '''
        for key in self.keys():
            self[key]._normalize_traces(method=method)
        
    def amplitude(self, duplicates=True):
        '''
        Returns: tuple of (x, y) of the highest point in the peak
        '''
        for key in self.keys():
            self[key]._get_amplitude(duplicates=duplicates)
        return [self[key].amplitude for key in self.keys()]
            
    def base(self, adjust_edge=True, dist=4, duplicates=True):
        '''
        Finds base of peak.

        Returns: tuple of (y0, y1, theta), where y0/y1 are the height at the edge of the base, and theta is the angle
        The (x, y) points can be found by calling base_pts

        keyword_arguments:
        adjust_edge will make the base flat for peaks that extend to the edge of the trace.
        dist is the distance from the edge that this adjustment will apply
        '''
        for key in self.keys():
            self[key]._get_base(adjust_edge=adjust_edge, dist=dist, duplicates=duplicates)
        return [self[key].base for key in self.keys()]

    def asymmetry(self, method='plateau', duplicates=True):
        '''
        Rough calculation of assymmetry of peak.

        Returns: value [0, 1] of where the point is on the whole length of the peak. 0.5 is perfectly symmetrical

        keyword arguments:
        plateau method: finds middle of peak plateau and measures asymmetry from there
        amplitude method: finds point of highest amplitude and measures asymmetry from there
        '''
        for key in self.keys():
            self[key]._get_asymmetry(method=method, duplicates=duplicates)
        return [self[key].asymmetry for key in self.keys()]

    def base_pts(self, adjust_edge=True, dist=4, duplicates=True):
        '''
        Returns: [(x1, y1), (x2, y2)] for the two points at the edge of the base of the peak.

        adjust_edge will make the base flat for peaks that extend to the edge of the trace.
        dist is the distance from the edge that this adjustment will apply
        '''
        for key in self.keys():
            self[key]._get_peak_base_pts(adjust_edge=adjust_edge, dist=dist, duplicates=duplicates)
        return [self[key].base_pts for key in self.keys()]

    def prominence(self, adjust_tracts=True, bi_directional=False, max_gap=12, duplicates=True):
        '''
        Returns: tuple of (prominence above base_height, base_height)
        prominence is defined in this case as the height of the peak above the base of the peak

        keyword arguments:
        adjust_tracts: if True, the base will be defined as the base of a tract of peaks
        bi_directional: if True, the base can be made higher or lower based on tracts, if false, only can be lower
        max_gap: the distance between peaks used for detecting tracts. (only used if adjust_tracts is True)
        '''
        for key in self.keys():
            self[key]._get_prominence(adjust_tracts=adjust_tracts, bi_directional=bi_directional, max_gap=max_gap, duplicates=duplicates)
        return [self[key].prominence for key in self.keys()]

    def peak_area_under_curve(self, duplicates=True):
        '''
        Calculate area under the curve for each peak
        '''
        for key in self.keys():
            self[key]._get_auc(area='peak', duplicates=duplicates)
        return [self[key].peak_auc for key in self.keys()]

    def total_area_under_curve(self):
        '''
        Calculate total area under the curve
        '''
        for key in self.keys():
            self[key]._get_auc(area='total')
        return [self[key].total_auc for key in self.keys()]

    def integral(self):
        '''
        Returns integrated activity over time
        '''
        for key in self.keys():
            self[key]._get_integral()
        return [self[key].integral for key in self.keys()]

    def derivative(self):
        '''
        Returns derivative of the trace at each point
        '''
        for key in self.keys():
            self[key]._get_derivative()
        return [self[key].derivative for key in self.keys()]

    def width(self, rel_height=0.5, abs_height=None, estimate='linear', return_widest=True, duplicates=True):
        '''
        rel_height is the height relative to the PROMINENCE of the peak at which to measure width
        if abs_height is a value, it will take precedence over rel_height. Finds width at that value
        estimate can be linear or base. if the height of the peak at which you want the width is below the base of the peak
            then it can estimate by fitting the slopes of the peak to a line, or by using just the width of the base
        if return_widest is true it will return the widest found width, otherwise returns the narrowest
        '''
        for key in self.keys():
            self[key]._get_width(rel_height=rel_height, abs_height=abs_height, estimate=estimate, return_widest=return_widest, duplicates=duplicates)

        if abs_height is None:
            attr = 'width_rel_{:.2f}'.format(rel_height).replace('.', '_')
        else:
            attr = 'width_abs_{:.2f}'.format(abs_height).replace('.', '_')

        return [getattr(self[key], attr) for key in self.keys()]

    def cross_pts(self, rel_height=0.5, abs_height=None, estimate='linear', return_widest=True, duplicates=True):
        '''
        rel_height is the height relative to the PROMINENCE of the peak at which to measure width
        if abs_height is a value, it will take precedence over rel_height. Finds width at that value
        estimate can be linear or base. if the height of the peak at which you want the width is below the base of the peak
            then it can estimate by fitting the slopes of the peak to a line, or by using just the width of the base
        if return_widest is true it will return the widest found width, otherwise returns the narrowest
        '''
        for key in self.keys():
            self[key]._get_cross_pts(rel_height=rel_height, abs_height=abs_height, estimate=estimate, return_widest=return_widest, duplicates=duplicates)

        if abs_height is None:
            attr = 'cross_rel_{:.2f}'.format(rel_height).replace('.', '_')
        else:
            attr = 'cross_abs_{:.2f}'.format(abs_height).replace('.', '_')

        return [getattr(self[key], attr) for key in self.keys()]

    def plateau_width(self, duplicates=True):
        for key in self.keys():
            self[key]._get_plateau_width(duplicates=duplicates)
        return [self[key].plateau_width for key in self.keys()]

    def slope_pts(self, duplicates=True):
        for key in self.keys():
            self[key]._get_slope_pts(duplicates=duplicates)
        return [self[key].slope_pts for key in self.keys()]

    def tracts(self, max_gap=12, duplicates=True):
        for key in self.keys():
            self[key]._get_tracts(duplicates=duplicates)
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
        for key in self.keys():
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
    def __init__(self, traces, peak_labels, seed_labels, predictions=None):
        self.traces = np.array(traces)
        self.raw_traces = np.copy(np.array(traces))
        self.peak_labels = np.array(peak_labels)
        self.seed_labels = np.array(seed_labels)
        self.predictions = np.array(predictions)
        self._peak_idxs = _labels_to_peak_idxs(self.peak_labels)
        self._plateau_idxs = _labels_to_peak_idxs(self.seed_labels)
        self._peak_masks = _labels_to_mask(self.peak_labels)
        self._plateau_masks = _labels_to_mask(self.seed_labels)
        self.peak_counts = np.array([len(p) for p in self._peak_idxs])

        # create de-duplicated peaks as well
        self._dedup_peak_idxs = _labels_to_peak_idxs(self.peak_labels)
        self._dedup_plateau_idxs = _labels_to_peak_idxs(self.seed_labels)
        self._remove_duplicate_peaks()
        self._dedup_peak_labels = _2d_idxs_to_labels(self.traces, self._dedup_peak_idxs)
        self._dedup_plateau_labels = _2d_idxs_to_labels(self.traces, self._dedup_plateau_idxs)

    def _clear_attr(self, attr):
        if hasattr(self, attr):
            delattr(self, attr)
        else:
            print('Attribute {0} not found'.format(attr))

    def _remove_duplicate_peaks(self, thres=0.8):
        '''
        Iterates through traces and removes peaks which appear in >1 trace.
        These indices are saved separately
        '''
        for n in range(self.traces.shape[0]):
            # subtract traces and find 0s, record overlaps
            diffs = np.array([self.traces[n] - self.traces[i] for i in range(self.traces.shape[0])])
            unique_overlaps, unique_counts = np.unique(np.where(diffs==0)[0], return_counts=True)

            # check traces where the overlap is greater than four pts
            potential_matches = [unique_overlaps[i] for i in range(len(unique_overlaps))
                if (unique_counts[i] > 4) and (unique_overlaps[i] != n)]

            #must be faster way to do this
            # not working for multiple peaks, only removes one at a time
            for p in potential_matches:
                # overlapping points with matching trace
                overlaps = np.where(self.traces[n] - self.traces[p] == 0)[0]

                # iterate over copy of peak list and remove matches
                for p_idx, peak1 in enumerate(list(self._dedup_peak_idxs[n])):
                    len_thres = thres * len(peak1)
                    for peak2 in self._dedup_peak_idxs[p]:
                        if len(peak1) == 0 or len(peak2) == 0:
                            pass
                        elif np.in1d(peak1, overlaps).all() and len(np.intersect1d(peak1, peak2)) > len_thres:
                            # remove indices from peaks and plateaus
                            _peak_remover(self._dedup_peak_idxs[n], peak1)
                            _peak_remover(self._dedup_plateau_idxs[n], self._plateau_idxs[n][p_idx])
    
    def _get_amplitude(self, duplicates=True):
        
        if not hasattr(self, 'amplitude'):
            self.amplitude = []
            for n in range(0, self.traces.shape[0]):
                if duplicates:
                    self.amplitude.append([du._peak_amplitude(self.traces[n], p) for p in self._peak_idxs[n]])
                else:
                    self.amplitude.append([du._peak_amplitude(self.traces[n], p) for p in self._dedup_peak_idxs[n]])
        
        return self.amplitude
    
    def _get_asymmetry(self, method='plateau', duplicates=True):
    
        if not hasattr(self, 'asymmetry'):
            if (not hasattr(self, 'amplitude')) and (method == 'amplitude'):
                self._get_amplitude(duplicates=duplicates)
            
            self.asymmetry = []
            for n in range(0, self.traces.shape[0]):
                # this could create problems if not consistent with using duplicates.
                # may need to reset sites to fix.
                if duplicates:
                    inputs = zip(self._peak_idxs[n], self.amplitude[n])
                else:
                    inputs = zip(self._dedup_peak_idxs[n], self.amplitude[n])

                if method == 'plateau':
                    self.asymmetry.append([du._peak_asymmetry(self.traces[n], p, a[0]) for (p, a) in inputs])
                elif method == 'amplitude':
                    self.asymmetry.append([du._peak_asymmetry_by_plateau(self.traces[n], p, pl) for (p, pl) in inputs])
            
        return self.asymmetry

    def _get_auc(self, area='total', duplicates=True):
        if area == 'peak':
            if not hasattr(self, 'peak_auc'):
                self.peak_auc = []
                for n in range(0, self.traces.shape[0]):
                    if duplicates:
                        self.peak_auc.append([du._area_under_curve(self.traces[n], pi) for pi in self._dedup_peak_idxs[n]])
                    else:
                        self.peak_auc.append([du._area_under_curve(self.traces[n], pi) for pi in self._peak_idxs[n]])

            return self.peak_auc

        elif area == 'total':
            if not hasattr(self, 'total_auc'):
                self.total_auc = []
                for n in range(0, self.traces.shape[0]):
                    self.total_auc.append(du._area_under_curve(self.traces[n], range(0, self.traces.shape[1])))

            return self.total_auc
    
    def _get_peak_base_pts(self, adjust_edge=True, dist=4, duplicates=True):
        
        if not hasattr(self, 'base_pts'):
            self.base_pts = []
            for n in range(0, self.traces.shape[0]):
                if duplicates:
                    self.base_pts.append([du._peak_base_pts(self.traces[n], p, adjust_edge, dist) for p in self._peak_idxs[n]])
                else:
                    self.base_pts.append([du._peak_base_pts(self.traces[n], p, adjust_edge, dist) for p in self._dedup_peak_idxs[n]])

        return self.base_pts
    
    def _get_base(self, adjust_edge=True, dist=4, duplicates=True):
        
        if not hasattr(self, 'base'):
            if not hasattr(self, 'base_pts'):
                self.base_pts = self._get_peak_base_pts(adjust_edge=adjust_edge, dist=dist, duplicates=duplicates)
                
            self.base = []
            for n in range(0, self.traces.shape[0]):
                if duplicates:
                    self.base.append([du._peak_base(self.traces[n], p, bp) for (p, bp) in zip(self._peak_idxs[n], self.base_pts[n])])
                else:
                    self.base.append([du._peak_base(self.traces[n], p, bp) for (p, bp) in zip(self._dedup_peak_idxs[n], self.base_pts[n])])
            
        return self.base

    def _get_integral(self):

        if not hasattr(self, 'integral'):
            self.integral = []
            for n in range(self.traces.shape[0]):
                self.integral.append(du._integrated_activity(self.traces[n]))

        return self.integral

    def _get_derivative(self):
        # TODO: add functionality to specificy indices

        if not hasattr(self, 'derivative'):
            self.derivative = []
            for n in range(self.traces.shape[0]):
                self.derivative.append(du._derivative_trace(self.traces[n]))

        return self.derivative

    def _get_tracts(self, max_gap=12, duplicates=True):

        if not hasattr(self, 'tracts'):
            self.tracts = []
            for n in range(0, self.traces.shape[0]):
                if duplicates:
                    self.tracts.append(du._detect_peak_tracts(self.traces[n], self.peak_labels[n], max_gap=max_gap))
                else:
                    self.tracts.append(du._detect_peak_tracts(self.traces[n], self._dedup_peak_labels[n], max_gap=max_gap))

        return self.tracts

    def _get_prominence(self, adjust_tracts=True, bi_directional=False, max_gap=10, duplicates=True):
        '''
        bidirectional and max_gap parameter only used if adjust_tracts is True
        '''

        if not hasattr(self, 'prominence'):
            self.prominence = []

            # will be used to calculate prominence below
            if not hasattr(self, 'amplitude'):
                self._get_amplitude(duplicates=duplicates)
            if not hasattr(self, 'base'):
                self._get_base(duplicates=duplicates)

            if adjust_tracts:
                if not hasattr(self, 'tracts'):
                    self._get_tracts(max_gap=max_gap, duplicates=duplicates)
                for n in range(0, self.traces.shape[0]):
                    if duplicates:
                        self.prominence.append(du._tract_adjusted_peak_prominence(self.traces[n], self.peak_labels[n], self.tracts[n],
                            self.base[n], self.amplitude[n], self.base_pts[n], bi_directional))
                    else:
                        self.prominence.append(du._tract_adjusted_peak_prominence(self.traces[n], self._dedup_peak_labels[n], self.tracts[n],
                            self.base[n], self.amplitude[n], self.base_pts[n], bi_directional))
            else:
                for n in range(0, self.traces.shape[0]):
                    if duplicates:
                        self.prominence.append([du._peak_prominence(self.traces[n], p, b, a) for (p, b, a) in zip(self._peak_idxs[n], self.base[n], self.amplitude[n])])
                    else:
                        self.prominence.append([du._peak_prominence(self.traces[n], p, b, a) for (p, b, a) in zip(self._dedup_peak_idxs[n], self.base[n], self.amplitude[n])])

        return self.prominence

    def _get_width(self, rel_height=0.5, abs_height=None, estimate='linear', return_widest=True, duplicates=True):
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
            #setattr(self, width_attr, [])
            
            if not hasattr(self, cross_attr):
                setattr(self, cross_attr, self._get_cross_pts(rel_height=rel_height, abs_height=abs_height, estimate=estimate, return_widest=return_widest, duplicates=duplicates))

            setattr(self, width_attr, [du._width_at_pts(c) for c in getattr(self, cross_attr)])

        return getattr(self, width_attr)

    def _get_cross_pts(self, rel_height=0.5, abs_height=None, estimate='linear', return_widest=True, duplicates=True):

        if abs_height is None:
            width_attr = 'width_rel_{:.2f}'.format(rel_height).replace('.','_')
            cross_attr = 'cross_rel_{:.2f}'.format(rel_height).replace('.','_')
            if not hasattr(self, 'prominence'):
                self._get_prominence(duplicates=duplicates)
        else:
            width_attr = 'width_abs_{:.2f}'.format(abs_height).replace('.','_')
            cross_attr = 'cross_abs_{:.2f}'.format(abs_height).replace('.','_')

        if (estimate == 'linear') and (not hasattr(self, 'slope_pts')):
            self._get_slope_pts(duplicates=duplicates)
        if (estimate == 'gauss') and (not hasattr(self, 'tracts')):
            self._get_tracts(duplicates=duplicates)

        if not hasattr(self, cross_attr):
            setattr(self, cross_attr, [])
            for n in range(0, self.traces.shape[0]):
                curr_state = getattr(self, cross_attr)
                if duplicates:
                    curr_state.append([du._get_crosses_at_height(self.traces[n], pi, rel_height, abs_height, self.tracts[n], estimate, return_widest,
                        a, p, sl) for (pi, a, p, sl) in zip(self._peak_idxs[n], self.amplitude[n], self.prominence[n], self.slope_pts[n])])
                else:
                    curr_state.append([du._get_crosses_at_height(self.traces[n], pi, rel_height, abs_height, self.tracts[n], estimate, return_widest,
                        a, p, sl) for (pi, a, p, sl) in zip(self._dedup_peak_idxs[n], self.amplitude[n], self.prominence[n], self.slope_pts[n])])

                setattr(self, cross_attr, curr_state)

        return getattr(self, cross_attr)

    def _get_plateau_width(self, duplicates=True):

        if not hasattr(self, 'plateau_width'):
            self.plateau_width = []
            self.plateau_pts = []
            for n in range(0, self.traces.shape[0]):
                if duplicates:
                    self.plateau_pts.append([du._plateau_pts(self.traces[n], p) for p in self._plateau_idxs[n]])
                else:
                    self.plateau_pts.append([du._plateau_pts(self.traces[n], p) for p in self._dedup_plateau_idxs[n]])

                self.plateau_width.append([x1 - x0 for ((x0, y0), (x1, y1)) in self.plateau_pts[n]])

        return self.plateau_width

    def _get_slope_pts(self, duplicates=True):

        if not hasattr(self, 'slope_pts'):
            self.slope_pts = []
            for n in range(0, self.traces.shape[0]):
                if duplicates:
                    self.slope_pts.append([du._slope_pts(self.traces[n], pe, pl) for (pe, pl) in zip(self._peak_idxs[n], self._plateau_idxs[n])])
                else:
                    self.slope_pts.append([du._slope_pts(self.traces[n], pe, pl) for (pe, pl) in zip(self._dedup_peak_idxs[n], self._dedup_plateau_idxs[n])])

        return self.slope_pts

    def _get_gaussians(self):
        '''
        Currently not implemented
        TODO: Need to figure out how to estimate gaussian mixture
        '''

        if not hasattr(self, 'gaussians'):
            if not hasattr(self, 'tracts'):
                self._get_tracts()

            self.gaussians = []

        return self.gaussians

    def _interp_nans(self):
        self.traces = nan_helper_2d(self.traces)

    def _normalize_traces(self, method):

        if method == 'base':
            for n in range(0, self.raw_traces.shape[0]):
                self.traces[n] = du.normalize_by_baseline(self.raw_traces[n])
        else:
            raise ValueError('Unknown normalization method {0}'.format(method))

def watershed_peak(traces, slope_prob, plateau_prob, steps=25, min_seed_prob=0.8, min_peak_prob=0.5, min_seed_length=2):
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
            seeds = watershed(trace, markers=seeds, mask=cand_mask, watershed_line=True, compactness=0)
        
    return seeds

def _idxs_to_labels(trace, seed_idxs):
    labels = np.zeros(trace.shape)
    
    for n, s in enumerate(seed_idxs):
        labels[s] = n + 1
    
    return labels

def _2d_idxs_to_labels(trace, idxs):
    labels = np.zeros(trace.shape)

    for i, idx in enumerate(idxs):
        for j, x in enumerate(idx):
            labels[i, x] = j + 1

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

def _peak_remover(peak_list, removed_peak):
    '''
    list.remove() raises a value error if multiple np arrays are present in the list
    So this is a replacement for that function
    '''
    idx = 0
    list_size = len(peak_list)

    while idx != list_size and not np.array_equal(peak_list[idx], removed_peak):
        idx += 1

    if idx != list_size:
        peak_list.pop(idx)
    else:
        # This should not happen
        raise ValueError('Peak not found.')