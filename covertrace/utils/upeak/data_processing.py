import numpy as np
from sklearn.preprocessing import normalize, maxabs_scale, minmax_scale
import scipy.stats as stats

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def nan_helper_2d(arr):
    #probably can be done faster
    temp = np.zeros(arr.shape)
    temp[:] = np.nan
    for n, y in enumerate(arr.copy()):
        nans, z = nan_helper(y)
        y[nans] = np.interp(z(nans), z(~nans), y[~nans])
        temp[n, :] = y 
    return temp

def pad_traces(traces, model_size, pad_mode='edge', cv=0):
    '''
    pad_mode is edge or constant.
    if edge, repeats last value from trace. if constant, pads with cv
    traces are padded at the end. might be good to add functionality to pad at the start of trace
    '''
    options_dict = {'constant_values':cv} if pad_mode == 'constant' else {}
    target_mod = 2 ** model_size
    diff = target_mod - (traces.shape[1] % target_mod)

    if diff == target_mod:
        # no padding needed
        return traces
    else:
        return np.pad(traces, pad_width=((0, 0), (0, diff), (0, 0)), mode=pad_mode, **options_dict)

def stack_sequences(seq_list, cv=np.nan):
    '''
    Input list of 2d arrays. pads ends of traces with nan to same length and stacks traces
    '''
    l_max = np.max([a.shape[1]for a in seq_list])
    seq_list = [np.pad(a, ((0, 0), (0, l_max - a.shape[1])), constant_values=cv) for a in seq_list]
    return np.vstack(seq_list)

def load_data(traces):
    '''
    Loads data. Makes traces correct dimension (3D) and converts labels to categorical
    traces of different lengths are padded.
    '''
    if type(traces) == np.ndarray:
        traces = nan_helper_2d(traces)
    else:
        traces = stack_sequences([nan_helper_2d(np.load(t)) for t in traces], cv=np.nan)
    traces = np.expand_dims(traces, axis=-1)

    return traces

def augment_decorator(func):
    def wrapper(arr, method, **kwargs):
        new_arr = func(arr, **kwargs)

        if method == 'stack':
            return np.vstack([arr, new_arr])
        elif method == 'concatenate':
            return np.concatenate([arr, new_arr], axis=-1)
        elif method == 'inplace':
            return new_arr
        else:
            raise ValueError('Unknown method: {0}'.format(method))
    return wrapper

@augment_decorator
def noise(arr, loc=1, scale=0.05):
    return arr * np.random.normal(loc=loc, scale=scale, size=arr.shape)

@augment_decorator
def amplitude(arr, scale=1000):
    return arr * scale * np.random.rand(arr.shape[0])[:, np.newaxis, np.newaxis]

@augment_decorator
def no_change(arr):
    return arr

@augment_decorator
def normalize_zscore(traces, by_row=True, offset=0, normalize=False):
    '''
    by_row, if true will normalize each trace individually. False will normalize the whole stack together
    offset can be added to prevent negative values
    if normalize is True, the zscore will be normalized to a range of [-1, 1]. May not work if taking in a concatenated array
    '''
    
    def z_func(a, offset=offset):
        return stats.zscore(a) + offset

    if by_row:
        arr = np.apply_along_axis(z_func, 1, traces) # (function, axis, array)
    else:
        arr = z_func(traces)

    if normalize:
        arr = arr[:, :, -1]
        arr = maxabs_scale(arr, axis=1)
        arr = np.expand_dims(arr, axis=-1)
        
    return arr

@augment_decorator
def normalize_amplitude(traces, by_row=True):
    '''
    by_row, if true will normalize each trace individually. False will normalize the whole stack together
    '''
    if by_row:
        row_maxs = np.nanmax(traces, axis=1)
        return traces / row_maxs[:, np.newaxis]
    else:
        return traces / np.nanmax(traces)

@augment_decorator
def normalize_maxabs(traces, feat=1):
    '''
    normalizes on a scale from [-1, 1]
    feature is the index of the feature to use (useful for con)
    '''
    if traces.ndims == 3:
        traces = traces[:, :, feat]
    
    traces = maxabs_scale(traces, axis=1)
    traces = np.expand_dims(traces, axis=-1)

    return traces

def _normalize(funcs, options, method, arr):
    '''
    '''
    assert len(funcs) == len(method), 'Functions and methods must be same length'
    func_dict = {'zscore' : normalize_zscore, 'amplitude' : normalize_amplitude, 'maxabs' : normalize_maxabs}
    to_run = [func_dict[f] for f in funcs]
    
    while len(to_run) > len(options):
        options.append({})
 
    normed = np.copy(arr)

    for t, m, o in zip(to_run, method, options):
        normed = t(normed, m, **o)

    return normed