from __future__ import division
import numpy as np
from collections import OrderedDict
from covertrace.data_array import Sites, DataArray

def modify_prop(func):
    def wrapper(arr, **args):
        if isinstance(arr, OrderedDict):
            for key, value in arr.iteritems():
                bool_arr = func(value, **args)
                value.prop[bool_arr] = 1
        else:
            bool_arr = func(arr, **args)
            arr.prop[bool_arr] = 1
    return wrapper

@modify_prop
def filter_by_range_single_frame(arr, LOWER=0, UPPER=np.inf):
    mask = (arr > LOWER) * (arr < UPPER)
    return ~mask

@modify_prop
def filter_by_percent_single_frame(arr, LOWER=0, UPPER=100):
    lowp = np.nanpercentile(arr, LOWER)
    highp = np.nanpercentile(arr, UPPER)
    mask = (arr > lowp) * (arr < highp)
    return ~mask

def remove_props_single_frame(site, pid=1):
    #used in conjunction with the drop_prop and blank_prop properties of Sites
    #input for this function is Sites - remove_props_single_frame(all_sites)
    #labels all props, which can then be removed from the Sites object - all_sites.drop_prop()
    #can also be reset in the Sites object - all_sites.blank_prop()
    for key, val in site.iteritems():
        mask = np.max(val.prop, axis=-1) == pid
        nval = np.expand_dims(val[:, ~mask, :], axis=2)
        narr = DataArray(nval, val.labels)
        narr._set_extra_attr(narr, val)
        site[key] = narr
    site._set_keys2attr()