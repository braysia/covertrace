'''Input arr is 2D. Make sure output is a slice of original array.

Output is a boolean array.
With @modify_prop decorator, it will turn prop into 1 if True.

'''
from __future__ import division
import numpy as np
from functools import partial
from utils.array_handling import extend_true, skip_outside_frame_start_to_end
import pandas as pd
from collections import OrderedDict


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
def filter_frames_by_range(arr, LOWER=-10000, UPPER=np.Inf, FRAME_START=0, FRAME_END=None):
    """Replace values with NaN if it's not in a range specified by LOWER and UPPER.
    FRAME_START and FRAME_END will determine which frames to look at.

    Examples:
        >>> arr = np.array([[0, 0, 0], [0, 100, 0]], np.float32)
        >>> filter_frames_by_range(arr, UPPER=1)
        array([[False, False, False],
               [False,  True, False]], dtype=bool)
        >>> arr1 = np.array([[0, 0, np.nan], [0, np.nan, 0]], np.float32)
        >>> filter_frames_by_range(arr1, UPPER=1)
        array([[False, False, False],
               [False, False, False]], dtype=bool)

    """
    arr_bool = (arr < UPPER) * (arr > LOWER)
    arr_bool[:, :FRAME_START] = True
    if isinstance(FRAME_END, int):
        arr_bool[:, FRAME_END:] = True
    arr_bool[np.isnan(arr)] = True  # ignore nan
    return ~arr_bool


@modify_prop
def cut_short_traces(arr, MINFRAME=5, FRAME_START=0, FRAME_END=None):
    """
    MINFRAME is a number of non NaN frames needed.
    Examples:
        >>> arr1 = np.array([[0, 0, np.nan], [0, 0, 0]], np.float32)
        >>> cut_short_traces(arr1, MINFRAME=3)
        array([[ True,  True,  True],
               [False, False, False]], dtype=bool)
    """

    arr_bool = np.zeros(arr.shape, np.bool)

    arr = arr[:, FRAME_START:FRAME_END]

    short_idx = (~np.isnan(arr)).sum(axis=1) < MINFRAME
    arr_bool[short_idx, :] = True
    return arr_bool


@modify_prop
def filter_frames_by_stats(arr, func=np.nanmean, LOWER=-np.Inf, UPPER=np.Inf, FRAME_START=0, FRAME_END=None):
    """
    Calculate statistics for each cells and replace values to NaN if it's not in a range.

    Examples:

        >>> arr = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 1]])
        >>> filter_frames_by_stats(arr, func=np.nanmax, UPPER=2)
        array([[False, False, False],
               [False, False, False],
               [ True,  True,  True]], dtype=bool)
    """
    vec_stats = func(arr, axis=1)
    vec_bool = (vec_stats < UPPER) * (vec_stats > LOWER)
    arr_bool = np.column_stack([vec_bool for i in range(arr.shape[1])])
    arr_bool[:, :FRAME_START] = True
    if isinstance(FRAME_END, int):
        arr_bool[:, FRAME_END:] = True
    arr_bool[np.isnan(arr)] = True  # ignore nan
    return ~arr_bool


@modify_prop
def filter_frames_by_percentile_stats(arr, func=np.nanmean, LOWER=0, UPPER=100, FRAME_START=0, FRAME_END=None):
    """
    Calculate statistics for each cells and replace values to NaN if
    values are not in a percentile indicated.

    Examples:

        >>> arr = np.reshape(np.arange(0, 10), (5, 2))
        >>> filter_frames_by_percentile_stats(arr, func=np.nanmean, LOWER=20, UPPER=80)
        array([[ True,  True],
               [False, False],
               [False, False],
               [False, False],
               [ True,  True]], dtype=bool)
    """
    vec_stats = func(np.array(arr), axis=1)
    LOWP = np.nanpercentile(vec_stats, LOWER)
    HIGHP = np.nanpercentile(vec_stats, UPPER)
    vec_bool = (vec_stats <= HIGHP) * (vec_stats >= LOWP)
    arr_bool = np.column_stack([vec_bool for i in range(arr.shape[1])])
    arr_bool[:, :FRAME_START] = True
    if isinstance(FRAME_END, int):
        arr_bool[:, FRAME_END:] = True
    arr_bool[np.isnan(arr)] = True  # ignore nan
    return ~arr_bool


@modify_prop
def filter_frames_by_diff(arr, pd_func_name='diff', PERIOD=1, THRES=0.1, FRAME_START=0,
                          FRAME_END=None, absolute=True, LEFT=0, RIGHT=0):
    """Outlier detection by diff or pct_change.
    Replace values with NaN based on diff or pct_change. (may choose eitherfor pd_func_name.)
    FRAME_START and FRAME_END will determine which frames to filter.
    LEFT and RIGHT will extend NaN from the outliers found.
    e.g. For cell death, RIGHT can be a large number so that you can filter out all values after
    a sharp spike.
    Use slider_filter_frames_by_diff to play with parameters.

    Examples:

        >>> arr = np.array([[0, 10, 0], [0, 0, 100]], np.float32)
        >>> filter_frames_by_diff(arr, THRES=15)
        array([[False, False, False],
               [False, False,  True]], dtype=bool)
        >>> arr = np.array([[0, 10, 0], [0, -100, 0]], np.float32)
        >>> filter_frames_by_diff(arr, THRES=15, absolute=True)
        array([[False, False, False],
               [ True,  True,  True]], dtype=bool)
    """
    tarr = np.concatenate((np.expand_dims(arr[:, 1:2], axis=1), arr), axis=1)  # pad='wrap'
    func = getattr(pd.DataFrame, pd_func_name)

    if not absolute:
        above_thres = func(pd.DataFrame(tarr), periods=PERIOD, axis=1).values > THRES
    elif absolute:
        above_thres = np.abs(func(pd.DataFrame(tarr), periods=PERIOD, axis=1).values) > THRES
    above_thres = above_thres[:, 1:]
    fn = partial(extend_true, LEFT=LEFT, RIGHT=RIGHT)
    above_thres = np.apply_along_axis(fn, axis=1, arr=above_thres)

    above_thres = skip_outside_frame_start_to_end(above_thres, FRAME_START, FRAME_END)
    return above_thres


@modify_prop
def filter_from_last_frames(arr, FRAME_START=0, FRAME_END=None, LEFT=0):
    """Find NaNs and propagate NaNs to previous frames.
    LEFT is how many frames you want to go back.

    Examples:
        >>> arr = np.array([[0, np.nan, np.nan], [0, 0, np.nan]], np.float32)
        >>> filter_from_last_frames(arr, LEFT=1)
        array([[ True,  True, False],
               [False,  True,  True]], dtype=bool)
    """
    df = pd.DataFrame(arr)
    nan_appeared = df.isnull().diff(axis=1).values
    nan_appeared = skip_outside_frame_start_to_end(nan_appeared, FRAME_START, FRAME_END) == 1
    fn = partial(extend_true, LEFT=LEFT, RIGHT=0)
    nan_appeared = np.apply_along_axis(fn, axis=1, arr=nan_appeared)
    return nan_appeared


@modify_prop
def calc_rolling_func_filter(arr, func_name='rolling_mean', window=3, threshold=0.1):
    """Calculate the differences from pandas rolling statistics and remove something above thres.
    Can use rolling_median/rolling_mean/rolling_sum...

    Examples:
        >>> arr = np.array([[10, 0, 0], [0, 0, 10]], np.float32)
        >>> calc_rolling_func_filter(arr, threshold=4)
        array([[ True, False, False],
               [False, False,  True]], dtype=bool)
    """
    # tarr = np.concatenate((arr[:, 1:2], arr), axis=1)  # pad='wrap'
    dataframe = pd.DataFrame(arr.T)
    func = getattr(pd, func_name)
    rm_dataframe = func(dataframe, window=window, center=True, min_periods=0)
    difference = rm_dataframe
    difference = np.abs(rm_dataframe - dataframe)
    diffarr = difference.values.T
    bool_arr = diffarr > threshold
    return bool_arr
