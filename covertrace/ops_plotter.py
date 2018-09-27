import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from collections import OrderedDict
from utils.canvas import canvas
from utils.sorting import kshape_clusters


def odict2prop_list(odict):
    store = []
    for key, value in odict.iteritems():
        max_prop = np.max(value.prop, axis=-1)
        uid = np.unique(max_prop)
        for ui in uid:
            store.append([key, ui, value[(max_prop == ui)]])
    return store


def iterate_axes(func):
    def wrapper(arr, **args):
        if isinstance(arr, OrderedDict):
            store = odict2prop_list(arr)
            fig, axes = canvas.make_axes(len(store))
            for ax, (key, pid, value) in zip(axes, store):
                func(value, ax=ax, **args)
                ax.set_title(value.condition + ", pid={0}".format(pid))
        else:
            fig, axes = canvas.make_axes(1)
            func(arr, ax=axes[0], **args)
        return fig, axes
    return wrapper


@iterate_axes
def plot_all(arr, ax=None, **kwargs):
    pd.DataFrame(arr.T, index=arr.time).plot(legend=False, ax=ax, **kwargs)


@iterate_axes
def plot_heatmap(arr, ax=None, **kwargs):
    sns.heatmap(arr, ax=ax, **kwargs)

    
@iterate_axes
def plot_sorted_heatmap(arr, ax=None, num_clusters=3, **kwargs):
    """
    Uses kshape to cluster traces based on general shape
    Removes any traces with nans to ensure better sorting of shape
    """
    no_nan_arr = arr[~np.isnan(arr).any(axis=1)]
    sort_arr = kshape_clusters(no_nan_arr, num_clusters)
    sns.heatmap(sort_arr, ax=ax, **kwargs)
    
@iterate_axes
def plot_tsplot(arr, ax=None, **kwargs):
    """
    Use seaborn tsplot function.
    """
    sns.tsplot(arr, time=arr.time, estimator=np.nanmean, ax=ax, **kwargs)


@iterate_axes
def plot_histogram_pdstats(arr, ax, pd_func_name='mean', **keys):
    func = getattr(pd.DataFrame, pd_func_name)
    df_stats = func(pd.DataFrame(arr))
    sns.distplot(df_stats.dropna(), ax=ax, **keys)
