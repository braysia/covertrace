import numpy as np
from os.path import join, basename, dirname
try:
    from labeledarray import LabeledArray
except:
    from covertrace.labeledarray import LabeledArray
from collections import OrderedDict
from covertrace.utils.canvas import canvas
from itertools import izip_longest


class DataArray(LabeledArray):
    def __new__(cls, arr=None, labels=None, idx=None):
        if arr is None:
            return np.asarray(arr).view(cls)
        obj = super(DataArray, cls).__new__(cls, arr, labels, idx)
        obj.prop = np.zeros((arr.shape[-2], arr.shape[-1]), np.uint8)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        if hasattr(obj, 'prop'):
            self.prop = getattr(obj, 'prop', None)
        if hasattr(obj, 'time'):
            self.time = getattr(obj, 'time', None)
        if hasattr(obj, 'condition'):
            self.condition = getattr(obj, 'condition', None)
        super(DataArray, self).__array_finalize__(obj)

    def add_prop(self, label, arr):
        arr = np.expand_dims(arr, axis=0)
        darr = DataArray(np.vstack((self, arr)), np.vstack((self.labels, label)))
        self._set_extra_attr(darr, self)
        return darr

    def hstack(self, larr):
        """merging second dimension (more cells)
        """
        if (self.labels == larr.labels).all():
            darr = DataArray(np.hstack((self, larr)), self.labels)
            self._set_extra_attr(darr, self)
            darr.prop = np.vstack((self.prop, larr.prop))
            return darr

    def _set_extra_attr(self, new_obj, obj):
        extra_fields = set(dir(obj)).difference(set(dir(DataArray)))
        for f in extra_fields:
            if f == 'prop':  # dirty
                continue
            setattr(new_obj, f, getattr(obj, f))

    @classmethod
    def load(cls, file_name):
        if not file_name.endswith('.npz'):
            file_name = file_name + '.npz'
        f = np.load(file_name)
        arr, labels = f['arr'], f['labels']
        la = DataArray(arr, labels)
        for key, value in f.iteritems():
            if not ('arr' == key or 'labels' == key):
                setattr(la, key, value)
        return la


class Sites(OrderedDict):
    def __init__(self, parent_folder, subfolders=None, conditions=[], file_name='df.npz'):
        super(Sites, self).__init__()
        for subf, condition in izip_longest(subfolders, conditions):
            larr = self._read_arr(join(parent_folder, subf, file_name))
            larr.name = subf
            larr.condition = condition
            larr.directory = join(parent_folder, subf)
            self.__setitem__(subf, larr)
        self._set_keys2attr()
        canvas.len_sites = len(subfolders)
        self.canvas = canvas

    def _read_arr(self, path):
        return darray_read(path)

    def _set_keys2attr(self):
        for key in self.iterkeys():
            setattr(self, key, self[key])

    def __getitem__(self, key):
        try:
            return super(Sites, self).__getitem__(key)
        except KeyError:
            """Sites['key1', 'key2']
            """
            store = OrderedDict()
            for name, arr in self.iteritems():
                store[name] = arr[key]
            return store

    def merge_conditions(self):
        """Merge sites if they have the same conditions.
        """
        set_cond = set([i.condition for i in self.itervalues()])
#         print set_cond
        group_by_cond = [[i.name for i in self.itervalues() if i.condition == sc] for sc in set_cond]
        for name_list in group_by_cond:
            if len(name_list) > 1:
                for pos in name_list[1:]:
                    self[name_list[0]] = self[name_list[0]].hstack(self[pos])
                    del self[pos]
                    delattr(self, pos)
        self._set_keys2attr()

    def save(self, file_name='ndf.npz'):
        for key, site in self.iteritems():
            site.save(join(site.directory, file_name))

    def drop_prop(self, pid=1):
        for key, arr in self.iteritems():
            mask = np.max(arr.prop, axis=-1) == pid
            narr = DataArray(arr[:, ~mask, :], arr.labels)
            narr._set_extra_attr(narr, arr)
            self[key] = narr
        self._set_keys2attr()

    def add_median_ratio(self):
        for pos, larr in self.iteritems():
            if 'nuc' in larr.labels[:, 0] and 'cyto' in larr.labels[:, 0]:
                channels = np.unique(larr['cyto'].labels[:, 0]).tolist()
                for ch in channels:
                    label = (['cyto', ch, 'median_ratio'], )
                    arr = larr['cyto', ch, 'median_intensity']/larr['nuc', ch, 'median_intensity']
                    larr = larr.add_prop(label, arr)
                    label = (['nuc', ch, 'median_ratio'], )
                    arr = larr['nuc', ch, 'median_intensity']/larr['cyto', ch, 'median_intensity']
                    larr = larr.add_prop(label, arr)
                self[pos] = larr
        self._set_keys2attr()
        
    def blank_prop(self):
        for key, arr in self.iteritems():
            arr.prop = np.zeros(arr.prop.shape)


def darray_read(path):
    larr = DataArray().load(path)
    larr.directory = dirname(path)
    larr.file_name = basename(path)
    return larr


if __name__ == "__main__":
    pass