
import numpy as np
import matplotlib.pyplot as plt


class CanvasMatplot(object):
    pad = 2.0
    w_pad = 0.5
    h_pad = 2.0
    figsize = (15, 5)
    num_row = 1
    sharey = True

    def __init__(self):
        pass

    def make_axes(self, len_sites):
        nrow, ncol = self.num_row, int(np.ceil(len_sites/self.num_row))
        fig, axes = plt.subplots(nrow, ncol, figsize=self.figsize, sharey=self.sharey)
        plt.tight_layout(pad=self.pad, w_pad=self.w_pad, h_pad=self.h_pad)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes, ]
        return fig, axes

canvas = CanvasMatplot()
