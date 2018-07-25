import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import numpy as np

# Add visualizing division and unconfident tracking.


class ImageVis(object):
    def __init__(self, images, data, state):
        self.images = images
        self.data = data
        self.state = state

    def mark_prop(self, frame=0, pid=0):
        ch = getattr(self.images, self.state[1])
        ch_img = ch(frame=frame, rgb=True)

        # label_id_arr = self.data.__getitem__((self.state[0], self.state[1], 'label_id'))
        label_id_arr = self.data.__getitem__('cell_id')

        prop_cell = set(label_id_arr[self.data.prop[:, frame] == pid, frame])
        obje = getattr(self.images, self.state[0])
        obj_img = obje(frame=frame)
        for cell in prop_cell:
            bool_img = obj_img == cell
            ch_img[bool_img, 1] = 255
        return ch_img

    def show_single_cell(self, label_id=1, MARGIN=30, frame=0):
        ch = getattr(self.images, self.state[1])
        label_id_arr = self.data.__getitem__('cell_id')
        idx = np.where(label_id_arr == label_id)[0][0]
        x_vec = self.data.__getitem__((self.state[0], self.state[1], 'x'))[idx, :]
        y_vec = self.data.__getitem__((self.state[0], self.state[1], 'y'))[idx, :]
        x, y = x_vec[frame], y_vec[frame]
        ch_img = ch(frame=frame)
        y_ran = slice_adjust_margin(y, ch_img.shape[0], MARGIN)
        x_ran = slice_adjust_margin(x, ch_img.shape[1], MARGIN)
        img = ch_img[y_ran, x_ran]
        return img

    def show_single_cell_movie(self, label_id=1, MARGIN=30, numframes=0):
        images = [show_single_cell(self, label_id, MARGIN, frame) for frame in numframes]
        return images

    def animate_single_cell_movie(self, fig, ax, label_id=1, MARGIN=30, numframes=0):
        frames = self.show_single_cell_movie(self, label_id, MARGIN, numframes)
        def animate(i):
            ax.imshow(frames[i])
        animation = matplotlib.animation.FuncAnimation(fig, animate, frames=numframes)
        return animation


def slice_adjust_margin(x, maximum, MARGIN):
    """Return a slice defined by x-MARGIN and x+MARGIN.
    If x is below 0, it uses 0 instead of x-MARGIN.
    If x is above maximum, it uses maximum instead of x-MARGIN.

    Examples:

        >>> slice_adjust_margin(20, 50, 30)
        slice(0, 50, None)
    """
    x = int(x)
    LOW = x - MARGIN if x-MARGIN >= 0 else 0
    HIGH = x + MARGIN if x+MARGIN < maximum else maximum
    return slice(LOW, HIGH)

