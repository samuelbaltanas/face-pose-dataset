#!/usr/bin/env python3
"""
https://stackoverflow.com/questions/58075822/pyside2-and-matplotlib-how-to-make-matplotlib-run-in-a-separate-process-as-i
https://stackoverflow.com/questions/35527439/pyqt4-wait-in-thread-for-user-input-from-gui/35534047#35534047
https://matplotlib.org/3.1.1/users/event_handling.html
"""
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide2.QtCore import Slot
from PySide2.QtWidgets import QVBoxLayout, QWidget

from face_pose_dataset.model import score

__all__ = ["MatplotlibWidget"]


def heatmap(
    data,
    row_labels,
    col_labels,
    value_range,
    ax=None,
    label_format="%.2f",
    cbar_kw={"fraction": 0.046, "pad": 0.04},
    cbarlabel="",
    **kwargs
):
    """

    Create a heatmap from a numpy array and two lists of labels.

    Heatmap code based on <https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py>

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.

    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    im.set_clim(value_range)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.invert_yaxis()
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.

    ax.set_xticklabels([label_format % col for col in col_labels])
    ax.set_yticklabels(label_format % row for row in row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)

    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def marker():
    # https://stackoverflow.com/questions/14324270/matplotlib-custom-marker-symbol
    star = matplotlib.path.Path.unit_regular_star(6)
    circle = matplotlib.path.Path.unit_circle()
    # concatenate the circle with an internal cutout of the star
    verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
    codes = np.concatenate([circle.codes, star.codes])
    return matplotlib.path.Path(verts, codes)


class MatplotlibWidget(QWidget):
    def __init__(self, storage: score.ScoreModel, parent=None):
        super().__init__(parent)
        # plt.style.use('dark_background')
        fig = figure.Figure(
            figsize=(7, 7), dpi=65, facecolor=(1, 1, 1), edgecolor=(0, 0, 0)
        )

        self.canvas = FigureCanvas(fig)
        # self.toolbar = NavigationToolbar(self.canvas, self)

        lay = QVBoxLayout(self)
        # lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)
        self.setLayout(lay)

        self.ax = fig.add_subplot(111)
        self.hmap = heatmap(
            data=storage.scores,
            row_labels=storage.y_range,
            col_labels=storage.x_range,
            value_range=storage.z_range,
            ax=self.ax,
            cmap="coolwarm",
        )

        self.storage = storage

        self.pointer = self.ax.scatter(
            x=[3.3], y=[3.3], color="black", marker=marker(), s=150
        )

        fig.tight_layout()
        # self.setFixedSize(600, 600)

    @Slot(np.ndarray)
    def update_plot(self, scores: np.ndarray):
        self.hmap.set_data(self.storage.scores)
        self.canvas.draw()

    @Slot(tuple)
    def update_pointer(self, pos: Tuple[float, float]):
        self.pointer.set_offsets(np.array(pos).reshape(-1, 2))
        self.canvas.draw()
