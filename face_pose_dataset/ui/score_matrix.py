#!/usr/bin/env python3
"""
https://stackoverflow.com/questions/58075822/pyside2-and-matplotlib-how-to-make-matplotlib-run-in-a-separate-process-as-i
https://stackoverflow.com/questions/35527439/pyqt4-wait-in-thread-for-user-input-from-gui/35534047#35534047
https://matplotlib.org/3.1.1/users/event_handling.html
"""
import sys
from typing import Tuple

import matplotlib
import numpy as np
from matplotlib import figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide2.QtCore import QObject, QTimer, Signal, Slot
from PySide2.QtWidgets import QApplication, QVBoxLayout, QWidget

from face_pose_dataset import pose_storage
from face_pose_dataset.visualization import heatmap


__all__ = ["MatplotlibWidget"]


class DSL(QObject):
    dataChanged = Signal(tuple)

    def __init__(self, storage: np.ndarray, parent=None):
        # LOAD HMI
        super().__init__(parent)

        # Data to be visualized
        self.idx = 0.1
        self.data: np.ndarray = storage

    def mainLoop(self):
        self.data.scores -= self.idx
        # Send data to graph
        self.dataChanged.emit(self.data.scores, (0, 0))
        # LOOP repeater
        QTimer.singleShot(10, self.mainLoop)


class MatplotlibWidget(QWidget):
    def __init__(self, storage: np.ndarray, parent=None):
        super().__init__(parent)
        # plt.style.use('dark_background')
        fig = figure.Figure(
            figsize=(7, 5), dpi=65, facecolor=(1, 1, 1), edgecolor=(0, 0, 0)
        )

        self.canvas = FigureCanvas(fig)
        # self.toolbar = NavigationToolbar(self.canvas, self)

        lay = QVBoxLayout(self)
        # lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)
        self.setLayout(lay)

        self.ax = fig.add_subplot(111)
        self.hmap = heatmap.heatmap(
            storage.scores,
            storage.x_range,
            storage.y_range,
            storage.z_range,
            ax=self.ax,
        )

        self.pointer = self.ax.scatter(x=[5], y=[5], color="r", marker=marker(), s=150)

        fig.tight_layout()
        self.setFixedSize(600, 600)

    @Slot(Tuple[float, float])
    def update_plot(self, pos: Tuple[float, float]):
        # self.hmap.set_data(data)
        self.pointer.set_offsets([[pos[0], pos[1]]])
        self.canvas.draw()




def marker():
    # https://stackoverflow.com/questions/14324270/matplotlib-custom-marker-symbol
    star = matplotlib.path.Path.unit_regular_star(6)
    circle = matplotlib.path.Path.unit_circle()
    # concatenate the circle with an internal cutout of the star
    verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
    codes = np.concatenate([circle.codes, star.codes])
    return matplotlib.path.Path(verts, codes)


if __name__ == "__main__":

    app = QApplication(sys.argv)

    dims = 10, 10
    yaw_range = pose_storage.DEFAULT_YAW_RANGE
    pitch_range = pose_storage.DEFAULT_PITCH_RANGE

    storage = pose_storage.ScoreMatrix(
        dims, pitch_range=pitch_range, yaw_range=yaw_range
    )

    dsl = DSL(storage.scores)

    matplotlib_widget = MatplotlibWidget(storage)
    matplotlib_widget.show()

    dsl.dataChanged.connect(matplotlib_widget.update_plot)
    dsl.mainLoop()

    sys.exit(app.exec_())
