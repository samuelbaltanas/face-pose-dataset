from typing import Tuple

import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets

__all__ = ["VideoWidget"]


def numpy_to_qimage(frame: np.ndarray, target_shape: Tuple[int, int]):
    if len(frame.shape) == 3:
        h, w, ch = frame.shape
        img_format = QtGui.QImage.Format_RGB888
    else:
        h, w = frame.shape
        ch = 1
        img_format = QtGui.QImage.Format_Grayscale16

    bytes_per_line = ch * w * frame.itemsize
    converted_img = QtGui.QImage(frame.data, w, h, bytes_per_line, img_format)

    width, height = target_shape
    return converted_img.scaled(width, height, QtCore.Qt.KeepAspectRatio)


class VideoWidget(QtWidgets.QWidget):
    def __init__(self, width=640, height=480):
        super().__init__()
        self.width = width
        self.height = height
        self._init_ui()

    @QtCore.Slot(np.ndarray)
    def set_image(self, image: np.ndarray):
        image = numpy_to_qimage(image, (self.width, self.height))
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    def _init_ui(self):
        # Create a label
        self.label = QtWidgets.QLabel(self)
        # self.label.resize(self.width, self.height)
        self.label.setScaledContents(True)

        # QWidget Layout
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.label)

        # Set the layout to the QWidget
        self.setLayout(self.layout)
