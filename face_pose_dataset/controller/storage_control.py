import logging

from PySide2 import QtCore
from PySide2.QtWidgets import QMessageBox

from face_pose_dataset.core import EstimationData
from face_pose_dataset.model import score, storage


class StorageController(QtCore.QObject):
    change_pos = QtCore.Signal(tuple)
    flag_pause = QtCore.Signal(bool)
    flag_end = QtCore.Signal(bool)

    def __init__(self, app, scores: score.ScoreModel, storage: storage.DatasetModel):
        super().__init__()
        self.scores = scores
        self.storage = storage
        self.app = app

    @QtCore.Slot(EstimationData)
    def process(self, data: EstimationData):
        # Marker for current location
        off = self.scores.iloc(data.angle)
        logging.debug("Current location location %s", off)
        self.change_pos.emit(off)

        # Update contents of storage
        res = self.scores.add_angle(data.angle)
        if res:
            # DONE. Image saving logic.
            pos = self.scores.locate_angle(data.angle)
            self.storage.save_image(pos, data)

    def terminateApp(self,):
        self.flag_pause.emit(True)
        msgBox = QMessageBox()
        msgBox.setText("The application will be shutdown and save all changes.")
        msgBox.setInformativeText("Do you really want to quit the application?")
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msgBox.setDefaultButton(QMessageBox.Cancel)
        ret = msgBox.exec_()
        if ret == QMessageBox.Ok:
            self.storage.dump_data()
            logging.debug("Quitting application")
            self.flag_end.emit(True)
            self.app.quit()
        elif ret == QMessageBox.Cancel:
            self.flag_pause.emit(False)
