import logging
import sys

from PySide2 import QtCore, QtWidgets

from face_pose_dataset.controller import estimation, logging_controller, storage_control
from face_pose_dataset.model import score, storage
from face_pose_dataset.view import login, main


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setWindowTitle("Dataset")
        self.layouts = {}
        self.kill_callback = None
        # self.setGeometry(300, 200, 500, 400)

    def register_layout(self, key, layout):
        self.layouts[key] = layout

    @QtCore.Slot(str)
    def change_layout(self, key: str):
        lay = self.layouts[key]
        self.setCentralWidget(lay)
        self.show()

    def closeEvent(self, event):
        if self.kill_callback is not None:
            self.kill_callback(event)
        else:
            quit_msg = "Are you sure you want to exit the program?"
            reply = QtWidgets.QMessageBox.question(
                self,
                "Message",
                quit_msg,
                QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.No,
            )

            if reply == QtWidgets.QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    app = QtWidgets.QApplication(sys.argv)

    dims = 7, 7
    yaw_range = score.DEFAULT_YAW_RANGE
    pitch_range = (-45, 45)

    # MODELS
    scores = score.ScoreModel(dims, pitch_range=pitch_range, yaw_range=yaw_range)

    store = storage.DatasetModel(shape=dims)

    # VIEW
    window = MainWindow()
    window.resize(500, 300)
    login_widget = login.Login()

    window.register_layout("login", login_widget)
    window.change_layout("login")

    # Central widget
    widget = main.MainWidget(scores)
    window.register_layout("main", widget)

    # CONTROLLERS
    store_controller = storage_control.StorageController(app, scores, store)
    store_controller.change_pos.connect(widget.plot.update_pointer)

    logger = logging_controller.LoggingController(login_widget, store)
    logger.change_layout.connect(window.change_layout)

    th = estimation.EstimationThread(800, 600)
    th.video_feed.connect(widget.video.set_image)
    th.result_signal.connect(store_controller.process)
    th.setTerminationEnabled(True)
    th.start()

    login_widget.switch_window.connect(logger.access)
    logger.set_camera.connect(th.init_camera)

    store_controller.flag_pause.connect(th.set_pause)
    logger.signal_pause.connect(th.set_pause)
    widget.controls.buttons[0].clicked.connect(th.toggle_pause)

    widget.controls.buttons[1].clicked.connect(store_controller.terminateApp)

    scores.change_score.connect(widget.plot.update_plot)

    window.kill_callback = lambda x: store_controller.terminateApp()

    # Execute application
    sys.exit(app.exec_())
