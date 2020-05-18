import argparse
import logging
import os
import sys

from PySide2 import QtCore, QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setWindowTitle("Dataset")
        self.layouts = {}
        self.kill_callback = None
        # self.setGeometry(300, 200, 500, 400)

    def register_layout(self, key, layout, resolution=(600, 360)):
        self.layouts[key] = layout, resolution

    @QtCore.Slot(str)
    def change_layout(self, key: str):
        lay = self.layouts[key]
        self.setCentralWidget(lay[0])
        self.resize(*lay[1])
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


def main(args):
    try:
        if args.quiet:
            logging.getLogger().setLevel(logging.ERROR)
        elif args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
        app = QtWidgets.QApplication(sys.argv)

        logging.info(args)

        if args.force_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        from face_pose_dataset.controller import estimation, logging_controller, storage_control
        from face_pose_dataset.model import score, storage
        from face_pose_dataset.view import login, main_view

        # PARAMS
        dims = 7, 7
        yaw_range = -65.0, 65.0
        pitch_range = -35.0, 35.0

        # MODELS
        scores = score.ScoreModel(dims, pitch_range=pitch_range, yaw_range=yaw_range)
        store = storage.DatasetModel(shape=dims)

        # VIEW
        window = MainWindow()
        window.resize(600, 360)

        # Login
        login_widget = login.Login()
        window.register_layout("login", login_widget)
        window.change_layout("login")

        # Main widget
        widget = main_view.MainWidget(scores)
        window.register_layout("main", widget, (1200, 720))

        # CONTROLLERS
        store_controller = storage_control.StorageController(scores, store)
        store_controller.change_pos.connect(widget.plot.update_pointer)


        logger = logging_controller.LoggingController(login_widget, store)
        logger.change_layout.connect(window.change_layout)

        if args.force_cpu:
            gpu = -1
        else:
            gpu = 0

        th = estimation.EstimationThread(gpu=gpu)
        th.video_feed.connect(widget.video.set_image)
        th.worker.result_signal.connect(store_controller.process)
        # th.setTerminationEnabled(True)

        login_widget.switch_window.connect(logger.access)
        logger.set_camera.connect(th.init_camera)

        store_controller.flag_pause.connect(th.set_pause)
        widget.controls.buttons[0].clicked.connect(th.toggle_pause)

        widget.controls.buttons[1].clicked.connect(store_controller.terminateApp)
        store_controller.flag_end.connect(th.set_stop)

        scores.change_score.connect(widget.plot.update_plot)

        window.kill_callback = lambda x: store_controller.terminateApp()

        # Execute application
        logging.info("[MAIN] Running main loop.")

        _excepthook = sys.excepthook

        def exception_hook(exctype, value, traceback):
            print(exctype, value, traceback)
            _excepthook(exctype, value, traceback)
            sys.exit(-1)

        sys.excepthook = exception_hook
        # th.start()
        res = app.exec_()
    finally:
        logging.info("[MAIN] Waiting for thread to terminate.")
        # th.wait(2000)
        logging.info("[MAIN] Terminated.")
        sys.exit(res)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Forces the estimator to use a CPU. When not set, it searches for any available GPU.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true")
    group.add_argument("-q", "--quiet", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
