from PySide2 import QtCore

from face_pose_dataset.model import storage


class LoggingController(QtCore.QObject):
    change_layout = QtCore.Signal(str)
    set_camera = QtCore.Signal(str)

    def __init__(self, widget, storage: storage.DatasetModel):
        super().__init__()
        self.widget = widget
        self.storage = storage

    @QtCore.Slot(str, str, str)
    def access(self, path, id, camera):
        # Marker for current location
        self.storage.path = path
        self.storage.add_identity(id)

        self.change_layout.emit("main")
        self.set_camera.emit(camera)
