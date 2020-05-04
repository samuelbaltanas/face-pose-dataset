import logging
import os

from PySide2 import QtCore, QtWidgets

# DONE. Interface for setting a path
# Example in https://gist.github.com/MalloyDelacroix/2c509d6bcad35c7e35b1851dfc32d161
# https://david-estevez.gitbooks.io/tutorial-pyside-pyqt4/content/06_dialog.html
from PySide2.QtWidgets import QDialog

_DEST_NAME = "faces_dataset"

class Login(QtWidgets.QWidget):
    switch_window = QtCore.Signal(str, str, str)

    def __init__(self, default_path="~"):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle("Login")

        layout = QtWidgets.QGridLayout()

        fol = self.make_dataset_widget(default_path)
        layout.addWidget(fol, 0, 0, 1, 1)

        fol = self.make_id_widget()
        layout.addWidget(fol, 1, 0, 1, 1)

        groupBox = self.make_radio_widget()
        layout.addWidget(groupBox, 2, 0, 1, 1)

        self.button = QtWidgets.QPushButton("Access")
        self.button.clicked.connect(self.login_callback)
        layout.addWidget(self.button, 3, 0, 1, 1)

        self.setLayout(layout)

    def make_radio_widget(self):
        groupBox = QtWidgets.QGroupBox("Exclusive Radio Buttons")
        self.radio1 = QtWidgets.QRadioButton("Webcam")
        self.radio2 = QtWidgets.QRadioButton("Orbecc Astra")
        self.radio2.setChecked(True)
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.radio1)
        vbox.addWidget(self.radio2)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

    def make_id_widget(self):
        fol = QtWidgets.QGroupBox(title="ID:")
        self.id_edit = QtWidgets.QLineEdit("sam")
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.id_edit)
        fol.setLayout(hbox)

        return fol

    def make_dataset_widget(self, default_path):
        fol = QtWidgets.QGroupBox(title="Dataset root:")
        default_path = os.path.expanduser(default_path)
        self.folder_edit = QtWidgets.QLineEdit(default_path)
        self.folder_select = QtWidgets.QToolButton(text="...")
        self.folder_select.clicked.connect(self.input_file_button_callback)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.folder_edit)
        hbox.addWidget(self.folder_select)
        fol.setLayout(hbox)

        return fol

    def login_callback(self):
        camera = ""
        if self.radio1.isChecked():
            self.camera_id = "webcam"
        elif self.radio2.isChecked():
            self.camera_id = "astra"

        self.form = Form(self)
        self.form.show()

    def input_file_button_callback(self):
        logging.debug("Toggled")
        selected_directory = QtWidgets.QFileDialog.getExistingDirectory()

        if selected_directory:
            self.folder_edit.setText(selected_directory)

    def start_application_callback(self):
        folder, iden = self.folder_edit.text(), self.id_edit.text()
        res = self.validate_path(folder, iden)
        if res is not None:
            folder, iden = res
            self.switch_window.emit(
                folder, iden, self.camera_id
            )

    def validate_path(self, root, id):
        dataset_dir = os.path.join(root, _DEST_NAME)

        if not os.path.isdir(dataset_dir):
            os.mkdir(dataset_dir)
            logging.debug("Creating root folder: %s", dataset_dir)

        id_dir = os.path.join(dataset_dir, id)

        if os.path.isdir(id_dir):
            msg = "The identity {} has already been saved. Do you want to continue? The previous data will be overriden.".format(id)
            reply = QtWidgets.QMessageBox.question(
                self,
                "Message",
                msg,
                QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes,
            )

            if reply == QtWidgets.QMessageBox.Yes:
                for f in os.listdir(id_dir):
                    os.remove(os.path.join(id_dir, f))
                return dataset_dir, id
            else:
                return None

        else:
            os.mkdir(id_dir)
            return dataset_dir, id

        logging.debug("Creating new folder: %s", self.id_path)




_PERMISSION = (
    "<html>"
    "<h1>Permiso de datos</h1>"
    "<p>"
    "Autorizo que mis imágenes sean utilizadas para el desarrollo del Trabajo de Fin de Máster "
    "Tecnicas para mejorar el reconocimiento de caras para su uso "
    'por robots móviles asistentes". realizado por Samuel Felipe Baltanás con DNI 51182572M. '
    "Los datos proporcionados podrán ser utilizados para la creación de un dataset, "
    "para el entrenamiento y evaluación de redes neuronales, la realización de "
    "la memoria del Trabajo de Fin de Máster o ser subidos a internet como parte del conjunto "
    "formado por el dataset y las redes neuronales creadas."
    "</p>"
    "<p>"
    "Los datos podrán ser eliminados una vez que el Trabajo de Fin de Máster haya sido "
    "evaluado, previa petición por correo electrónico a <a href=mailto://sambalmol@uma.es>sambalmol@uma.es</a>."
    "</p>"
    "<p>"
    "El tratamiento de los datos, tanto por el responsable del Trabajo de Fin de Máster, "
    "como por terceros, se realizará de acuerdo con el REGLAMENTO (UE) 2016/679 DEL "
    "PARLAMENTO EUROPEO Y DEL CONSEJO de 27 de abril de 2016 relativo a la "
    "protección de las personas fı́sicas en lo que respecta al tratamiento de datos personales "
    "y a la libre circulación de estos datos y por el que se deroga la Directiva 95/46/CE "
    "(Reglamento general de protección de datos)."
    "</p>"
    "</html>"
)


class Form(QDialog):
    def __init__(
        self, origin: Login, parent=None,
    ):
        super(Form, self).__init__(parent)
        self.origin = origin
        self.setWindowTitle("Form")

        # Create widgets
        self.edit = QtWidgets.QLabel(_PERMISSION)
        self.edit.setFrameStyle(QtWidgets.QFrame.Panel)
        self.edit.setTextFormat(QtCore.Qt.RichText)
        self.edit.setStyleSheet("color: black; background-color: white")
        self.edit.setWordWrap(True)
        self.button = QtWidgets.QPushButton("Aceptar")
        self.button2 = QtWidgets.QPushButton("Cancelar")
        # Create layout and add widgets
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.edit)
        layout.addWidget(self.button)
        layout.addWidget(self.button2)
        # Set dialog layout
        self.setLayout(layout)
        # Add button signal to greetings slot
        self.button.clicked.connect(self.accept)
        self.button2.clicked.connect(self.deny)

    def deny(self):
        self.destroy()

    def accept(self):
        self.origin.start_application_callback()
        self.destroy()
