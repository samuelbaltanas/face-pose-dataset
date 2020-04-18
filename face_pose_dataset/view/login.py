import logging

from PySide2 import QtCore, QtWidgets

# TODO. Interface for setting a path
# Example in https://gist.github.com/MalloyDelacroix/2c509d6bcad35c7e35b1851dfc32d161
# https://david-estevez.gitbooks.io/tutorial-pyside-pyqt4/content/06_dialog.html


class Login(QtWidgets.QWidget):
    switch_window = QtCore.Signal(str, str, str)

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle("Login")

        layout = QtWidgets.QGridLayout()

        fol = QtWidgets.QGroupBox(title="Dataset root:")
        self.folder_edit = QtWidgets.QLineEdit("/home/sam/Desktop/sample/")
        self.folder_select = QtWidgets.QToolButton(text="...")
        self.folder_select.clicked.connect(self.onInputFileButtonClicked)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.folder_edit)
        hbox.addWidget(self.folder_select)
        fol.setLayout(hbox)
        layout.addWidget(fol, 0, 0, 1, 1)

        self.id_edit = QtWidgets.QLineEdit("sam")

        self.button = QtWidgets.QPushButton("Access")
        self.button.clicked.connect(self.login)

        fol = QtWidgets.QGroupBox(title="ID:")
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.id_edit)
        fol.setLayout(hbox)

        layout.addWidget(fol, 1, 0, 1, 1)

        groupBox = QtWidgets.QGroupBox("Exclusive Radio Buttons")

        self.radio1 = QtWidgets.QRadioButton("Webcam")
        self.radio2 = QtWidgets.QRadioButton("Orbecc Astra")

        self.radio2.setChecked(True)
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.radio1)
        vbox.addWidget(self.radio2)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        layout.addWidget(groupBox, 2, 0, 1, 1)

        layout.addWidget(self.button, 3, 0, 1, 1)

        self.setLayout(layout)

    def login(self):
        camera = ""
        if self.radio1.isChecked():
            camera = "webcam"
        elif self.radio2.isChecked():
            camera = "astra"
        self.switch_window.emit(self.folder_edit.text(), self.id_edit.text(), camera)

    def onInputFileButtonClicked(self):
        logging.debug("Toggled")
        selected_directory = QtWidgets.QFileDialog.getExistingDirectory()

        if selected_directory:
            self.folder_edit.setText(selected_directory)
