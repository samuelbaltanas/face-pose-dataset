# Installation instructions

## Setup

1. Clone or download the repository from github.
    ```bash
    git clone https://github.com/samuelbaltanas/face-pose-dataset.git
    ```
    2. (Optional) Create a virtual environment using conda or virtualenv.
2. Install the projects' requirements using:
    ```bash
   pip3 install -r requirements.txt
   ```
   1. (Optional) In Ubuntu, Pyside2 might throw some errors at runtime. For example:
        ```
        ImportError: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5: version `Qt_5.14' not found ...
        ```
       It is usually a problem of an incorrect seting of the $LD_LIBRARY_PATH bash variable.
       Check <https://pypi.org/project/PySide2/> or [this question](https://stackoverflow.com/questions/36128645/error-on-execution-version-qt-5-not-found-required-by#answer-36195503),
       in my case it was solved by removing `/usr/lib/x86_64-linux-gnu` from $LD_LIBRARY_PATH
3. From the projects' root folder run:
    ```bash
    python3 face_pose_dataset
    ```