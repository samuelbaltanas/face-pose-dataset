# Installation instructions

## Notes

This application requires that [Python 3](https://www.python.org/downloads/) is installed in the system to run.

The system has been tested under:
- Python version 3.6.9 in Ubuntu.
- Python version 3.8.3 in Windows.

(WARN) If installing Python 3 in Windows ensure that you are installing the x86-64 version.

## Setup

1. Clone or download the repository from github.
    ```bash
        git clone https://github.com/samuelbaltanas/face-pose-dataset.git
    ```
2. (Optional) Create a virtual environment using [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
2. Install the projects' requirements:
    - Option 1. For general use install the CPU-only requirements:
        ```bash
            pip3 install -r requirements-cpu.txt
        ```
    - Option 2. If you have a GPU and CUDA installed use:
        ```bash
            pip3 install -r requirements.txt
        ```

3. From the projects' root folder run:
    ```bash
    python3 -m face_pose_dataset
    ```


## Troubleshooting

- In Ubuntu, Pyside2 might throw some errors at runtime. For example:
    ```
    ImportError: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5: version `Qt_5.14' not found ...
    ```
   It is usually a problem of an incorrect seting of the $LD_LIBRARY_PATH bash variable.
   Check <https://pypi.org/project/PySide2/> or [this question](https://stackoverflow.com/questions/36128645/error-on-execution-version-qt-5-not-found-required-by#answer-36195503),
   in my case it was solved by removing `/usr/lib/x86_64-linux-gnu` from $LD_LIBRARY_PATH
   
- Tensorflow binaries are only distributed for x64 systems. Therefore, the installation of requirements
will fail if used on Python for x32 systems.