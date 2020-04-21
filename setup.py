from setuptools import setup, find_packages

setup(
    name="face_pose_dataset",
    version="0.1.0",
    packages=find_packages(),
    url="",
    license="",
    author="Samuel Baltanas",
    author_email="sambalmol@uma.es",
    description="",
    entry_points={
        "console_scripts": [
            "face_pose_dataset = face_pose_dataset.__main__:main",
        ],
    },
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'Pillow',
        'scipy',
        'openni',
        'opencv-python',
        'pyside2',
        'tensorflow>=2',
        'scikit-image',
        'torch',
        'torchvision',
        'tqdm',
    ],
    include_package_data=True,
)
