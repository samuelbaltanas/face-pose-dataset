from setuptools import setup, find_packages

setup(
    name="face_pose_dataset",
    version="0.1.0",
    packages=["face_pose_dataset"],
    url="",
    license="",
    author="Samuel Baltanas",
    author_email="sambalmol@uma.es",
    description="Face pose gathering tool.",
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
    # package_data={'face_pose_dataset': ['models']},
    include_package_data=True,
)
