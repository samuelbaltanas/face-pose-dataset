SRC = test face_pose_dataset

.PHONY: clean clean-pyc clean-build clean-test test lint build format run

VENV_NAME?=venv
VENV_ACTIVATE=. $(VENV_NAME)/bin/activate
PYTHON=python

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .pytest_cache/

format:
	${PYTHON} -m isort -rc ${SRC}
	${PYTHON} -m black ${SRC}

lint: format
	${PYTHON} -m mypy ${SRC}
	${PYTHON} -m flake8 ${SRC}

test:
	${PYTHON} -m pytest

#dist: clean ## builds source and wheel package
#	python setup.py sdist
#	python setup.py bdist_wheel
#	ls -l dist

install: clean ## install the package to the active Python's site-packages
	${PYTHON} -m pip install -r requirements.txt
	${PYTHON} setup.py install

install-dev: clean
	${PYTHON} -m pip install -r requirements-dev.txt
	${PYTHON} setup.py develop

build: clean
	${PYTHON} setup.py develop

run:
	face_pose_dataset
