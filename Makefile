.PHONY: setup test lint

setup:
	conda env create -f conda.yaml

test:
	python -m pytest tests

lint:
	python -m flake8 main.py

