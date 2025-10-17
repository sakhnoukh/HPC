.PHONY: help setup test download train clean

help:
	@echo "HPC CIFAR-10 Training - Available Commands"
	@echo "=========================================="
	@echo "make setup      - Install Python dependencies"
	@echo "make test       - Run environment tests"
	@echo "make download   - Download CIFAR-10 dataset"
	@echo "make train      - Run single-GPU training (1 epoch test)"
	@echo "make clean      - Clean results and cache files"

setup:
	pip install -r env/requirements.txt
	@echo "✓ Dependencies installed"

test:
	python test_setup.py

download:
	python data/fetch_cifar10.py --data-dir ./data

train:
	python src/train.py --epochs 1 --batch-size 64 --data ./data

clean:
	rm -rf results/csv/*.csv
	rm -rf results/plots/*.png
	rm -rf results/logs/*.out
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned results and cache files"
