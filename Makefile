# Heart Reader Challenge – convenience targets

PYTHON := python3

install:
	pip install -r requirements.txt

data:
	bash download_data.sh

train:
	$(PYTHON) train.py

evaluate:
	$(PYTHON) evaluate.py

visualize:
	$(PYTHON) visualize_results.py

optimize:
	$(PYTHON) optimize.py --quantize

all: train evaluate visualize

.PHONY: install data train evaluate visualize optimize all
