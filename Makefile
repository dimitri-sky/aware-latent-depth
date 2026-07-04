PY ?= .venv/bin/python

.PHONY: data test audit validity train_tiny eval ablate report aa

data:
	$(PY) scripts/make_data.py --split train --per-family 4000
	$(PY) scripts/make_data.py --split eval --per-family 400

test:
	$(PY) -m pytest tests/ -q

audit:
	$(PY) -m sage.contamination.audit --train-dir data/sage/train --eval-dir data/sage/eval

validity:
	$(PY) scripts/validity_gate.py

train_tiny:
	$(PY) scripts/train_tiny.py --config experiments/configs/exp001_loop_falsifier.yaml

eval:
	$(PY) scripts/eval.py

ablate:
	$(PY) scripts/ablate.py

report:
	$(PY) scripts/report.py

aa:
	$(PY) scripts/aa_test.py
