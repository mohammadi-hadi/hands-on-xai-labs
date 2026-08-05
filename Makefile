# Hands-On Explainable AI — companion labs
# Requires: uv (https://docs.astral.sh/uv/)

UV ?= uv
PY := .venv/bin/python

.PHONY: setup setup-ml setup-mechinterp nb-run nb-smoke

setup:
	$(UV) sync

setup-ml:
	$(UV) sync --group llm

setup-mechinterp:
	$(UV) sync --group llm --group mechinterp

nb-run:
	$(PY) scripts/run_notebooks.py

nb-smoke:
	BOOK_SMOKE=1 $(PY) scripts/run_notebooks.py
