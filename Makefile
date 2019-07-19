SHELL = /bin/bash
PYTHON = python
PIP = pip

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps         pip install -r requirements.txt"
	@echo "    deps-test    pip install -r requirements_test.txt"
	@echo "    install      pip install -e ."
	@echo "    test         python -m pytest test"
	@echo "    test/assets  prepare test assets"
	@echo ""
	@echo "  Variables"
	@echo ""

# END-EVAL

# pip install -r requirements.txt
deps:
	$(PIP) install -r requirements.txt

# pip install -r requirements_test.txt
deps-test:
	$(PIP) install -r requirements_test.txt

# pip install -e .
install:
	$(PIP) install -e .

# python -m pytest test
test: test/assets
	test -f model_dta_test.h5 || keraslm-rate train -m model_dta_test.h5 test/assets/*.txt
	keraslm-rate test -m model_dta_test.h5 test/assets/*.txt
	$(PYTHON) -m pytest test $(PYTEST_ARGS)

# prepare test assets
test/assets:
	test/prepare_gt.bash $@

.PHONY: help deps deps-test install test
