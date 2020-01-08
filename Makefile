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
	@echo "    install      pip install ."
	@echo "    test         python -m pytest test"
	@echo "    test/assets  prepare test assets"
	@echo ""
	@echo "  Variables"
	@echo "    PYTEST_ARGS  pytest args. Set to '-s' to see log output during test execution, '--verbose' to see individual tests. Default: '$(PYTEST_ARGS)'"
	@echo ""

# END-EVAL

deps:
	$(PIP) install -r requirements.txt

deps-test: install
	$(PIP) install -r requirements_test.txt

install: deps
	$(PIP) install .

export TF_CPP_MIN_LOG_LEVEL = 1
test: test/assets deps-test
	test -f model_dta_test.h5 || keraslm-rate train -m model_dta_test.h5 test/assets/*.txt
	keraslm-rate test -m model_dta_test.h5 test/assets/*.txt
	$(PYTHON) -m pytest test $(PYTEST_ARGS)

# prepare test assets
test/assets:
	# TODO: instead of this, use bag repos, or add something useful to OCR-D/assets
	test/prepare_gt.bash $@

.PHONY: help deps deps-test install test
