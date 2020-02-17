SHELL = /bin/bash
PYTHON ?= python
PIP ?= pip

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
	@echo "    PYTHON       name of the Python binary. Default: $(PYTHON)"
	@echo "    PIP          name of the Python packager. Default: $(PIP)"
	@echo "    PYTEST_ARGS  pytest args. Set to '-s' to see log output during test execution, '--verbose' to see individual tests. Default: '$(PYTEST_ARGS)'"
	@echo ""

# END-EVAL

deps:
	$(PIP) install -r requirements.txt

deps-test:
	$(PIP) install -r requirements_test.txt

install: deps
	$(PIP) install .

export TF_CPP_MIN_LOG_LEVEL = 1
test: test/assets deps-test
	test -f model_dta_test.h5 || keraslm-rate train -m model_dta_test.h5 test/assets/*.txt
	keraslm-rate test -m model_dta_test.h5 test/assets/*.txt
	$(PYTHON) -m pytest test $(PYTEST_ARGS)

# prepare test assets
test/assets: repo/assets
	mkdir -p $@
	ocrd workspace clone $</data/kant_aufklaerung_1784/data/mets.xml -a $@/kant_aufklaerung_1784
	bash test/prepare_gt.bash $@

repo/assets: always-update
	git submodule sync $@
	git submodule update --init $@

clean:
	$(RM) -r test/assets model_dta_test.h5

.PHONY: help deps deps-test install test clean always-update
