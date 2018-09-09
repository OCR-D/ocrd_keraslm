# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps       pip install -r requirements.txt"
	@echo "    deps-test  pip install -r requirements_test.txt"
	@echo ""
	@echo "    install    pip install -e ."
	@echo "    test       python -m pytest test"

# END-EVAL

deps:
	pip install -r requirements.txt

deps-test:
	pip install -r requirements_test.txt

install:
	pip install -e .

test: test/assets
	test -f model_dta_test.weights.h5 -a -f model_dta_test.config.pkl || keraslm-rate train -m model_dta_test.weights.h5 -c model_dta_test.config.pkl test/assets/*.txt
	keraslm-rate test -m model_dta_test.weights.h5 -c model_dta_test.config.pkl test/assets/*.txt
	python -m pytest test $(PYTEST_ARGS)

test/assets:
	test/prepare_gt.bash $@

.PHONY: help deps deps-test install test
