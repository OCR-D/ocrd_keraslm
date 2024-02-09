SHELL = /bin/bash
PYTHON ?= python
PIP ?= pip
TAG ?= ocrd/keraslm

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps         pip install -r requirements.txt"
	@echo "    install      pip install ."
	@echo "    deps-test    pip install -r requirements_test.txt"
	@echo "    test         python -m pytest test"
	@echo "    test/assets  prepare test assets"
	@echo "    docker       build Docker image"
	@echo ""
	@echo "  Variables"
	@echo "    PYTHON       name of the Python binary. Default: $(PYTHON)"
	@echo "    PIP          name of the Python packager. Default: $(PIP)"
	@echo "    TAG          name of the Docker image. Default: $(TAG)"
	@echo "    PYTEST_ARGS  pytest args. Set to '-s' to see log output during test execution, '--verbose' to see individual tests. Default: '$(PYTEST_ARGS)'"
	@echo ""

# END-EVAL

deps:
	$(PIP) install -r requirements.txt

deps-test:
	$(PIP) install -r requirements_test.txt

nvidia-tensorflow:
	if $(PYTHON) -c 'import sys; print("%u.%u" % (sys.version_info.major, sys.version_info.minor))' | fgrep 3.8 && \
	! pip show -q tensorflow-gpu; then \
	  pip install nvidia-pyindex && \
	  pushd $$(mktemp -d) && \
	  pip download --no-deps nvidia-tensorflow && \
	  for name in nvidia_tensorflow-*.whl; do name=$${name%.whl}; done && \
	  $(PYTHON) -m wheel unpack $$name.whl && \
	  for name in nvidia_tensorflow-*/; do name=$${name%/}; done && \
	  newname=$${name/nvidia_tensorflow/tensorflow_gpu} &&\
	  sed -i s/nvidia_tensorflow/tensorflow_gpu/g $$name/$$name.dist-info/METADATA && \
	  sed -i s/nvidia_tensorflow/tensorflow_gpu/g $$name/$$name.dist-info/RECORD && \
	  sed -i s/nvidia_tensorflow/tensorflow_gpu/g $$name/tensorflow_core/tools/pip_package/setup.py && \
	  pushd $$name && for path in $$name*; do mv $$path $${path/$$name/$$newname}; done && popd && \
	  $(PYTHON) -m wheel pack $$name && \
	  pip install $$newname*.whl && popd && rm -fr $$OLDPWD; \
	fi

install:
	$(PIP) install .

docker:
	docker build \
	-t $(TAG) \
	--build-arg VCS_REF=$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") .

export TF_CPP_MIN_LOG_LEVEL = 1
test: test/assets
	test -f model_dta_test.h5 || keraslm-rate train -m model_dta_test.h5 test/assets/*.txt
	keraslm-rate test -m model_dta_test.h5 test/assets/*.txt
	$(PYTHON) -m pytest test $(PYTEST_ARGS)

# prepare test assets
test/assets: repo/assets
	mkdir -p $@
	ocrd workspace -d $@/kant_aufklaerung_1784 clone $</data/kant_aufklaerung_1784/data/mets.xml --download
	bash test/prepare_gt.bash $@

repo/assets: always-update
	git submodule sync $@
	git submodule update --init $@

clean:
	$(RM) -r test/assets model_dta_test.h5

.PHONY: help deps deps-test install test clean always-update nvidia-tensorflow
