SHELL = /bin/bash
PYTHON ?= python
PIP ?= pip
DOCKER_BASE_IMAGE ?= docker.io/ocrd/core-cuda:v2.69.0
DOCKER_TAG ?= ocrd/keraslm
PYTEST_ARGS ?= -vv

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps         pip install -r requirements.txt"
	@echo "    install      pip install ."
	@echo "    install-dev  pip install -e ."
	@echo "    deps-test    pip install -r requirements_test.txt"
	@echo "    test         python -m pytest test"
	@echo "    test/assets  prepare test assets"
	@echo "    build        python -m build ."
	@echo "    docker       build Docker image"
	@echo ""
	@echo "  Variables"
	@echo "    PYTHON       name of the Python binary [$(PYTHON)]"
	@echo "    PIP          name of the Python packager [$(PIP)]"
	@echo "    TAG          name of the Docker image [$(DOCKER_TAG)]"
	@echo "    PYTEST_ARGS  extra runtime arguments for test [$(PYTEST_ARGS)]"
	@echo ""

# END-EVAL

deps:
	$(PIP) install -r requirements.txt

DEU_FRAK_URL = https://github.com/tesseract-ocr/tessdata/raw/4.1.0/deu_frak.traineddata
deps-test:
	$(PIP) install -r requirements_test.txt
	ocrd resmgr download -n $(DEU_FRAK_URL) ocrd-tesserocr-recognize deu-frak.traineddata

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

install-dev:
	$(PIP) install -e .

build:
	$(PIP) install build wheel
	$(PYTHON) -m build .

docker:
	docker build \
	-t $(DOCKER_TAG) \
	--build-arg DOCKER_BASE_IMAGE=$(DOCKER_BASE_IMAGE) \
	--build-arg VCS_REF=$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") .

export TF_CPP_MIN_LOG_LEVEL = 1
test: test/assets
ifeq ($(TEST_TRAINING),)
test: export OCRD_KERASLM_MODEL = model_dta_full.h5
test:
	ocrd resmgr download ocrd-keraslm-rate $(OCRD_KERASLM_MODEL)
else
test: export OCRD_KERASLM_MODEL = $(CURDIR)/model_dta_test.h5
test:
	test -f $(OCRD_KERASLM_MODEL) || keraslm-rate train -m $(OCRD_KERASLM_MODEL) test/assets/*.txt
	keraslm-rate test -m $(OCRD_KERASLM_MODEL) test/assets/*.txt
endif
	$(PYTHON) -m pytest test $(PYTEST_ARGS)

# prepare test assets
test/assets: repo/assets
	mkdir -p $@
	ocrd workspace -d $@/kant_aufklaerung_1784 clone --download $</data/kant_aufklaerung_1784/data/mets.xml
	bash test/prepare_gt.bash $@

repo/assets: always-update
	git submodule sync $@
	git submodule update --init $@

clean:
	$(RM) -r test/assets model_dta_test.h5

.PHONY: help deps deps-test install install-dev build test clean docker always-update nvidia-tensorflow
