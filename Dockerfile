ARG DOCKER_BASE_IMAGE
FROM $DOCKER_BASE_IMAGE
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://github.com/OCR-D/ocrd_keraslm/issues" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/OCR-D/ocrd_keraslm" \
    org.label-schema.build-date=$BUILD_DATE

SHELL ["/bin/bash", "-c"]
WORKDIR /build/ocrd_keraslm

COPY setup.py .
COPY ocrd-tool.json .
COPY ocrd_keraslm ./ocrd_keraslm
COPY Makefile .
COPY requirements.txt .
COPY README.md .
RUN make nvidia-tensorflow
# - preempt conflict over numpy between scikit-image and tensorflow
# - preempt conflict over numpy between tifffile and tensorflow (and allow py36)
RUN pip install imageio==2.14.1 "tifffile<2022"
# - preempt conflict over numpy between h5py and tensorflow
RUN pip install "numpy<1.24"
RUN pip install .
RUN rm -fr /build/ocrd_keraslm

WORKDIR /data
VOLUME ["/data"]
