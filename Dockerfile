ARG DOCKER_BASE_IMAGE=docker.io/ocrd/core-cuda-tf1
FROM $DOCKER_BASE_IMAGE
ARG DOCKER_BASE_IMAGE=docker.io/ocrd/core-cuda-tf1
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://ocr-d.de/en/contact" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/OCR-D/ocrd_keraslm" \
    org.label-schema.build-date=$BUILD_DATE \
    org.opencontainers.image.vendor="DFG-Funded Initiative for Optical Character Recognition Development" \
    org.opencontainers.image.title="ocrd_keraslm" \
    org.opencontainers.image.description="Simple character-based language model using Keras" \
    org.opencontainers.image.source="https://github.com/OCR-D/ocrd_keraslm" \
    org.opencontainers.image.documentation="https://github.com/OCR-D/ocrd_keraslm/blob/${VCS_REF}/README.md" \
    org.opencontainers.image.revision=$VCS_REF \
    org.opencontainers.image.created=$BUILD_DATE \
    org.opencontainers.image.base.name=$DOCKER_BASE_IMAGE

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONIOENCODING utf8
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# avoid HOME/.local/share (hard to predict USER here)
# so let XDG_DATA_HOME coincide with fixed system location
# (can still be overridden by derived stages)
ENV XDG_DATA_HOME /usr/local/share
# avoid the need for an extra volume for persistent resource user db
# (i.e. XDG_CONFIG_HOME/ocrd/resources.yml)
ENV XDG_CONFIG_HOME /usr/local/share/ocrd-resources

SHELL ["/bin/bash", "-c"]

WORKDIR /build/ocrd_keraslm

COPY . .
COPY ocrd-tool.json .

# prepackage ocrd-tool.json as ocrd-all-tool.json
RUN ocrd ocrd-tool ocrd-tool.json dump-tools > $(dirname $(ocrd bashlib filename))/ocrd-all-tool.json
# install everything and reduce image size
RUN pip install . && rm -fr /build/ocrd_keraslm

WORKDIR /data
VOLUME /data
