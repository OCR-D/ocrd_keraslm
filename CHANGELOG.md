Change Log
==========
Versioned according to [Semantic Versioning](http://semver.org/).

## Unreleased

## [0.5.1] - 2025-08-13

Changed:

 * wrapper.rate: when projecting from `textequiv_level` upwards,
   respect recursive ReadingOrder / `@textLineOrder` / `@readingDirection`,
   and also update `@conf`

## [0.5.0] - 2025-04-16

Fixed:

 * train: continuing from checkpoint

Added:

 * train: allow passing single file for `val_data`
 * train: save training history and metrics to model
 * wrapper.rate: reference `model_dta_full.h5` in ocrd-tool.json
 * docker: labels and envvars according to spec
 * docker: preinstall `ocrd-all-tool.json`

Changed:

 * wrapper.rate: use normal `resolve_resource` for model file
 * wrapper.rate: improve docstring/help and readme
 * wrapper.rate: adapt to OCR-D v3
 * test: allow running with published model instead of training from scratch
 * test: improve, update, migrate unittest → pytest
 * test: split scoring and decoding, add assertions, add modes w/ METS Server / page worker
 * docker: rebase on ocrd/core-cuda-tf1 v3
 * setup.py → pyproject.toml, versioning based on ocrd-tool.json

## [0.4.3] - 2024-03-11

Changed:

  * `plot_context_embeddings_projection`: add year labels

Added:

  * test: allow passing directory for test data, too
  * generate: option `--variants` for nr of alternative sequences

## [0.4.2] - 2024-03-04

Fixed:

  * adapt to recent numpy/h5py/protobuf and OCR-D changes
  * prolong Tensorflow 1 life on Py38 via nvidia-tensorflow image
  * suppress TF verbose messages

Changed:

  * lib: improve augmentation and stateless modes
  * tests: use OCR-D resmgr for Tesseract models
  * CI: update tests, adapt to ocrd/tesserocr changes

Added:

  * add continuous deployment via Docker
  * train: allow passing directory for training data, too
  * lib: add checkpointing
  * train: allow continuing from checkpoint

## [0.4.1] - 2020-09-24

Changed:

  * logging according to OCR-D/core#599

## [0.4.0] - 2020-08-21

Fixed:

  * deps: relax tensorflow, use -gpu variant
  * deps: restrict keras<2.4

Changed:

  * adapt tests to core#397
  * update tests from obsolete bags to assets repo
  * create CircleCI config
  * adapt to 1-output-file-group convention, use `make_file_id` and `assert_file_grp_cardinality`, #17
  * set pcGtsId to file ID, #17

## [0.3.2] - 2019-11-18

Fixed:

  * deps: restrict tensorflow<2
  * deps: require ocrd>=2

## [0.3.1] - 2019-10-26



<!-- link-labels -->
[0.5.1]: ../../compare/v0.5.0...v0.5.1
[0.5.0]: ../../compare/v0.4.3...v0.5.0
[0.4.3]: ../../compare/v0.4.2...v0.4.3
[0.4.2]: ../../compare/v0.4.1...v0.4.2
[0.4.1]: ../../compare/v0.4.0...v0.4.1
[0.4.0]: ../../compare/0.3.2...v0.4.0
[0.3.2]: ../../compare/0.3.1...0.3.2
[0.3.1]: ../../compare/HEAD...0.3.1
