Change Log
==========
Versioned according to [Semantic Versioning](http://semver.org/).

## Unreleased

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
[0.4.2]: ../../compare/v0.4.1...v0.4.2
[0.4.1]: ../../compare/v0.4.0...v0.4.1
[0.4.0]: ../../compare/0.3.2...v0.4.0
[0.3.2]: ../../compare/0.3.1...0.3.2
[0.3.1]: ../../compare/HEAD...0.3.1
