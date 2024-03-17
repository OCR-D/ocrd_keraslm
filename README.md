# ocrd_keraslm
    character-level language modelling using Keras

[![CircleCI](https://circleci.com/gh/OCR-D/ocrd_keraslm.svg?style=svg)](https://circleci.com/gh/OCR-D/ocrd_keraslm)
[![Docker Automated build](https://img.shields.io/docker/automated/ocrd/keraslm.svg)](https://hub.docker.com/r/ocrd/keraslm/tags/)

 * [Introduction](#introduction)
    * [Architecture](#architecture)
    * [Modes of operation](#modes-of-operation)
    * [Context conditioning](#context-conditioning)
    * [Underspecification](#underspecification)
 * [Installation](#installation)
 * [Usage](#usage)
    * [Command line interface `keraslm-rate`](#command-line-interface-keraslm-rate)
    * [OCR-D processor interface `ocrd-keraslm-rate`](#ocr-d-processor-interface-ocrd-keraslm-rate)
    * [Models](#models)
 * [Testing](#testing)

## Introduction

This is a tool for statistical _language modelling_ (predicting text from context) with recurrent neural networks. It models probabilities not on the word level but the _character level_ so as to allow open vocabulary processing (avoiding morphology, historic orthography and word segmentation problems). It manages a vocabulary of mapped characters, which can be easily extended by training on more text. Above that, unmapped characters are treated with underspecification.

In addition to character sequences, (meta-data) context variables can be configured as extra input. 

### Architecture

The model consists of:

0. an input layer: characters are represented as indexes from the vocabulary mapping, in windows of a number `length` of characters,
1. a character embedding layer: window sequences are converted into dense vectors by looking up the indexes in an embedding weight matrix,
2. a context embedding layer: context variables are converted into dense vectors by looking up the indexes in an embedding weight matrix, 
3. character and context vector sequences are concatenated,
4. a number `depth` of hidden layers: each with a number `width` of hidden recurrent units of _LSTM cells_ (Long Short-term Memory) connected on top of each other,
5. an output layer derived from the transposed character embedding matrix (weight tying): hidden activations are projected linearly to vectors of dimensionality equal to the character vocabulary size, then softmax is applied returning a probability for each possible value of the next character, respectively.

![model graph depiction](model-graph.png "graph with 1 context variable")

The model is trained by feeding windows of text in index representation to the input layer, calculating output and comparing it to the same text shifted backward by 1 character, and represented as unit vectors ("one-hot coding") as target. The loss is calculated as the (unweighted) cross-entropy between target and output. Backpropagation yields error gradients for each layer, which is used to iteratively update the weights (stochastic gradient descent).

This is implemented in [Keras](https://keras.io) with [Tensorflow](https://www.tensorflow.org/) as backend. It automatically uses a fast CUDA-optimized LSTM implementation (Nividia GPU and Tensorflow installation with GPU support, see below), both in learning and in prediction phase, if available.


### Modes of operation

Notably, this model (by default) runs _statefully_, i.e. by implicitly passing hidden state from one window (batch of samples) to the next. That way, the context available for predictions can be arbitrarily long (above `length`, e.g. the complete document up to that point), or short (below `length`, e.g. at the start of a text). (However, this is a passive perspective above `length`, because errors are never back-propagated any further in time during gradient-descent training.) This is favourable to stateless mode because all characters can be output in parallel, and no partial windows need to be presented during training (which slows down).

Besides stateful mode, the model can also be run _incrementally_, i.e. by explicitly passing hidden state from the caller. That way, multiple alternative hypotheses can be processed together. This is used for generation (sampling from the model) and alternative decoding (finding the best path through a sequence of alternatives).

### Context conditioning

Every text has meta-data like time, author, text type, genre, production features (e.g. print vs typewriter vs digital born rich text, OCR version), language, structural element (e.g. title vs heading vs paragraph vs footer vs marginalia), font family (e.g. Antiqua vs Fraktura) and font shape (e.g. bold vs letter-spaced vs italic vs normal) etc. 

This information (however noisy) can be very useful to facilitate stochastic modelling, since language has an extreme diversity and complexity. To that end, models can be conditioned on extra inputs here, termed _context variables_. The model learns to represent these high-dimensional discrete values as low-dimensional continuous vectors (embeddings), also entering the recurrent hidden layers (as a form of simple additive adaptation).

### Underspecification

Index zero is reserved for unmapped characters (unseen contexts). During training, its embedding vector is regularised to occupy a center position of all mapped characters (all other contexts), and the hidden layers get to see it every now and then by random degradation. At runtime, therefore, some unknown character (some unknown context) represented as zero does not disturb follow-up predictions too much.


## Installation

Required Ubuntu packages:

* Python (``python`` or ``python3``)
* pip (``python-pip`` or ``python3-pip``)
* virtualenv (``python-virtualenv`` or ``python3-virtualenv``)

Create and activate a virtualenv as usual.

If you need a custom version of ``keras`` or ``tensorflow`` (like [GPU support](https://www.tensorflow.org/install/install_sources)), install them via `pip` now.

To install Python dependencies and this module, then do:
```shell
make deps install
```
Which is the equivalent of:
```shell
pip install -r requirements.txt
pip install -e .
```

Useful environment variables are:
- ``TF_CPP_MIN_LOG_LEVEL`` (set to `1` to suppress most of Tensorflow's messages
- ``CUDA_VISIBLE_DEVICES`` (set empty to force CPU even in a GPU installation)


## Usage

This packages has two user interfaces:

### command line interface `keraslm-rate`

To be used with string arguments and plain-text files.

```shell
Usage: keraslm-rate [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  train                           train a language model
  test                            get overall perplexity from language model
  apply                           get individual probabilities from language model
  generate                        sample characters from language model
  print-charset                   Print the mapped characters
  prune-charset                   Delete one character from mapping
  plot-char-embeddings-similarity
                                  Paint a heat map of character embeddings
  plot-context-embeddings-similarity
                                  Paint a heat map of context embeddings
  plot-context-embeddings-projection
                                  Paint a 2-d PCA projection of context embeddings
```

Examples:
```shell
keraslm-rate train --width 64 --depth 4 --length 256 --model model_dta_64_4_256.h5 dta_komplett_2017-09-01/txt/*.tcf.txt
keraslm-rate generate -m model_dta_64_4_256.h5 --number 6 "für die Wiſſen"
keraslm-rate apply -m model_dta_64_4_256.h5 "so schädlich ist es Borkickheile zu pflanzen"
keraslm-rate test -m model_dta_64_4_256.h5 dta_komplett_2017-09-01/txt/grimm_*.tcf.txt
```

### [OCR-D processor](https://github.com/OCR-D/core) interface `ocrd-keraslm-rate`

To be used with [PageXML](https://www.primaresearch.org/tools/PAGELibraries) documents in an [OCR-D](https://github.com/OCR-D/spec/) annotation workflow. Input could be anything with a textual annotation (`TextEquiv` on the given `textequiv_level`). The LM rater could be used for both quality control (without alternative decoding, using only each first index `TextEquiv`) and part of post-correction (with `alternative_decoding=True`, finding the best path among `TextEquiv` indexes).

```shell
Usage: ocrd-keraslm-rate [worker|server] [OPTIONS]

  Rate elements of the text with a character-level LSTM language model in Keras

  > Rate text with the language model, either for scoring or finding the
  > best path across alternatives.

  > Open and deserialise PAGE input files, then iterate over the segment
  > hierarchy down to the requested `textequiv_level`, making sequences
  > of first TextEquiv objects (if `alternative_decoding` is false), or
  > of lists of all TextEquiv objects (otherwise) as a linear graph for
  > input to the LM. If the level is above glyph, then insert artificial
  > whitespace TextEquiv where implicit tokenisation rules require it.

  > Next, if `alternative_decoding` is false, then pass the concatenated
  > string of the page text to the LM and map the returned sequence of
  > probabilities to the substrings in the input TextEquiv. For each
  > TextEquiv, calculate the average character probability (LM score)
  > and combine that with the input confidence (OCR score) by applying
  > `lm_weight`. Assign the resulting probability as new confidence to
  > the TextEquiv, and ensure no other TextEquiv remain on the segment.
  > Finally, calculate the overall average LM probability,  and the
  > character and segment-level perplexity, and print it on the logger.

  > Otherwise (i.e with `alternative_decoding=true`), search for the
  > best paths through the input graph of the page (with TextEquiv
  > string alternatives as edges) by applying the LM successively via
  > beam search using `beam_width` (keeping a traceback of LM state
  > history at each node, passing and updating LM state explicitly). As
  > in the above trivial case without `alternative_decoding`, then
  > combine LM scores weighted by `lm_weight` with input confidence on
  > the graph's edges. Also, prune worst paths and apply LM state
  > history clustering to avoid expanding all possible combinations.
  > Finally, look into the current best overall path, traversing back to
  > the last node of the previous page's graph. Lock into that node by
  > removing all current paths that do not derive from it, and making
  > its history path the final decision for the previous page: Apply
  > that path by removing all but the chosen TextEquiv alternatives,
  > assigning the resulting confidences, and making the levels above
  > `textequiv_level` consistent with that textual result (via
  > concatenation joined by whitespace). Also, calculate the overall
  > average LM probability, and the character and segment-level
  > perplexity, and print it on the logger. Moreover, at the last page
  > at the end of the document, lock into the current best path
  > analogously.

  > Produce new output files by serialising the resulting hierarchy for
  > each page.

Subcommands:
    worker      Start a processing worker rather than do local processing
    server      Start a processor server rather than do local processing

Options for processing:
  -m, --mets URL-PATH             URL or file path of METS to process [./mets.xml]
  -w, --working-dir PATH          Working directory of local workspace [dirname(URL-PATH)]
  -I, --input-file-grp USE        File group(s) used as input
  -O, --output-file-grp USE       File group(s) used as output
  -g, --page-id ID                Physical page ID(s) to process instead of full document []
  --overwrite                     Remove existing output pages/images
                                  (with "--page-id", remove only those)
  --profile                       Enable profiling
  --profile-file PROF-PATH        Write cProfile stats to PROF-PATH. Implies "--profile"
  -p, --parameter JSON-PATH       Parameters, either verbatim JSON string
                                  or JSON file path
  -P, --param-override KEY VAL    Override a single JSON object key-value pair,
                                  taking precedence over --parameter
  -U, --mets-server-url URL       URL of a METS Server for parallel incremental access to METS
                                  If URL starts with http:// start an HTTP server there,
                                  otherwise URL is a path to an on-demand-created unix socket
  -l, --log-level [OFF|ERROR|WARN|INFO|DEBUG|TRACE]
                                  Override log level globally [INFO]

Options for information:
  -C, --show-resource RESNAME     Dump the content of processor resource RESNAME
  -L, --list-resources            List names of processor resources
  -J, --dump-json                 Dump tool description as JSON
  -D, --dump-module-dir           Show the 'module' resource location path for this processor
  -h, --help                      Show this message
  -V, --version                   Show version

Parameters:
   "model_file" [string - REQUIRED]
    path of h5py weight/config file for model trained with keraslm
   "textequiv_level" [string - "glyph"]
    PAGE XML hierarchy level to evaluate TextEquiv sequences on
    Possible values: ["region", "line", "word", "glyph"]
   "alternative_decoding" [boolean - true]
    whether to process all TextEquiv alternatives, finding the best path
    via beam search, and delete each non-best alternative
   "beam_width" [number - 10]
    maximum number of best partial paths to consider during search with
    alternative_decoding
   "lm_weight" [number - 0.5]
    share of the LM scores over the input confidences
```

Examples:
```shell
make deps-test # installs ocrd_tesserocr
make test/assets # downloads GT, imports PageXML, builds workspaces
ocrd workspace -d ws1 clone -a test/assets/kant_aufklaerung_1784/mets.xml
cd ws1
ocrd-tesserocr-segment-region -I OCR-D-IMG -O OCR-D-SEG-BLOCK
ocrd-tesserocr-segment-line -I OCR-D-SEG-BLOCK -O OCR-D-SEG-LINE
ocrd-tesserocr-recognize -I OCR-D-SEG-LINE -O OCR-D-OCR-TESS-WORD -P textequiv_level word -P model Fraktur
ocrd-tesserocr-recognize -I OCR-D-SEG-LINE -O OCR-D-OCR-TESS-GLYPH -P textequiv_level glyph -P model deu-frak
# download Deutsches Textarchiv language model
ocrd resmgr download ocrd-keraslm-rate model_dta_full.h5
# get confidences and perplexity:
ocrd-keraslm-rate -I OCR-D-OCR-TESS-WORD -O OCR-D-OCR-LM-WORD -P model_file model_dta_full.h5 -P textequiv_level word -P alternative_decoding false
# also get best path:
ocrd-keraslm-rate -I OCR-D-OCR-TESS-GLYPH -O OCR-D-OCR-LM-GLYPH -P model_file model_dta_full.h5 -P textequiv_level glyph -P alternative_decoding true -P beam_width 10
```

### Models

Pretrained models will be published under [Github release assets](https://github.com/OCR-D/ocrd_keraslm/releases)
and made visible via [OCR-D Resource Manager](https://ocr-d.de/en/models).

So far, the only published models are:

- [model_dta_full.h5](https://github.com/OCR-D/ocrd_keraslm/releases/download/v0.4.3/model_dta_full.h5)  
  This LM was configured as stateful contiguous LSTM model (2 layers, 128 hidden nodes each, window length 256),
  and trained on the complete [Deutsches Textarchiv](https://deutsches-textarchiv.de/) fulltext (80%/20% split).  
  It achieves a perplexity of 2.51 on the validation subset after 4 epochs.

## Testing

```shell
make deps-test test
```
Which is the equivalent of:
```shell
pip install -r requirements_test.txt
test -e test/assets || test/prepare_gt.bash test/assets
test -f model_dta_test.h5 || keraslm-rate train -m model_dta_test.h5 test/assets/*.txt
keraslm-rate test -m model_dta_test.h5 test/assets/*.txt
python -m pytest test $(PYTEST_ARGS)
```

Set `PYTEST_ARGS="-s --verbose"` to see log output (`-s`) and individual test results (`--verbose`).
