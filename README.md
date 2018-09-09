# ocrd_keraslm
    Simple character-based language model using Keras


## Introduction

This is a tool for statistical _language modelling_ (predicting text from context) with recurrent neural networks. It models probabilities not on the word level but the UTF-8 _byte level_. That way, there is no fixed vocabulary of known/allowed words/characters, and no word segmentation ambiguity. 

### Architecture

The model consists of:
1. no embedding layer: input bytes are represented as unit vectors ("one-hot coding") in 256 dimensions, in windows of a number `length` of bytes,
2. a number `depth` of hidden layers, each with a number `width` of hidden recurrent units of _LSTM cells_ (Long Short-term Memory),
3. a softmax output layer, returning a probability for each possible value of the next byte, respectively.

This is implemented in [Keras](https://keras.io) with [Tensorflow](https://www.tensorflow.org/) as backend. It automatically uses a fast CUDA-optimized LSTM implementation if available (Nividia GPU and Tensorflow installation with GPU support, see below), both in learning and in prediction phase. 


### Modes of operation

Notably, this model (by default) runs _statefully_, i.e. by implicitly passing hidden state from one window (batch) to the next. That way, the context available for predictions can be arbitrarily long (above `length`, e.g. the complete document up to that point), or short (below `length`, e.g. at the start of a text). (However, this is a passive perspective above `length`, because errors are never back-propagated any further in time during gradient-descent training.) 

Besides stateful mode, the model can also be run _incrementally_, i.e. by explicitly passing hidden state from the caller. That way, multiple alternative hypotheses can be processed together. This is used for generation (sampling from the model) and alternative decoding (finding the best path through a sequence of alternatives).


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
  apply     get individual probabilities from language model
  generate  sample characters from language model
  test      get overall perplexity from language model
  train     train a language model
```

Examples:
```shell
keraslm-rate train --width 64 --depth 4 --length 256 --model model_dta_64_4_256.weights.h5 --config model_dta_64_4_256.config.pkl dta_komplett_2017-09-01/txt/*.tcf.txt
keraslm-rate generate -m model_dta_64_4_256.weights.h5 -c model_dta_64_4_256.config.pkl --number 6 "für die Wiſſen"
keraslm-rate apply -m model_dta_64_4_256.weights.h5 -c model_dta_64_4_256.config.pkl "so schädlich ist es Borkickheile zu pflanzen"
keraslm-rate test -m model_dta_64_4_256.weights.h5 -c model_dta_64_4_256.config.pkl dta_komplett_2017-09-01/txt/grimm_*.tcf.txt
```

### [OCR-D processor](https://github.com/OCR-D/core) interface `ocrd-keraslm-rate`

To be used with [PageXML](https://www.primaresearch.org/tools/PAGELibraries) documents in an [OCR-D](https://github.com/OCR-D/spec/) annotation workflow.

```json
  "tools": {
    "ocrd-keraslm-rate": {
      "executable": "ocrd-keraslm-rate",
      "categories": [
        "Text recognition and optimization"
      ],
      "steps": [
        "recognition/text-recognition"
      ],
      "description": "Rate each element of the text with a byte-level LSTM language model in Keras",
      "parameters": {
        "weight_file": {
          "type": "string",
          "description": "path of h5 weight file for model trained with keraslm",
          "required": true
        },
        "config_file": {
          "type": "string",
          "description": "path of pkl config file for model trained with keraslm",
          "required": true
        },
        "textequiv_level": {
          "type": "string",
          "enum": ["region", "line", "word", "glyph"],
          "default": "glyph",
          "description": "PAGE XML hierarchy level to evaluate TextEquiv sequences on"
        },
        "add_space_glyphs": {
          "type": "boolean",
          "description": "whether to insert whitespace and newline as pseudo-glyphs in result (at glyph level)",
          "default": false
        },
        "alternative_decoding": {
          "type": "boolean",
          "description": "whether to process all TextEquiv alternatives, finding the best path via beam search, and delete each non-best alternative",
          "default": true
        },
        "beam_width": {
          "type": "number",
          "format": "integer",
          "description": "maximum number of best partial paths to consider during search with alternative_decoding",
          "default": 100
        }
      }
    }
  }
```

Examples:
```shell
make deps-test # installs ocrd_tesserocr
make test/assets # downloads GT, imports PageXML, builds workspaces
ocrd workspace clone -a -l kant_aufklaerung_1784/mets.xml ws1
cd ws1
ocrd-tesserocr-segment-region -I OCR-D-IMG -O OCR-D-SEG-BLOCK -p <(echo "{}")
ocrd-tesserocr-segment-line -I OCR-D-SEG-BLOCK -O OCR-D-SEG-LINE -p <(echo "{}")
cat <<EOF > param-tess-word.json
{
  "textequiv_level" : "word",
  "model" : "Fraktur"
}
EOF
cat <<EOF > param-tess-glyph.json
{
  "textequiv_level" : "glyph",
  "model" : "deu-frak"
}
EOF
ocrd-tesserocr-recognize -I OCR-D-SEG-LINE -O OCR-D-OCR-TESS-WORD -p param-tess-word.json
ocrd-tesserocr-recognize -I OCR-D-SEG-LINE -O OCR-D-OCR-TESS-GLYPH -p param-tess-glyph.json
cat <<EOF > param-lm-word-1.json
{
    "weight_file": "model_dta_64_4_256.weights.h5",
    "config_file": "model_dta_64_4_256.config.pkl",
    "textequiv_level": "word",
    "add_space_glyphs": false,
    "alternative_decoding": false
}
EOF
cat <<EOF > param-lm-glyph-all.json
{
    "weight_file": "model_dta_64_4_256.weights.h5",
    "config_file": "model_dta_64_4_256.config.pkl",
    "textequiv_level": "glyph",
    "add_space_glyphs": true,
    "alternative_decoding": true,
    "beam_width": 10
}
EOF
ocrd-keraslm-rate -I OCR-D-OCR-TESS-WORD -O OCR-D-OCR-LM-WORD -p param-lm-word-1.json # get confidences and perplexity
ocrd-keraslm-rate -I OCR-D-OCR-TESS-GLYPH -O OCR-D-OCR-LM-GLYPH -p param-lm-glyph-all.json # also get best path
```

## Testing

```shell
make deps-test test
```
Which is the equivalent of:
```shell
pip install -r requirements_test.txt
test -e test/assets || test/prepare_gt.bash test/assets
test -f model_dta_test.weights.h5 -a -f model_dta_test.config.pkl || keraslm-rate train -m model_dta_test.weights.h5 -c model_dta_test.config.pkl test/assets/*.txt
keraslm-rate test -m model_dta_test.weights.h5 -c model_dta_test.config.pkl test/assets/*.txt
python -m pytest test $(PYTEST_ARGS)
```

Set `PYTEST_ARGS="-s --verbose"` to see log output (`-s`) and individual test results (`--verbose`).
