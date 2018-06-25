# -*- coding: utf-8 -*-

import click, sys, numpy, pickle
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences

@click.group()
def cli():
    pass

@cli.command()
@click.option('-t', '--training', required=True)
@click.option('-o', '--output', default="model")
def train(training, output):
    """Train a language model"""

    #
    # Step 1: read and encode training data
    #
    with open(str(training),"r") as f:
        raw_data = f.read()

        #
        # vocabulary
        chars = sorted(list(set(raw_data)))
        c_i = dict((c, i) for i, c in enumerate(chars))
        i_c = dict((i, c) for i, c in enumerate(chars))
        # vocabulary size
        vocab_size = len(c_i)
        click.echo(u'Vocabulary Size: %d' % vocab_size)

        length = 5
        step = 3
        sequences = []
        next_chars = []
        with click.progressbar(range(0, len(raw_data) - length, step)) as bar:
            for i in bar:
                sequences.append(raw_data[i: i + length])
                next_chars.append(raw_data[i + length])
        click.echo(u'Sequences Size: %d' % len(sequences))

        x = numpy.zeros((len(sequences), length, vocab_size), dtype=numpy.bool)
        y = numpy.zeros((len(sequences), vocab_size), dtype=numpy.bool)

        with click.progressbar(enumerate(sequences)) as bar:
            for i, sequence in bar:
                for t, char in enumerate(sequence):
                    x[i, t, c_i[char]] = 1
                    y[i, c_i[next_chars[i]]] = 1

    # define model
    model = Sequential()
    model.add(LSTM(128, input_shape=(length, vocab_size)))
    model.add(Dense(vocab_size, activation='softmax'))
    click.echo(model.summary())

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit model
    model.fit(x, y, batch_size=128, epochs=100, verbose=2)

    # save model and dicts
    model.save(u"%s.h5" % output)
    pickle.dump((c_i,i_c), open(u"%s.map" % output, mode='wb'))

@cli.command()
@click.option('-m', '--model', required=True)
@click.option('-M', '--mapping', required=True)
@click.argument('TEXT')
def apply(model, mapping, text):
    """Apply a language model"""

    # load model
    mdl = load_model(model)

    # load mapping
    c_i, i_c = pickle.load(open(mapping, mode="rb"))
    vocab_size = len(c_i)
    length = 5 

    # read input
    in_string = u""
    if text:
        if text[0] == u"-":
            text = sys.stdin
        for line in text:
            in_string += line
    else:
        pass

    encoded_buffer = []
    for c in in_string:
        x = numpy.zeros((1, length, vocab_size))
        encoded = pad_sequences([encoded_buffer], maxlen=length, truncating='pre')
        for t, i in enumerate(encoded[0]):
            x[0, t, i] = 1.
        pred = dict(enumerate(mdl.predict(x, verbose=0).tolist()[0]))
        i = c_i[c]
        click.echo("%s: %.8f" % (c, pred[i]))
        encoded_buffer.append(i)

if __name__ == '__main__':
    cli()
