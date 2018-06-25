from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences

import click, numpy, pickle

class Rater(object):

    def __init__(self):
        '''
        The constructor.
        '''

        self.clear()

    def clear(self):
        '''
        Resets rater.
        '''

        self.model = Sequential()
        self.status = 0
        self.length = 5

    def train(self, training_data):
        '''
        Trains an RNN model.
        '''
        if self.status != 0:
            self.clear()

        #
        # mapping
        chars = sorted(list(set(training_data)))
        c_i = dict((c, i) for i, c in enumerate(chars))
        i_c = dict((i, c) for i, c in enumerate(chars))
        self.mapping = (c_i, i_c)


        #
        # data preparation

        # encode
        step = 3
        sequences = []
        next_chars = []
        with click.progressbar(range(0, len(training_data) - self.length, step)) as bar:
            for i in bar:
                sequences.append(training_data[i: i + self.length])
                next_chars.append(training_data[i + self.length])

        # vectorization
        x = numpy.zeros((len(sequences), self.length, len(self.mapping[0])), dtype=numpy.bool)
        y = numpy.zeros((len(sequences), len(self.mapping[0])), dtype=numpy.bool)
        with click.progressbar(enumerate(sequences)) as bar:
            for i, sequence in bar:
                for t, char in enumerate(sequence):
                    x[i, t, c_i[char]] = 1
                    y[i, c_i[next_chars[i]]] = 1

        #
        # model

        # define model
        self.model.add(LSTM(128, input_shape=(5, len(self.mapping[0]))))
        self.model.add(Dense(len(self.mapping[0]), activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # fit model
        self.model.fit(x, y, batch_size=128, epochs=100, verbose=2)

        # set state
        self.status = 1

    def save(self, prefix):
        '''
        Saves model and mapping.
        '''
        
        if self.status:
            self.model.save(u"%s.h5" % prefix)
            pickle.dump(self.mapping, open(u"%s.map" % prefix, mode='wb'))

    def load(self, mapping, model):
        '''
        Loads model and mapping.
        '''

        # load model
        self.model = load_model(model)
        self.mapping = pickle.load(open(mapping, mode="rb"))
        self.status = 1

    def rate(self, text):
        '''
        Rates the characters in text according to the model.
        '''

        if self.status:
            encoded_buffer = []
            result = []
            for c in text:
                x = numpy.zeros((1, self.length, len(self.mapping[0])))
                encoded = pad_sequences([encoded_buffer], maxlen=self.length, truncating='pre')
                for t, i in enumerate(encoded[0]):
                    x[0, t, i] = 1.
                pred = dict(enumerate(self.model.predict(x, verbose=0).tolist()[0]))
                i = self.mapping[0][c]
                result.append((c, pred[i]))
                encoded_buffer.append(i)
            return result
        else:
            return []

    def rate_single(self, text):
        '''
        Rates the last character in text according to the model.
        '''

        if self.status :
            try:
                encoded_buffer = [self.mapping[0][c] for c in text[:-1]]
                x = numpy.zeros((1, self.length, len(self.mapping[0])))
                encoded = pad_sequences([encoded_buffer], maxlen=self.length, truncating='pre')
                for t, i in enumerate(encoded[0]):
                    x[0, t, i] = 1.
                pred = dict(enumerate(self.model.predict(x, verbose=0).tolist()[0]))
                i = self.mapping[0][text[-1]]
                return pred[i]
            except:
                return 0.0
        else:
            return 0.0
