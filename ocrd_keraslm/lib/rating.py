from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping

import click, numpy, pickle
from math import log, exp, ceil

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
        self.minibatch_size = 128
        self.validation_split = 0.2
    
    def train(self, training_data):
        '''
        Trains an RNN language model on `training_data` files (UTF-8 byte sequences).
        '''
        if self.status != 0:
            self.clear()

        total_size = 0
        max_size = 0
        chars = set([])
        with click.progressbar(training_data) as bar:
            for f in bar:
                text = f.read()
                size = len(text)
                total_size += size
                max_size = max(max_size, size)

        steps = 3
        epoch_size = total_size/steps/self.minibatch_size
        training_epoch_size = ceil(epoch_size*(1-self.validation_split))
        validation_epoch_size = ceil(epoch_size*self.validation_split)
        
        #
        # data preparation

        split = numpy.random.uniform(0,1, (ceil(max_size/steps),))
        def gen_data(files, train):
            # encode
            while True:
                for f in files:
                    f.seek(0)
                    text = f.read()
                    sequences = []
                    next_chars = []
                    for i in range(0, len(text) - self.length, steps):
                        if (split[int(i/steps)]<self.validation_split) == train:
                            continue
                        sequences.append(text[i: i + self.length])
                        next_chars.append(text[i + self.length])
                        if (len(sequences) % self.minibatch_size == 0 or 
                            i + steps > len(text) - self.length): # last minibatch
                            # vectorization
                            x = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(list(map(bytearray,sequences)), dtype=numpy.uint8)]
                            y = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(bytearray(next_chars), dtype=numpy.uint8)]
                            yield (x,y)
                            sequences = []
                            next_chars = []
        
        #
        # model
        
        # define model # todo: automatically switch to CuDNNLSTM if CUDA GPU is available
        self.model.add(LSTM(128, input_shape=(self.length, 256), return_sequences=True))
        self.model.add(LSTM(128))
        self.model.add(Dense(256, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # fit model
        early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
        self.model.fit_generator(gen_data(training_data, True), steps_per_epoch=training_epoch_size, epochs=100, verbose=1, validation_data=gen_data(training_data, False), validation_steps=validation_epoch_size, callbacks=[early_stopping]) # todo: make iterator thread-safe and use_multiprocesing=True
        
        # set state
        self.status = 1
    
    def save(self, prefix):
        '''
        Saves model.
        '''
        
        if self.status:
            self.model.save(u"%s.h5" % prefix)
    
    def load(self, model):
        '''
        Loads model.
        '''
        
        # load model
        self.model = load_model(model)
        self.status = 1
    
    def rate(self, text):
        '''
        Calculates probabilities (individually) and perplexity (accumulated)
        of the character sequence in `text` according to the current model.
        '''

        # perplexity calculation is a lot slower that way than via tensor metric, cf. test()
        # todo: allow graph input (by pruning via history clustering or push forward algorithm;
        #                       or by aggregating lattice input)
        # todo: make incremental
        if self.status:
            x = numpy.zeros((1, self.length, 256))
            entropy = 0
            result = []
            length = 0
            for c in text:
                p = 1.0
                for b in c.encode("utf-8"):
                    pred = dict(enumerate(self.model.predict(x, verbose=0).tolist()[0]))
                    entropy -= log(pred[b], 2)
                    length += 1
                    p *= pred[b]
                    x = numpy.roll(x, -1, axis=1) # left-shifted by 1
                    x[0,-1] = numpy.eye(256)[b] # one-hot vector for b in last pos
                result.append((c, p))
            return result, pow(2.0, entropy/length)
        else:
            return [], 0
    
    def test(self, test_data):
        '''
        Calculates the perplexity of the character sequences in all `test_data` files
        (UTF-8 byte sequences) according to the current model.
        '''

        total_size = 0
        with click.progressbar(test_data) as bar:
            for f in bar:
                text = f.read()
                size = len(text)
                total_size += size
        steps = 1
        epoch_size = ceil(total_size/self.minibatch_size)

        # data preparation
        def gen_data(files):
            # encode
            while True:
                for f in files:
                    f.seek(0)
                    text = f.read()
                    sequences = []
                    next_chars = []
                    for i in range(0, len(text) - self.length, steps):
                        sequences.append(text[i: i + self.length])
                        next_chars.append(text[i + self.length])
                        if (len(sequences) % self.minibatch_size == 0 or 
                            i + steps > len(text) - self.length): # last minibatch
                            # vectorization
                            x = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(list(map(bytearray,sequences)), dtype=numpy.uint8)]
                            y = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(bytearray(next_chars), dtype=numpy.uint8)]
                            yield (x,y)
                            sequences = []
                            next_chars = []
        
        if self.status:
            loss, accuracy = self.model.evaluate_generator(gen_data(test_data), steps=epoch_size, verbose=1) # todo: make iterator thread-safe and use_multiprocesing=True
            return exp(loss)
        else:
            return 0
        
