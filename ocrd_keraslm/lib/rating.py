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
        
        from keras.models import Sequential

        self.model = Sequential()
        self.status = 0
        self.length = 15
        self.variable_length = False # also train on partially filled windows
        self.minibatch_size = 128
        self.validation_split = 0.2
    
    def train(self, training_data, width, depth, length):
        '''
        Trains an RNN language model on `training_data` files (UTF-8 byte sequences).
        '''
        
        from keras.layers import Dense
        from keras.layers import LSTM, CuDNNLSTM
        from keras.callbacks import EarlyStopping
        from keras import backend as K
        
        if self.status != 0:
            self.clear()
        self.length = length # now from CLI

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
        if self.variable_length:
            training_epoch_size *= ceil((self.length-1)/steps) # training data augmented with partial windows
        
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
                            if train and self.variable_length: # also train on partial windows?
                                for j in range(1,self.length-1, steps):
                                    #x[:, 0:j, :] = False # complete batch gets erased by sublength from the left to simulate running in with zero padding as in rate()
                                    #yield (x,y)
                                    yield (x[:,-j:,:],y) # complete batch gets shortened to sublength from the right to simulate running in with short sequences in rate()
                            sequences = []
                            next_chars = []
        
        #
        # model
        
        # define model
        # width = 128 # now from CLI
        # depth = 2 # now from CLI
        length = None if self.variable_length else self.length
        # automatically switch to CuDNNLSTM if CUDA GPU is available:
        has_cuda = K.backend() == 'tensorflow' and K.tensorflow_backend._get_available_gpus()
        print('using', 'GPU' if has_cuda else 'CPU', 'LSTM implementation to compile model of depth',depth,'width',width,'length',self.length)
        lstm = CuDNNLSTM if has_cuda else LSTM # todo: do not use save/load_model but save_weights/load_weights (so GPU and CPU-only hosts can share models) with extra pickle file for configuration
        for i in range(depth):
            args = {'return_sequences': (i+1<depth)}
            if not has_cuda:
                args['activation'] = 'sigmoid' # instead of default 'hard_sigmoid' which deviates from CuDNNLSTM
            if i == 0:
                args['input_shape'] = (length, 256)
            self.model.add(lstm(width,
                                **args))
        self.model.add(Dense(256, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # fit model
        early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
        self.model.fit_generator(gen_data(training_data, True), steps_per_epoch=training_epoch_size, epochs=100, verbose=1, validation_data=gen_data(training_data, False), validation_steps=validation_epoch_size, callbacks=[early_stopping]) # todo: make iterator thread-safe and use_multiprocesing=True
        
        # set state
        self.status = 1
    
    def save(self, filename):
        '''
        Saves model.
        '''
        
        if self.status:
            self.model.save(filename)
    
    def load(self, filename):
        '''
        Loads model.
        '''

        from keras.models import load_model

        # load model
        self.model = load_model(filename)
        # get parameters from model which are relevant to preprocessing
        batch_input_shape = self.model.get_config()[0]['config']['batch_input_shape']
        if batch_input_shape[0] and batch_input_shape[0] != self.minibatch_size:
            print('overriding minibatch_size %d by %d from saved model' % (self.minibatch_size, batch_input_shape[0]))
            self.minibatch_size = batch_input_shape[0]
        if batch_input_shape[1] and batch_input_shape[1] != self.length:
            print('overriding length %d by %d from saved model' % (self.length, batch_input_shape[1]))
            self.length = batch_input_shape[1]
            self.variable_length = False
        if not batch_input_shape[1] and not self.variable_length:
            print('overriding variable length from saved model (representation might be suboptimal)')
            self.variable_length = True
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
            x = numpy.zeros((1, self.length, 256), dtype=numpy.bool)
            entropy = 0
            result = []
            length = 0
            for c in text:
                p = 1.0
                for b in c.encode("utf-8"):
                    x_input = x[:,x.any(axis=2)[0]] if self.variable_length else x
                    if x_input.shape[1] > 0: # to prevent length 0 input
                        pred = dict(enumerate(self.model.predict(x_input, verbose=0).tolist()[0]))
                        entropy -= log(pred[b], 2)
                        length += 1
                        p *= pred[b]
                    x = numpy.roll(x, -1, axis=1) # left-shifted by 1
                    x[0,-1] = numpy.eye(256, dtype=numpy.bool)[b] # one-hot vector for b in last pos
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
        
