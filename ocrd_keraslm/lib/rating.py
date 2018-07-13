from keras.callbacks import Callback
import click, numpy, pickle
from random import shuffle
from math import log, exp, ceil, floor

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
        self.status = 0 # trained?
        
        self.length = 0 # will be overwritten by CLI for train / by load model for rate/test
        self.width = 0 # will be overwritten by CLI for train / by load model for rate/test
        self.depth = 0 # will be overwritten by CLI for train / by load model for rate/test
        self.length = 0 # will be overwritten by CLI for train / by load model for rate/test
        
        self.variable_length = False # also train on partially filled windows
        self.stateful = False # keep states across batches within one text
        self.minibatch_size = 128 # will be overwritten by length if stateful
        self.validation_split = 0.2 # fraction of training data to use for validation (generalization control)
    
    def configure(self):
        from keras.layers import Dense
        from keras.layers import LSTM, CuDNNLSTM
        from keras import backend as K
        
        #
        # model
        length = None if self.variable_length else self.length
        # automatically switch to CuDNNLSTM if CUDA GPU is available:
        has_cuda = K.backend() == 'tensorflow' and K.tensorflow_backend._get_available_gpus()
        print('using', 'GPU' if has_cuda else 'CPU', 'LSTM implementation to compile',
              'stateful' if self.stateful else 'stateless', 'model of depth',
              self.depth, 'width', self.width,'length', self.length)
        lstm = CuDNNLSTM if has_cuda else LSTM
        for i in range(self.depth):
            args = {'return_sequences': (i+1 < self.depth), 'stateful': self.stateful}
            if not has_cuda:
                args['recurrent_activation'] = 'sigmoid' # instead of default 'hard_sigmoid' which deviates from CuDNNLSTM
            if i == 0:
                if self.stateful:
                    args['batch_input_shape'] = (self.minibatch_size, length, 256) # batch size must be constant
                else:
                    args['input_shape'] = (length, 256) # batch size not fixed (e.g. different between training and prediction)
            self.model.add(lstm(self.width,
                                **args))
        self.model.add(Dense(256, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    def train(self, data, width, depth, length):
        '''
        Trains an RNN language model on `data` files (UTF-8 byte sequences).
        '''
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        
        if self.status != 0:
            self.clear()
        self.width = width
        self.depth = depth
        self.length = length
        if self.stateful:
            self.minibatch_size = self.length # make sure states are consistent with windows after 1 minibatch
        
        self.configure()
        
        data = list(data)
        shuffle(data) # random order of files (because generators cannot shuffle within files)
        if self.stateful: # we must split file-wise in stateful mode
            steps = 1 # really necessary?
            split = ceil(len(data)*self.validation_split) # split position in randomized file list
            training_data, validation_data = data[:-split], data[-split:] # reserve last files for validation
            for f in validation_data:
                print ('using input', f.name, 'for validation only')
            training_epoch_size = 0
            for f in training_data:
                text = f.read()
                training_epoch_size += floor(len(text)/steps/self.minibatch_size)
            validation_epoch_size = 0
            for f in validation_data:
                text = f.read()
                validation_epoch_size += floor(len(text)/steps/self.minibatch_size)
            reset_cb = ResetStatesCallback()
        else: # we can split window by window in stateless mode
            steps = 3
            total_size = 0
            max_size = 0
            with click.progressbar(data) as bar:
                for f in bar:
                    text = f.read()
                    size = len(text)
                    total_size += size
                    max_size = max(max_size, size)
            epoch_size = total_size/steps/self.minibatch_size
            training_epoch_size = ceil(epoch_size*(1-self.validation_split))
            validation_epoch_size = ceil(epoch_size*self.validation_split)
            if self.variable_length:
                training_epoch_size *= ceil((self.length-1)/steps) # training data augmented with partial windows
            validation_data, training_data = data, data # same data, different generators (see below)
            split = numpy.random.uniform(0,1, (ceil(max_size/steps),)) # reserve split fraction at random positions
        
        #
        # data preparation
        def gen_data(files, train):
            while True:
                for f in files:
                    f.seek(0)
                    if self.stateful:
                        reset_cb.reset()
                    text = f.read()
                    # encode
                    sequences = []
                    next_chars = []
                    for i in range(0, len(text) - self.length, steps):
                        if not self.stateful and (split[int(i/steps)]<self.validation_split) == train:
                            continue # data shared between training and split: belongs to other generator
                        sequences.append(text[i: i + self.length])
                        next_chars.append(text[i + self.length])
                        if (len(sequences) % self.minibatch_size == 0 # next minibatch full
                            or i + steps > len(text) - self.length): # last minibatch
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
        
        
        # fit model
        callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=1),
                     ModelCheckpoint('model_last.weights.h5', monitor='val_loss', # to be able to replay long epochs (crash/overfitting)
                                     save_best_only=True, save_weights_only=True, mode='min')]
        if self.stateful:
            callbacks.append(reset_cb)
        self.model.fit_generator(gen_data(training_data, True), steps_per_epoch=training_epoch_size, epochs=100, 
                                 validation_data=gen_data(validation_data, False), validation_steps=validation_epoch_size,
                                 verbose=1, callbacks=callbacks) # todo: make iterator thread-safe and use_multiprocesing=True
        
        # set state
        self.status = 1
    
    def save(self, filename):
        '''
        Saves model into `filename`.
        (Cannot preserve weights across CPU/GPU implementations or input shape configurations.)
        '''
        
        if self.status:
            self.model.save(filename)
    
    def load(self, filename):
        '''
        Loads model from `filename`.
        (Cannot preserve weights across CPU/GPU implementations or input shape configurations.)
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

    def save2(self, configfilename, weightfilename):
        '''
        Saves model configuration into `configfilename` and model weights into `weightfilename`.
        (This preserves weights across CPU/GPU implementations or input shape configurations.)
        '''
        
        if self.status:
            config = {'width': self.width, 'depth': self.depth, 'length': self.length, 'stateful': self.stateful, 'variable_length': self.variable_length}
            pickle.dump(config, open(configfilename, mode='wb'))
            self.model.save_weights(weightfilename)
    
    def load2(self, configfilename, weightfilename):
        '''
        Loads model configuration from `configfilename`, compiles a new model from that, then loads weights into it from `weightfilename`.
        (This preserves weights across CPU/GPU implementations or input shape configurations.)
        '''
        
        if self.status != 0:
            self.clear()
        config = pickle.load(open(configfilename, mode='rb'))
        self.width = config['width']
        self.depth = config['depth']
        self.length = config['length']
        self.stateful = config['stateful']
        self.variable_length = config['variable_length']
        self.configure()
        
        self.model.load_weights(weightfilename)
        
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
            self.model.reset_states()
            for c in text: # could be single characters or words later-on (when applying incrementally or from graph)
                p = 1.0
                for b in c.encode("utf-8"):
                    x_input = x[:,x.any(axis=2)[0]] if self.variable_length else x
                    if x_input.shape[1] > 0: # to prevent length 0 input
                        pred = dict(enumerate(self.model.predict(x_input, batch_size=1, verbose=0).tolist()[0]))
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
        
        # todo: Since Keras does not allow callbacks within evaluate() / evaluate_generator() / test_loop(),
        #       we cannot reset_states() between input files as we do in train().
        #       Thus we should evaluate each file individually, reset in between, and accumulate losses.
        #       But this looks awkward, since we get N progress bars instead of 1, in contrast to training.
        #       Perhaps the overall error introduced into stateful models by not resetting is not that high
        #       after all?
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

class ResetStatesCallback(Callback):
    '''Callback to be called by `fit_generator()` or even `evaluate_generator()`:

       do `model.reset_states()` whenever generator sees EOF (on_batch_begin with self.eof),
       and between training and validation (on_batch_end with batch>=steps_per_epoch-1)
    '''
    def __init__(self):
        self.eof = False
    
    def reset(self):
        self.eof = True
    
    def on_batch_begin(self, batch, logs={}):
        if self.eof:
            # between training files
            self.model.reset_states()
            self.eof = False
    
    def on_batch_end(self, batch, logs={}):
        if (self.params['do_validation'] and batch >= self.params['steps']-1):
            # in fit_generator just before evaluate_generator
            self.model.reset_states()

