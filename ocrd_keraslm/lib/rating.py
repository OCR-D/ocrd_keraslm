from keras.callbacks import Callback
import click, numpy, pickle, codecs
from random import shuffle
from math import log, exp, ceil, floor

class Rater(object):
    '''A character-level RNN language model for rating text.
    
    Uses Keras to define, compile, train, run and test an RNN
    (LSTM) language model on the (UTF-8) byte level. The model's
    topology (layers depth, per-layer width, window length) can
    be controlled before training.

    To be used by stand-alone CLI (`scripts.train` for training,
    `scripts.apply` for prediction, `scripts.test` for evaluation),
    or OCR-D processing (`wrapper.ocrd_keraslm_rate`). 

    Interfaces:
    - `Rater.train`/`scripts.train` : file handles of byte sequences
    - `Rater.test`/`scripts.test` : file handles of byte sequences
    - `Rater.rate`/`scripts.apply` : character string
    - `Rater.rate_once`/`wrapper.ocrd_keraslm_rate` : character string
    - `Rater.rate_single`/`scripts.generate` : alternative list of bytes and states
    '''
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        '''Reset model and set all parameters to their defaults.'''
        
        self.model = None
        self.status = 0 # empty / compiled / trained?
        
        self.length = 0 # will be overwritten by CLI for train / by load model for rate/test
        self.width = 0 # will be overwritten by CLI for train / by load model for rate/test
        self.depth = 0 # will be overwritten by CLI for train / by load model for rate/test
        self.length = 0 # will be overwritten by CLI for train / by load model for rate/test
        
        self.variable_length = False # also train on partially filled windows
        self.stateful = True # keep states across batches within one text (implicit state transfer)
        self.minibatch_size = 128 # will be overwritten by length if stateful
        self.validation_split = 0.2 # fraction of training data to use for validation (generalization control)
        
        self.incremental = False # whether compiled with additional (initial) input state and (final) output state (explicit state transfer)
    
    def configure(self):
        '''Define and compile model for the given parameters.'''
        from keras.layers import Dense, TimeDistributed, Input
        from keras.layers import LSTM, CuDNNLSTM
        from keras import backend as K
        from keras.models import Model

        length = None if self.variable_length else self.length
        # automatically switch to CuDNNLSTM if CUDA GPU is available:
        has_cuda = K.backend() == 'tensorflow' and K.tensorflow_backend._get_available_gpus()
        print('using', 'GPU' if has_cuda else 'CPU', 'LSTM implementation to compile',
              'stateful' if self.stateful else 'stateless',
              'incremental' if self.incremental else 'contiguous',
              'model of depth', self.depth, 'width', self.width, 'length', self.length)
        lstm = CuDNNLSTM if has_cuda else LSTM
        if self.stateful:
            self.minibatch_size = 1
            input_args = {'batch_shape': (self.minibatch_size, None, 256)} # batch size must be constant, variable length
        elif self.incremental:
            states_input_args = {'shape': (self.width,)}
            model_states_input = []
            model_states_output = []
            input_args = {'shape': (1, 256)} # batch size not fixed
        else:
            input_args = {'shape': (length, 256)} # batch size not fixed (e.g. different between training and prediction)
        model_input = Input(**input_args)
        model_output = model_input # init layer loop
        for i in range(self.depth): # layer loop
            args = {'return_sequences': (i+1 < self.depth) or self.stateful, 'stateful': self.stateful}
            if not has_cuda:
                args['recurrent_activation'] = 'sigmoid' # instead of default 'hard_sigmoid' which deviates from CuDNNLSTM
            if self.incremental:
                # incremental prediction needs additional inputs and outputs for state (h,c):
                states = [Input(**states_input_args), Input(**states_input_args)]
                model_states_input.extend(states)
                model_output, state_h, state_c = lstm(self.width, return_state = True, **args)(model_output, initial_state = states)
                model_states_output.extend([state_h, state_c])
            else:
                model_output = lstm(self.width, **args)(model_output)
        if self.stateful:
            model_output = TimeDistributed(Dense(256, activation='softmax'))(model_output)
        else:
            model_output = Dense(256, activation='softmax')(model_output)
        if self.incremental:
            self.model = Model([model_input] + model_states_input, [model_output] + model_states_output)
        else:
            self.model = Model(model_input, model_output)            
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.status = 1
    
    def train(self, data):
        '''Train model on text files.
        
        Pass the UTF-8 byte sequences in all `data` files to the loop
        training model weights with stochastic gradient descent. 
        It will open file by file, repeating over the complete set (epoch)
        as long as validation error does not increase in between (early stopping).
        Validate on a random fraction of the file set automatically separated before.
        '''
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        
        assert self.status > 0 # incremental training is allowed
        assert self.incremental == False # no explicit state transfer
        
        data = list(data)
        shuffle(data) # random order of files (because generators cannot shuffle within files)
        if self.stateful: # we must split file-wise in stateful mode
            steps = self.length
            split = ceil(len(data)*self.validation_split) # split position in randomized file list
            training_data, validation_data = data[:-split], data[-split:] # reserve last files for validation
            for f in validation_data:
                print ('using input', f.name, 'for validation only')
            training_epoch_size = 0
            for f in training_data:
                text = f.read()
                training_epoch_size += ceil((len(text)-self.length)/steps/self.minibatch_size)
            validation_epoch_size = 0
            for f in validation_data:
                text = f.read()
                validation_epoch_size += ceil((len(text)-self.length)/steps/self.minibatch_size)
            reset_cb = ResetStatesCallback()
        else: # we can split window by window in stateless mode
            steps = 3
            total_size = 0
            max_size = 0
            with click.progressbar(data) as bar:
                for f in bar:
                    text = f.read()
                    size = len(text)
                    total_size += size - self.length
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
                        reset_cb.reset(f.name)
                    text = f.read()
                    # encode
                    sequences = []
                    next_chars = []
                    for i in range(0, len(text) - self.length, steps):
                        if not self.stateful and (split[int(i/steps)]<self.validation_split) == train:
                            continue # data shared between training and split: belongs to other generator
                        sequences.append(text[i: i + self.length])
                        if self.stateful:
                            next_chars.append(text[i+1: i+1 + self.length])
                        else:
                            next_chars.append(text[i + self.length])
                        if (len(sequences) % self.minibatch_size == 0 # next minibatch full
                            or i + steps >= len(text) - self.length): # last minibatch: partially filled batch
                            # vectorization
                            x = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(list(map(bytearray,sequences)), dtype=numpy.uint8)]
                            if self.stateful:
                                y = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(list(map(bytearray,next_chars)), dtype=numpy.uint8)]
                            else:
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
        self.status = 2
    
    def save(self, filename):
        '''Save model into `filename`.
        (Cannot preserve weights across CPU/GPU implementations or input shape configurations.)
        '''
        
        assert self.status != 0
        self.model.save(filename)
    
    def load(self, filename):
        '''Load model from `filename`.
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

    def save_config(self, configfilename):
        '''Save parameters from configuration.

        Save configured model parameters into `configfilename`.
        '''
        
        assert self.status > 0
        config = {'width': self.width, 'depth': self.depth, 'length': self.length, 'stateful': self.stateful, 'variable_length': self.variable_length}
        pickle.dump(config, open(configfilename, mode='wb'))
    
    def save_weights(self, weightfilename):
        '''Save weights of the trained model.

        Save trained model weights into `weightfilename`.
        (This preserves weights across CPU/GPU implementations or input shape configurations.)
        '''
        
        assert self.status > 1
        self.model.save_weights(weightfilename)
    
    def load_config(self, configfilename):
        '''Load parameters to prepare configuration/compilation.

        Load model configuration from `configfilename`.
        '''
        
        if self.status > 0:
            self.clear()
        config = pickle.load(open(configfilename, mode='rb'))
        self.width = config['width']
        self.depth = config['depth']
        self.length = config['length']
        self.stateful = config['stateful']
        self.variable_length = config['variable_length']
    
    def load_weights(self, weightfilename):
        '''Load weights into the configured/compiled model.

        Load weights from `weightfilename` into the compiled and configured model.
        (This preserves weights across CPU/GPU implementations or input shape configurations.)
        '''
        
        self.model.load_weights(weightfilename)
        
        self.status = 2
    
    def rate(self, text):
        '''Predict probabilities from model one by one.

        Calculate probabilities (individually) and perplexity (accumulated)
        of the character sequence in `text` according to the current model
        (predicting one by one).

        Return a list of character-probability tuples, and the overall perplexity.
        '''
        
        # prediction calculation is a lot slower that way than via batched generator, cf. rate_once() / test()
        # perplexity calculation is a lot slower that way than via tensor metric, cf. test()
        assert self.status > 1
        assert self.incremental == False # no explicit state transfer
        x = numpy.zeros((1, self.length, 256), dtype=numpy.bool)
        entropy = 0
        result = []
        length = 0
        self.model.reset_states()
        for c in text:
            p = 1.0
            for b in c.encode("utf-8"):
                x_input = x[:,x.any(axis=2)[0]] if self.variable_length else x
                if x_input.shape[1] > 0: # to prevent length 0 input
                    output = self.model.predict_on_batch(x_input).tolist()
                    pred = dict(enumerate(output[0][0] if self.stateful else output[0]))
                    entropy -= log(pred[b], 2)
                    length += 1
                    p *= pred[b]
                x = numpy.roll(x, -1, axis=1) # left-shifted by 1
                x[0,-1] = numpy.eye(256, dtype=numpy.bool)[b] # one-hot vector for b in last pos
            result.append((c, p))
        return result, pow(2.0, entropy/length)

    def rate_single(self, candidates, initial_states):
        '''Predict probability from model, passing initial and final state.

        Calculate the output probability distribution for a single input byte 
        incrementally according to the current model. Do so in parallel for 
        any number of hypotheses (i.e. batch size), identified by list position: 
        For `candidates` hypotheses with their `initial_states`, return a tuple of 
        their probabilities and their final states (for the next run).
        If any of `initial_states` is None, it is treated like reset (zero states).

        Return a list of probability arrays and of final states.

        (To be called by an adapter tracking history paths and input alternatives,
         combining them up to a maximum number of best running candidates, i.e. beam.
         See `scripts.generate` and `wrapper.ocrd_keraslm_rate` and `lib.Node`.)
        (Requires the model to be compiled in an incremental configuration.)
        '''
        
        assert self.status > 1
        assert self.stateful == False # no implicit state transfer
        assert self.incremental == True # only explicit state transfer
        # todo: allow graph input (by pruning via history clustering or push forward algorithm;
        #                       or by aggregating lattice input)
        assert len(candidates) == len(initial_states)
        n = len(candidates)
        # each initial_states[i] is a layer list (h1,c1,h2,c2,...) of state vectors
        # thus, each layer is a single input (and output) in addition to normal input (and output)
        # for batch processing, all hypotheses must be passed together:
        for i, initial_state in enumerate(initial_states):
            if not initial_state:
                initial_states[i] = [numpy.zeros((self.width), dtype=numpy.float) for n in range(0,self.depth*2)] # h+c per layer
        states_input = [numpy.vstack([initial_state[layer] for initial_state in initial_states]) for layer in range(0,self.depth*2)] # stack layers across batch (h+c per layer)
        x = numpy.expand_dims(numpy.eye(256, dtype=numpy.bool)[candidates], axis=1) # one-hot vector for all bytes; add time dimension
        output = self.model.predict_on_batch([x] + states_input)
        probs_output = output[0] # actually we need a (hypo) list of (score) vectors
        states_output = list(output[1:]) # from (layers) tuple
        preds = []
        final_states = []
        for i in range(0,n):
            preds.append(probs_output[i,:])
            final_states.append([layer[i:i+1] for layer in states_output])
        return preds, final_states
    
    def rate_once(self, textstring):
        '''Predict probabilities from model all at once.

        Calculate the probabilities of the character sequence in `textstring`
        according to the current model (predicting all at once).

        Return a list of probabilities (one per character/codepoint).
        '''
        
        assert self.status > 1
        assert self.incremental == False # no explicit state transfer
        text = textstring.encode("utf-8") # byte sequence
        size = len(text)
        steps = self.length if self.stateful else 1
        epoch_size = ceil((size-1)/self.minibatch_size/steps)
        
        # data preparation
        def gen_data(text):
            # encode
            while True:
                sequences = []
                for i in range(self.length if self.stateful else 0, size-1, steps): # sequence must not be length zero with tensorflow
                    if i < self.length:
                        if self.variable_length:
                            # partial window (needs interim minibatch size 1)
                            sequences.append(text[0:i])
                            x = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(list(map(bytearray,sequences)), dtype=numpy.uint8)]
                            yield x
                            sequences = []
                        else:
                            # zero padding
                            sequences.append(b'\0' * (self.length - i) + text[0:i])
                    else:
                        sequences.append(text[i - self.length: i])
                    if (len(sequences) % self.minibatch_size == 0 or 
                        i + steps >= size-1): # last minibatch: partially filled batch (smaller than self.minibatch_size)
                        # vectorization
                        x = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(list(map(bytearray,sequences)), dtype=numpy.uint8)]
                        yield x
                        sequences = []
                    if i + steps >= size-1: # last minibatch: 1 sample with partial length
                        if self.stateful:
                            sequences.append(text[i: size-1])
                        else:
                            sequences.append(text[size-1 - self.length: size-1])
                        # vectorization
                        x = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(list(map(bytearray,sequences)), dtype=numpy.uint8)]
                        yield x
                break # cause StopIteration exception (epoch size miscalculation)
        
        preds = self.model.predict_generator(gen_data(text), steps=epoch_size, verbose=1) # todo: make iterator thread-safe and use_multiprocesing=True
        preds = preds.reshape((size-1,256)) # reshape concatenation of batches to a contiguous temporal sequence
        # get predictions for true symbols (bytes)
        probs = [1/256]+preds[range(size-1),bytearray(text)[1:]].tolist() # all symbols but first byte (uniform prediction)
        # get predictions for true symbols (characters)
        encoder = codecs.getincrementalencoder("utf-8")()
        cprobs = [1.0] * len(textstring)
        j = 0
        for (i,c) in enumerate(textstring):
            for k in range(len(encoder.encode(c))):
                cprobs[i] *= probs[j]
                j += 1
        assert j == len(text)
        return cprobs
    
    def test(self, test_data):
        '''Evaluate model on `test_data` files.

        Calculate the perplexity of the UTF-8 byte sequences in 
        all `test_data` files according to the current model.

        Return the overall perplexity.
        '''
        
        assert self.status > 1
        assert self.incremental == False # no explicit state transfer
        self.model.reset_states()
        # todo: Since Keras does not allow callbacks within evaluate() / evaluate_generator() / test_loop(),
        #       we cannot reset_states() between input files as we do in train().
        #       Thus we should evaluate each file individually, reset in between, and accumulate losses.
        #       But this looks awkward, since we get N progress bars instead of 1, in contrast to training.
        #       Perhaps the overall error introduced into stateful models by not resetting is not that high
        #       after all?
        epoch_size = 0
        steps = self.length if self.stateful else 1
        with click.progressbar(test_data) as bar:
            for f in bar:
                text = f.read()
                size = len(text)
                epoch_size += ceil((size-1)/self.minibatch_size/steps)
        
        # data preparation
        def gen_data(files):
            # encode
            while True:
                for f in files:
                    f.seek(0)
                    text = f.read()
                    # if self.stateful: reset_cb.reset(f.name)
                    sequences = []
                    next_chars = []
                    for i in range(self.length if self.stateful else 0, len(text)-1, steps):
                        if i < self.length:
                            if self.variable_length:
                                # partial window (needs interim minibatch size 1)
                                sequences.append(text[0:i])
                                x = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(list(map(bytearray,sequences)), dtype=numpy.uint8)]
                                y = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(bytearray(next_chars), dtype=numpy.uint8)]
                                yield (x,y)
                                sequences = []
                                next_chars = []
                            else:
                                # zero padding
                                sequences.append(b'\0' * (self.length - i) + text[0:i])
                        else:
                            sequences.append(text[i - self.length: i])
                        if self.stateful:
                            next_chars.append(text[i+1 - self.length: i+1])
                        else:
                            next_chars.append(text[i])
                        if (len(sequences) % self.minibatch_size == 0 or 
                            i + steps >= len(text)-1): # last minibatch: partially filled batch (smaller than self.minibatch_size)
                            # vectorization
                            x = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(list(map(bytearray,sequences)), dtype=numpy.uint8)]
                            if self.stateful:
                                y = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(list(map(bytearray,next_chars)), dtype=numpy.uint8)]
                            else:
                                y = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(bytearray(next_chars), dtype=numpy.uint8)]
                            yield (x,y)
                            sequences = []
                            next_chars = []
                        if i + steps >= len(text)-1: # last minibatch: 1 sample with partial length
                            if self.stateful:
                                next_chars.append(text[i+1: len(text)])
                                sequences.append(text[i: len(text)-1])
                            else:
                                next_chars.append(text[len(text)])
                                sequences.append(text[len(text)-1 - self.length: len(text)-1])
                            # vectorization
                            x = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(list(map(bytearray,sequences)), dtype=numpy.uint8)]
                            if self.stateful:
                                y = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(list(map(bytearray,next_chars)), dtype=numpy.uint8)]
                            else:
                                y = numpy.eye(256, dtype=numpy.bool)[numpy.asarray(bytearray(next_chars), dtype=numpy.uint8)]
                            yield (x,y)
                break # cause StopIteration exception (epoch size miscalculation)
        
        loss, accuracy = self.model.evaluate_generator(gen_data(test_data), steps=epoch_size, verbose=1) # todo: make iterator thread-safe and use_multiprocesing=True
        return exp(loss)
    

class ResetStatesCallback(Callback):
    '''Keras callback for stateful models to reset state between files.
    
    Callback to be called by `fit_generator()` or even `evaluate_generator()`:
    do `model.reset_states()` whenever generator sees EOF (on_batch_begin with self.eof),
    and between training and validation (on_batch_end with batch>=steps_per_epoch-1).
    '''
    def __init__(self):
        self.eof = False
        self.here = ''
        self.next = ''
    
    def reset(self, where):
        self.eof = True
        self.next = where
    
    def on_batch_begin(self, batch, logs={}):
        if self.eof:
            # between training files
            self.model.reset_states()
            self.eof = False
            self.here = self.next
    
    def on_batch_end(self, batch, logs={}):
        if logs.get('loss') > 25:
            print('huge loss in', self.here, 'at', batch)
        if (self.params['do_validation'] and batch >= self.params['steps']-1):
            # in fit_generator just before evaluate_generator
            self.model.reset_states()

            
class Node(object):
    '''One node in a tree of textual alternatives for beam search.
    
    Each node has a parent attribute:
    - `parent`: previous node in tree
    and 3 content attributes:
    - `value`: byte at that position in the sequence
    - `state`: LM state vectors representing the past sequence
    - `extras`: UTF-8 incremental decoder state after value, or node identifier of value
    as well as a score attribute:
    - `cum_cost`: cumulative LM score of sequence after value
    and two convenience attributes:
    - `length`: length of sequence (number of nodes/bytes) starting from root
    - `_sequence`: list of nodes in the sequence

    This data structure is needed for for beam search of best paths.'''
    
    def __init__(self, parent, state, value, cost, extras):
        super(Node, self).__init__()
        self.value = value # byte
        self.parent = parent # parent Node, None for root
        self.state = state # list of recurrent hidden layers states (h and c for each layer)
        self.cum_cost = parent.cum_cost + cost if parent else cost
        self.length = 1 if parent is None else parent.length + 1
        self.extras = extras # UTF-8 decoder state or node identifier
        self._sequence = None
        #print('added node', bytes([n.value for n in self.to_sequence()]).decode("utf-8", "ignore"))
    
    def to_sequence(self):
        """Return a sequence of nodes from root to current node."""
        if not self._sequence:
            self._sequence = []
            current_node = self
            while current_node:
                self._sequence.insert(0, current_node)
                current_node = current_node.parent
        return self._sequence
    
    def __lt__(self, other):
        return self.cum_cost < other.cum_cost
    def __le__(self, other):
        return self.cum_cost <= other.cum_cost
    def __eq__(self, other):
        return self.cum_cost == other.cum_cost
    def __ne__(self, other):
        return self.cum_cost != other.cum_cost
    def __gt__(self, other):
        return self.cum_cost > other.cum_cost
    def __ge__(self, other):
        return self.cum_cost >= other.cum_cost
