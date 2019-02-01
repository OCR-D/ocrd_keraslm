import os
import tempfile
import pickle
import codecs
from random import shuffle
from math import log, exp, ceil
import signal
import logging
import click
import numpy
import h5py
from keras.callbacks import Callback

class Rater(object):
    '''A character-level RNN language model for rating text.
    
    Uses Keras to define, compile, train, run and test an RNN
    (LSTM) language model on the character level. The model's
    topology (layers depth, per-layer width, window length) can
    be controlled before training.
    
    To be used by stand-alone CLI (`scripts.train` for training,
    `scripts.apply` for prediction, `scripts.test` for evaluation),
    or OCR-D processing (`wrapper.ocrd_keraslm_rate`).
    
    Interfaces:
    - `Rater.train`/`scripts.train` : file handles of character sequences
    - `Rater.test`/`scripts.test` : file handles of character sequences
    - `Rater.rate`/`scripts.apply` : character string
    - `Rater.rate_once`/`wrapper.ocrd_keraslm_rate` : character string
    - `Rater.rate_single`/`scripts.generate` or `wrapper.ocrd_keraslm_rate` :
      alternative list of characters and states
    '''
    
    def __init__(self, logger=None):
        '''Reset model and set all parameters to their defaults.'''
        
        # configuration variables -- will be overwritten by CLI for train() / by load_config for rate()/test()
        self.width = 0 # number of LSTM cells per hidden layer
        self.depth = 0 # number of LSTM hidden layers
        self.length = 0 # number of backpropagation timesteps per LSTM cell
        self.variable_length = True # also train on partially filled windows
        self.stateful = True # keep states across batches within one text (implicit state transfer)
        self.mapping = ({},{}) # indexation of (known/allowed) input and output characters (i.e. vocabulary)
        # configuration constants
        self.batch_size = 128 # will be overwritten by length if stateful
        self.validation_split = 0.2 # fraction of training data to use for validation (generalization control)
        # runtime variables
        self.logger = logger or logging.getLogger('')
        self.incremental = False # whether compiled with additional (initial) input state and (final) output state (explicit state transfer)
        self.model = None
        self.status = 0 # empty / compiled / trained?
        self.voc_size = 0 # (derived from mapping)
    
    def configure(self):
        '''Define and compile model for the given parameters.'''
        from keras.layers import Dense, TimeDistributed, Input, Embedding, Lambda
        from keras.layers import LSTM, CuDNNLSTM
        from keras import backend as K
        from keras.models import Model

        if self.stateful:
            self.variable_length = False # (to avoid inconsistency)
        length = None if self.variable_length else self.length
        # automatically switch to CuDNNLSTM if CUDA GPU is available:
        has_cuda = K.backend() == 'tensorflow' and K.tensorflow_backend._get_available_gpus()
        print('using', 'GPU' if has_cuda else 'CPU', 'LSTM implementation to compile',
              'stateful' if self.stateful else 'stateless',
              'incremental' if self.incremental else 'contiguous',
              'model of depth', self.depth, 'width', self.width, 'length', self.length, 'size', self.voc_size)
        lstm = CuDNNLSTM if has_cuda else LSTM
        if self.stateful:
            self.batch_size = 1
            input_args = {'batch_shape': (self.batch_size, None)} # batch size must be constant, variable length
        elif self.incremental:
            states_input_args = {'shape': (self.width,)}
            model_states_input = []
            model_states_output = []
            input_args = {'shape': (1,)} # batch size not fixed
        else:
            input_args = {'shape': (length,)} # batch size not fixed (e.g. different between training and prediction)
        input_args['dtype'] = 'int32'
        model_input = Input(**input_args)
        embedding = Embedding(self.voc_size, self.width, name='char_embedding')
        model_output = embedding(model_input) # mask_zero=True does not work with CuDNNLSTM
        for i in range(self.depth): # layer loop
            args = {'return_sequences': (i+1 < self.depth) or self.stateful, 'stateful': self.stateful}
            if not has_cuda:
                args['recurrent_activation'] = 'sigmoid' # instead of default 'hard_sigmoid' which deviates from CuDNNLSTM
            if self.incremental:
                # incremental prediction needs additional inputs and outputs for state (h,c):
                states = [Input(**states_input_args), Input(**states_input_args)]
                layer = lstm(self.width, return_state=True, **args)
                model_states_input.extend(states)
                model_output, state_h, state_c = layer(model_output, initial_state=states)
                model_states_output.extend([state_h, state_c])
            else:
                layer = lstm(self.width, **args)
                model_output = layer(model_output)
        if self.stateful:
            layer = TimeDistributed(Lambda(lambda x: K.softmax(K.dot(x, K.transpose(embedding.embeddings)))))
        else:
            Lambda(lambda x: K.softmax(K.dot(x, K.transpose(embedding.embeddings))))
        model_output = layer(model_output)
        if self.incremental:
            self.model = Model([model_input] + model_states_input, [model_output] + model_states_output)
        else:
            self.model = Model(model_input, model_output)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.status = 1
    
    def train(self, data, val_data=None):
        '''Train model on text files.
        
        Pass the character sequences in all `data` files to the loop
        training model weights with stochastic gradient descent.
        It will open file by file, repeating over the complete set (epoch)
        as long as validation error does not increase in between (early stopping).
        Validate on a random fraction of the file set automatically separated before.
        (Data are split by window/file in stateless/stateful mode.)
        
        If `val_data` is given, then do not split, but use those files
        for validation instead (regardless of mode).
        '''
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        
        assert self.status > 0 # must be configured already, but incremental training is allowed
        assert self.incremental is False # no explicit state transfer
        
        # extract character mapping and calculate epoch size:
        chars = set(self.mapping[0].keys())
        data = list(data)
        shuffle(data) # random order of files (because generators cannot shuffle within files)
        total_size = 0
        if self.stateful: # we must split file-wise in stateful mode
            steps = self.length
            if val_data:
                training_data = data
                validation_data = val_data
            else:
                split = ceil(len(data)*self.validation_split) # split position in randomized file list
                training_data, validation_data = data[:-split], data[-split:] # reserve last files for validation
            assert len(training_data) > 0, "stateful mode needs at least one file for training"
            assert len(validation_data) > 0, "stateful mode needs at least one file for validation"
            for file in validation_data:
                print('using input', file.name, 'for validation only')
            training_epoch_size = 0
            for file in training_data:
                text = file.read()
                size = len(text)
                total_size += size
                training_epoch_size += ceil((size-self.length)/steps/self.batch_size)
                chars.update(set(text))
            validation_epoch_size = 0
            for file in validation_data:
                text = file.read()
                size = len(text)
                total_size += size
                validation_epoch_size += ceil((size-self.length)/steps/self.batch_size)
                chars.update(set(text))
            split = None
            reset_cb = ResetStatesCallback()
        else: # we can split window by window in stateless mode
            steps = 3
            max_size = 0
            with click.progressbar(data) as pbar:
                for file in pbar:
                    text = file.read()
                    size = len(text)
                    total_size += size - self.length
                    max_size = max(max_size, size)
                    chars.update(set(text))
            if val_data:
                training_epoch_size = ceil(total_size/steps/self.batch_size)
                with click.progressbar(val_data) as pbar:
                    for file in pbar:
                        text = file.read()
                        size = len(text)
                        total_size += size - self.length
                validation_epoch_size = ceil(total_size/steps/self.batch_size)
                training_data = data
                validation_data = val_data
                split = None
            else:
                epoch_size = total_size/steps/self.batch_size
                training_epoch_size = ceil(epoch_size*(1-self.validation_split))
                validation_epoch_size = ceil(epoch_size*self.validation_split)
                validation_data, training_data = data, data # same data, different generators (see below)
                split = numpy.random.uniform(0, 1, (ceil(max_size/steps),)) # reserve split fraction at random positions
            if self.variable_length:
                training_epoch_size *= 1.1 # training data augmented with partial windows (1+subsampling ratio)
        chars = sorted(list(chars))
        self.voc_size = len(chars) + 1 # reserve 0 for padding
        c_i = dict((c, i) for i, c in enumerate(chars, 1))
        i_c = dict((i, c) for i, c in enumerate(chars, 1))
        self.mapping = (c_i, i_c)
        print('training on %d files / %d batches per epoch / %d character tokens for %d character types' % (len(training_data), training_epoch_size, total_size, self.voc_size))
        
        # update mapping-specific layers:
        embedding = self.model.get_layer(name='char_embedding')
        #dense = self.model.get_layer(name='char_dense')
        if embedding.input_dim < self.voc_size: # more chars than during last training?
            if self.status >= 2: # weights exist already (i.e. incremental training)?
                print('transferring weights from previous model with only %d character types' % embedding.input_dim)
                # get old weights:
                layer_weights = [layer.get_weights() for layer in self.model.layers]
                # reconfigure with new mapping size (and new initializers):
                self.configure()
                # set old weights:
                for layer, weights in zip(self.model.layers, layer_weights):
                    if layer.name == 'char_embedding':
                        # transfer weights from previous Embedding layer to new one:
                        new_weights = layer.get_weights()
                        new_weights[0][0:embedding.input_dim, 0:embedding.output_dim] = weights[0]
                        layer.set_weights(new_weights)
                    elif layer.name == 'char_dense':
                        # transfer kernel and bias weights from previous Dense layer to new one:
                        new_weights = layer.get_weights()
                        new_weights[0][0:dense.input_shape[2], 0:dense.output_shape[2]] = weights[0]
                        new_weights[1][0:dense.output_shape[2]] = weights[1]
                        layer.set_weights(new_weights)
                    else:
                        # use old weights:
                        layer.set_weights(weights)
            else:
                self.configure()
        
        # fit model
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)]
        if self.stateful:
            callbacks.append(reset_cb)
        def stopper(sig, frame):
            print('stopping training')
            self.model.stop_training = True
        signal.signal(signal.SIGINT, stopper)
        self.model.fit_generator(self.gen_data_from_files(training_data, steps, split=split, train=reset_cb if self.stateful else True, repeat=True),
                                 steps_per_epoch=training_epoch_size, epochs=100,
                                 workers=1, use_multiprocessing=True,
                                 validation_data=self.gen_data_from_files(validation_data, steps, split=split, train=False, repeat=True),
                                 validation_steps=validation_epoch_size,
                                 verbose=1, callbacks=callbacks)
        # set state
        self.status = 2
    
    def save(self, filename):
        '''Save weights of the trained model, and configuration parameters.
        
        Save both the configured parameters and the trained weights
        of the model into `filename`.
        (This preserves weights across CPU/GPU implementations or input shape configurations.)
        '''
        assert self.status > 1
        self.model.save_weights(filename)
        with h5py.File(filename, 'a') as f:
            g = f.create_group('config')
            g.create_dataset('width', data=numpy.array(self.width))
            g.create_dataset('depth', data=numpy.array(self.depth))
            g.create_dataset('length', data=numpy.array(self.length))
            g.create_dataset('stateful', data=numpy.array(self.stateful))
            g.create_dataset('variable_length', data=numpy.array(self.variable_length))
            g.create_dataset('mapping', data=numpy.fromiter((ord(self.mapping[1][i]) if i in self.mapping[1] else 0 for i in range(self.voc_size)), dtype='uint32'))
    
    def load_config(self, filename):
        '''Load parameters to prepare configuration/compilation.

        Load model configuration from `filename`.
        '''
        assert self.status == 0
        with h5py.File(filename, 'r') as f:
            g = f['config']
            self.width = g['width'][()]
            self.depth = g['depth'][()]
            self.length = g['length'][()]
            self.stateful = g['stateful'][()]
            self.variable_length = g['variable_length'][()]
            c_i = dict((chr(c), i) for i, c in enumerate(g['mapping'][()]) if c > 0)
            i_c = dict((i, chr(c)) for i, c in enumerate(g['mapping'][()]) if c > 0)
            self.mapping = (c_i, i_c)
            self.voc_size = len(c_i) + 1
        self.status = 1
    
    def load_weights(self, filename):
        '''Load weights into the configured/compiled model.

        Load weights from `filename` into the compiled and configured model.
        (This preserves weights across CPU/GPU implementations or input shape configurations.)
        '''
        assert self.status > 0
        self.model.load_weights(filename)
        self.status = 2
    
    def rate(self, text):
        '''Rate a string one by one.

        Calculate probabilities (individually) and perplexity (accumulated)
        of the character sequence in `text` according to the current model
        (predicting one by one).

        Return a list of character-probability tuples, and the overall perplexity.
        '''
        
        # prediction calculation is a lot slower that way than via batched generator, cf. rate_once() / test()
        # perplexity calculation is a lot slower that way than via tensor metric, cf. test()
        assert self.status > 1
        assert self.incremental is False # no explicit state transfer
        x = numpy.zeros((1, 1 if self.stateful else self.length), dtype=numpy.uint32)
        entropy = 0
        result = []
        if self.stateful:
            self.model.reset_states()
        for i, char in enumerate(text):
            if char not in self.mapping[0]:
                self.logger.error('unmapped character "%s" at input position %d', char, i)
                idx = 0
            else:
                idx = self.mapping[0][char]
            if i == 0:
                result.append((char, 1.0)) # or merely uniform baseline?
            else:
                x_input = x[:, -i:] if self.variable_length else x
                output = self.model.predict_on_batch(x_input).tolist()
                pred = dict(enumerate(output[0][0] if self.stateful else output[0]))
                prob = pred[idx]
                entropy -= log(max(prob,1e-99), 2)
                result.append((char, prob))
            x = numpy.roll(x, -1, axis=1) # left-shift by 1 time-step
            x[0, -1] = idx # fill with next char
        return result, pow(2.0, entropy/len(text))

    def predict(self, candidates, initial_states):
        '''Predict character probabilities, passing initial and final states.
        
        Calculate the output probability distribution for a single input character
        incrementally according to the current model. Do so in parallel for
        any number of hypotheses (i.e. batch size), identified by list position:
        For `candidates` hypotheses with their `initial_states`, return a tuple of
        their probabilities and their final states (for the next run).
        If any of `initial_states` is None, it is treated like reset (zero states).
        
        Return a list of probability arrays and a list of final states.
        
        (To be called by an adapter tracking history paths and input alternatives,
         combining them up to a maximum number of best running candidates, i.e. beam.
         See `scripts.generate` and `wrapper.ocrd_keraslm_rate` and `lib.Node`.)
        (Requires the model to be compiled in an incremental configuration.)
        '''
        
        assert self.status > 1
        assert self.stateful is False # no implicit state transfer
        assert self.incremental is True # only explicit state transfer
        assert len(candidates) == len(initial_states), "number of inputs (%d) and number of states (%d) inconsistent" % (len(candidates), len(initial_states))
        n = len(candidates)
        inputs = numpy.zeros((n, 1), dtype=numpy.uint32)
        for i in range(n):
            char = candidates[i]
            if char not in self.mapping[0]:
                self.logger.error('unmapped character "%s" at input alternative %d', char, i)
                idx = 0
            else:
                idx = self.mapping[0][char]
            inputs[i, 0] = idx
        # each initial_states[i] is a layer list (h1,c1,h2,c2,...) of state vectors
        # thus, each layer is a single input (and output) in addition to normal input (and output)
        # for batch processing, all hypotheses must be passed together:
        for i, initial_state in enumerate(initial_states):
            if not initial_state:
                initial_states[i] = [numpy.zeros((self.width), dtype=numpy.float) for n in range(0, self.depth*2)] # h+c per layer
        states_inputs = [numpy.vstack([initial_state[layer] for initial_state in initial_states]) for layer in range(0, self.depth*2)] # stack layers across batch (h+c per layer)
        
        outputs = self.model.predict_on_batch([inputs] + states_inputs)
        probs_outputs = outputs[0]
        states_outputs = list(outputs[1:]) # we need a (layers) list instead of a tuple
        preds = [] # we need a (hypo) list of (score) vectors instead of an array
        final_states = [] # we need a (hypo) list of (layers) list of state vectors
        for i in range(n):
            preds.append(probs_outputs[i, :])
            final_states.append([layer[i:i+1] for layer in states_outputs])
        return preds, final_states
    
    def rate_once(self, text):
        '''Rate a string all at once.
        
        Calculate the probabilities of the character sequence in `text`
        according to the current model (predicting all at once).
        
        Return a list of probabilities (one per character/codepoint).
        
        (To be called on (subsequent chunks of) text directly, as a faster
         replacement for `rate`. See also `wrapper.ocrd_keraslm_rate`.)
        (Requires the model to be compiled in a non-incremental configuration.)
        '''
        
        assert self.status > 1
        assert self.incremental is False # no explicit state transfer
        size = len(text)
        steps = self.length if self.stateful else 1
        epoch_size = ceil((size-1)/self.batch_size/steps)
        preds = self.model.predict_generator(self.gen_data(text, steps), steps=epoch_size, verbose=1)
        preds = preds.reshape((size-1, self.voc_size))
        
        # get predictions for true symbols (characters)
        probs = [1.0] # or merely uniform baseline?
        for pred, next_char in zip(preds, list(text[1:])):
            if next_char not in self.mapping[0]:
                idx = 0
            else:
                idx = self.mapping[0][next_char]
            probs.append(pred[idx])
            if len(probs) >= size:
                break # stop short of last batch residue
        return probs
    
    def test(self, test_data):
        '''Evaluate model on `test_data` files.
        
        Calculate the perplexity of the character sequences in
        all `test_data` files according to the current model.
        
        Return the overall perplexity.
        '''
        
        assert self.status > 1
        assert self.incremental is False # no explicit state transfer
        if self.stateful:
            self.model.reset_states()
        # todo: Since Keras does not allow callbacks within evaluate() / evaluate_generator() / test_loop(),
        #       we cannot reset_states() between input files as we do in train().
        #       Thus we should evaluate each file individually, reset in between, and accumulate losses.
        #       But this looks awkward, since we get N progress bars instead of 1, in contrast to training.
        #       Perhaps the overall error introduced into stateful models by not resetting is not that high
        #       after all?
        epoch_size = 0
        steps = self.length if self.stateful else 1
        with click.progressbar(test_data) as pbar:
            for file in pbar:
                text = file.read()
                size = len(text)
                epoch_size += ceil((size-1)/self.batch_size/steps)
        
        # todo: make iterator thread-safe and then use_multiprocesing=True
        loss, _accuracy = self.model.evaluate_generator(self.gen_data_from_files(test_data, steps), steps=epoch_size, verbose=1)
        return exp(loss)
    
    # data preparation
    def gen_data_from_files(self, files, steps, split=None, train=False, repeat=False):
        '''Generate numpy arrays suitable for batch processing.
        
        Split the character sequences read from `files` into windows (as configured), 
        progressing by `steps` at a time. Yield successive batches of
        input and expected output arrays, accordingly.
        
        If `split` is given, then omit windows randomly at a rate equal to
        validation_split (as configured).
        '''
        while True:
            for file in files:
                file.seek(0)
                if self.stateful and train:
                    train.reset(file.name)
                yield from self.gen_data(file.read(), steps, train, split)
            if not repeat:
                break # causes StopIteration exception if calculated epoch size is too large

    # todo: make iterator thread-safe and then use_multiprocesing=True
    def gen_data(self, text, steps, train=False, split=None):
        '''Generate numpy arrays suitable for batch processing.
        
        Split the character sequence `text` into windows (as configured), 
        progressing by `steps` at a time. Yield successive batches of
        input and expected output arrays, accordingly.
        
        If `split` is given, then omit windows randomly at a rate equal to
        validation_split (as configured).
        '''
        # `steps` also needed by caller, therefore rather passed as arguments:
        # if self.stateful:
        #     steps = self.length
        # else:
        #     if train:
        #         steps = 3
        #     else:
        #         steps = 1
        size = len(text)
        sequences = []
        next_chars = []
        for i in range(self.length if self.stateful else 0, size, steps):
            if isinstance(split, numpy.ndarray) and (split[int(i/steps)] < self.validation_split) == train:
                continue # data shared between training and validation: belongs to other generator, resp.
            if i < self.length:
                if self.variable_length and not train:
                    # partial window (needs interim batch size 1 for interim length i):
                    yield self.vectorize([text[0:i]], [text[i]], length=i, batch_size=1)
                    continue
                else:
                    # below, vectorize() will do zero right-padding
                    sequences.append(text[0:i])
            else:
                sequences.append(text[i - self.length: i])
            if self.stateful:
                next_chars.append(text[i+1 - self.length: i+1])
            else:
                next_chars.append(text[i])
            if (len(sequences) % self.batch_size == 0 or # next full batch or
                i + steps >= size): # last (partially filled) batch?
                x, y = self.vectorize(sequences, next_chars)
                yield x, y
                if train and self.variable_length and isinstance(split, numpy.ndarray): # also train for partial windows by subsampling?
                    r = (split[int(i/steps)]-self.validation_split)/(1-self.validation_split) # re-use rest of random number for subsampling
                    r_max = 0.1 # effective subsampling ratio
                    if r < r_max:
                        j = int((self.length-1) * r / r_max) + 1 # re-use rest of random number for length
                        # erase complete batch by sublength from the left to simulate running in with zero padding as in rate():
                        # x[:, 0:j] = 0
                        # yield (x, y)
                        # shorten complete batch to sublength from the right to simulate running in with short sequences in rate():
                        yield (x[:, -j:], y)
                sequences = []
                next_chars = []
            if self.stateful and i + steps >= size: # last batch: 1 sample with partial length
                next_chars.append(text[i+1: size])
                sequences.append(text[i: size-1])
                yield self.vectorize(sequences, next_chars, length=size-i-1)
    
    def vectorize(self, inputs, outputs=None, length=None, batch_size=None):
        '''Convert a sequence of characters into numpy arrays.
        '''
        if not length:
            length = self.length
        if not batch_size:
            batch_size = self.batch_size
        # vectorization
        x = numpy.zeros((batch_size, length), dtype=numpy.uint32)
        if self.stateful:
            y = numpy.zeros((batch_size, length, self.voc_size), dtype=numpy.bool)
        else:
            y = numpy.zeros((batch_size, self.voc_size), dtype=numpy.bool)
        for k, sequence in enumerate(inputs):
            assert k < batch_size, 'input sequence %d (%s) exceeds batch size' % (k, sequence)
            for t, char in enumerate(sequence):
                assert t < length, 'input sequence %d (%s) exceeds window length' % (t, sequence)
                if char not in self.mapping[0]:
                    self.logger.error('unmapped character "%s" at input position %d', char, t + k * length)
                    idx = 0
                else:
                    idx = self.mapping[0][char]
                x[k, t] = idx
                if outputs:
                    if self.stateful:
                        char = outputs[k][t]
                    else:
                        char = outputs[k]
                    if char not in self.mapping[0]:
                        self.logger.error('unmapped character "%s" at output position %d', char, t + k * length)
                        idx = 0
                    else:
                        idx = self.mapping[0][char]
                    if self.stateful:
                        y[k, t, idx] = 1
                    else:
                        y[k, idx] = 1
        return x, y
    
    def print_charset(self):
        '''Print the mapped characters, newline-separated.'''
        assert self.status > 0
        for i, c in self.mapping[1].items():
            print('%d: "%s"' % (i, c))

class ResetStatesCallback(Callback):
    '''Keras callback for stateful models to reset state between files.
    
    Callback to be called by `fit_generator()` or even `evaluate_generator()`:
    do `model.reset_states()` whenever generator sees EOF (on_batch_begin with self.eof),
    and between training and validation (on_batch_end with batch>=steps_per_epoch-1).
    '''
    def __init__(self):
        super(ResetStatesCallback, self).__init__()
        self.eof = False
        self.here = ''
        self.next = ''
    
    def reset(self, where):
        '''Reset the model after the end of the current batch.'''
        self.eof = True
        self.next = where
    
    def on_batch_begin(self, batch, logs=None):
        if self.eof:
            # between training files
            self.model.reset_states()
            self.eof = False
            self.here = self.next
    
    def on_batch_end(self, batch, logs=None):
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
    - `extras`: node identifier of value (not used yet)
    as well as a score attribute:
    - `cum_cost`: cumulative LM score of sequence after value
    and two convenience attributes:
    - `length`: length of sequence (number of nodes/bytes) starting from root
    - `_sequence`: list of nodes in the sequence

    This data structure is needed for for beam search of best paths.'''
    
    def __init__(self, state, value, cost, parent=None, extras=None):
        super(Node, self).__init__()
        self.value = value # character
        self.parent = parent # parent Node, None for root
        self.state = state # list of recurrent hidden layers states (h and c for each layer)
        self.cum_cost = parent.cum_cost + cost if parent else cost
        self.length = 1 if parent is None else parent.length + 1
        self.extras = extras # node identifier
        self._sequence = None
        #print('added node', ''.join([n.value for n in self.to_sequence()]))
    
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
