import os
import codecs
import unicodedata
from random import shuffle
from math import log, exp, ceil
from bisect import insort_left
import signal
import logging
import click
import numpy as np
import h5py
from keras.callbacks import Callback

class Rater(object):
    '''A character-level RNN language model for rating text.
    
    Uses Keras to define, compile, train, run and test an RNN
    (LSTM) language model on the character level. The model's
    topology (layers depth, per-layer width, window length) can
    be controlled before training.
    
    To be used by stand-alone CLI (`scripts.run.train` for training,
    `scripts.run.test` for evaluation, `scripts.run.generate` for
    generation, `scripts.run.apply` for prediction),
    or OCR-D processing (`wrapper.rate`).
    
    Interfaces:
    - `Rater.train`/`scripts.run.train` : file handles of character sequences
    - `Rater.test`/`scripts.run.test` : file handles of character sequences
    - `Rater.rate2`/`scripts.run.apply` : character string
    - `Rater.rate`/`wrapper.rate` : character string
    - `Rater.rate_best`/`wrapper.rate` : lattice graph
    - `Rater.generate`/`scripts.run.generate` :
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
        self.mapping = ({}, {}) # indexation of (known/allowed) input and output characters (i.e. vocabulary)
        # configuration constants
        self.batch_size = 128 # will be overwritten by length if stateful
        self.validation_split = 0.2 # fraction of training data to use for validation (generalization control)
        self.smoothing = 0.2
        # runtime variables
        self.logger = logger or logging.getLogger(__name__)
        self.reset_cb = None # ResetStatesCallback instance in stateful training
        self.incremental = False # whether compiled with additional (initial) input state and (final) output state (explicit state transfer)
        self.model = None # (assigned by configure)
        self.status = 0 # empty / configured / trained?
        self.voc_size = 0 # (derived from mapping after loading or preparing training)
    
    def configure(self):
        '''Define and compile model for the given parameters.'''
        from keras.initializers import RandomNormal
        from keras.layers import Dense, TimeDistributed, Input
        from keras.layers import Embedding, Lambda, Concatenate
        from keras.layers import LSTM, CuDNNLSTM, Dropout
        from keras.models import Model
        from keras.optimizers import Adam
        from keras.regularizers import l2
        from keras import backend as K
        
        if self.stateful:
            self.variable_length = False # (to avoid inconsistency)
        length = None if self.variable_length else self.length
        # automatically switch to CuDNNLSTM if CUDA GPU is available:
        has_cuda = K.backend() == 'tensorflow' and K.tensorflow_backend._get_available_gpus()
        self.logger.info('using %s LSTM implementation to compile %s %s model of depth %d width %d length %d size %d',
                         'GPU' if has_cuda else 'CPU',
                         'stateful' if self.stateful else 'stateless',
                         'incremental' if self.incremental else 'contiguous',
                         self.depth, self.width, self.length, self.voc_size)
        lstm = CuDNNLSTM if has_cuda else LSTM
        
        # batch size and window length:
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

        # input layers:
        char_input = Input(name='char_input', **input_args)
        char_embedding = Embedding(self.voc_size, self.width, # (self.width - context_width)
                                   embeddings_initializer=RandomNormal(stddev=0.001),
                                   embeddings_regularizer=self._regularise_chars, # see below
                                   name='char_embedding')
        char_hidden = char_embedding(char_input) # mask_zero=True does not work with CuDNNLSTM
        
        context_inputs = [Input(name='context1_input', **input_args)] # context variable year # todo: author etc (meta-data)
        context_embeddings = [Embedding(200, 10, # year/10 from 0 to 2000 AD; 10 outdim seems fair
                                        embeddings_initializer=RandomNormal(stddev=0.001),
                                        embeddings_regularizer=self._regularise_contexts, # crucial, see below
                                        name='context1_embedding')]
        context_hiddens = [e(v) for v, e in zip(context_inputs, context_embeddings)]
        #context_width = sum(int(c.shape[-1]) for c in context_hiddens) # dimensionality of context vector
        
        # places to be modified as well when adding more contexts here:
        # * underspecify_contexts() -- calculation of default
        # * _gen_data_from_files() -- calculation from filename
        # * train() -- incremental training on older models with fewer contexts
        # * data preparation in scripts.run.generate/apply and wrapper.rate
        
        # hidden layers:
        model_output = Concatenate(name='concat_hidden_input')([char_hidden] + context_hiddens)
        for i in range(self.depth): # layer loop
            args = {'return_sequences': (i+1 < self.depth) or self.stateful,
                    'stateful': self.stateful,
                    # l2 does not converge:
                    #'kernel_regularizer': l2(0.01),
                    #'recurrent_regularizer': l2(0.01),
                    'name': 'lstm_%d' % (i+1)}
            if not has_cuda:
                args['recurrent_activation'] = 'sigmoid' # instead of default 'hard_sigmoid' which deviates from CuDNNLSTM
            if self.incremental:
                # incremental prediction needs additional inputs and outputs for state (h,c):
                states = [Input(name='initial_h_%d_input' % (i+1), **states_input_args),
                          Input(name='initial_c_%d_input' % (i+1), **states_input_args)]
                layer = lstm(self.width, return_state=True, **args)
                model_states_input.extend(states)
                model_output, state_h, state_c = layer(model_output, initial_state=states)
                model_states_output.extend([state_h, state_c])
            else:
                layer = lstm(self.width, **args)
                model_output = layer(model_output)
            if i > 0: # only hidden-to-hidden layer:
                if self.stateful:
                    constant_shape = (self.batch_size, 1, self.width) # variational dropout (time-constant)
                else:
                    constant_shape = (1, self.width) # variational dropout (time-constant)
                # LSTM (but not CuDNNLSTM) has the (non-recurrent) dropout keyword option for this:
                model_output = Dropout(0.1, noise_shape=constant_shape)(model_output)
        
        # output layer:
        def char_output(h):
            # re-use input embedding (weight tying), but add a bias vector, and also add a linear projection in hidden space
            # (see Press & Wolf 2017)
            # y = softmax( V * P * h + b ) with V=U the input embedding; initialise P as identity matrix and b as zero
            #proj = K.variable(np.eye(self.width), name='char_output_projection') # trainable=True by default
            #bias = K.variable(np.zeros((self.voc_size,)), name='char_output_bias') # trainable=True by default
            #return K.softmax(K.dot(h, K.transpose(K.dot(char_embedding.embeddings, proj))) + bias)
            # simplified variant with no extra weights (50% faster, equally accurate):
            return K.softmax(K.dot(h, K.transpose(char_embedding.embeddings)))
        if self.stateful:
            layer = TimeDistributed(Lambda(char_output), name='char_output')
        else:
            layer = Lambda(char_output, name='char_output')
        model_output = layer(model_output)
        
        if self.incremental:
            self.model = Model([char_input] + context_inputs + model_states_input, [model_output] + model_states_output)
        else:
            self.model = Model([char_input] + context_inputs, model_output)
        
        # does not converge: clipnorm=1...5, lr=1e-4
        # slower to converge, not better: amsgrad=True, decay=1e-2...1e-6
        # for transfer of old models without NaN losses: epsilon=0.1
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(clipvalue=5.0), metrics=['accuracy']) # 'adam'
        self.status = 1
    
    def underspecify_contexts(self):
        '''Get a default input for context variables.'''
        ncontexts = sum(1 for t in self.model.inputs if t.name.startswith('context')) # exclude char and initial states inputs
        self.logger.info('using underspecification (zero) for %d context variables', ncontexts)
        return [0] * ncontexts
    
    def _regularise_contexts(self, embedding_matrix):
        '''Calculate L2 loss of some context embedding weights to control for
        - low rank of the embedding matrix,
        - smoothness of adjacent embedding vectors,
        - underspecification at zero
          (by interpolating between other embedding vectors).
        '''
        from keras import backend as K
        
        em_dims = embedding_matrix.shape.as_list()
        lowrank = K.sum(K.square(embedding_matrix)) # generalization/sparsity
        vecs1 = K.slice(embedding_matrix, [1, 0], [em_dims[0]-2, em_dims[1]]) # all vectors except zero and last
        vecs2 = K.slice(embedding_matrix, [2, 0], [em_dims[0]-2, em_dims[1]]) # all vectors except zero and first
        vecs1 = K.stop_gradient(vecs1) # make sure only vecs2 is affected, i.e. t is not influenced by t+1
        smoothness = K.sum(K.square(vecs1 - vecs2)) # dist(t, t+1)
        # todo: learn adjacency matrix for categorical meta-data (update every epoch)
        vec0 = K.slice(embedding_matrix, [0, 0], [1, em_dims[1]])            # zero vector only,
        vec0 = K.repeat_elements(vec0, em_dims[0]-1, axis=0)                 # repeated
        vecs = K.slice(embedding_matrix, [1, 0], [em_dims[0]-1, em_dims[1]]) # all vectors except zero
        vecs = K.stop_gradient(vecs) # make sure only vec0 is affected, i.e. vecs change only via global loss
        # self-product increases weight of embedding vectors with large magnitude (i.e. more evidence):
        wgts = K.batch_dot(vecs, vecs, axes=1)
        underspecification = K.sum(K.square(vec0 - wgts * vecs)) # t=0 ~ weighted mean of t>0
        
        # todo: find good relative weights between these subtargets
        return self.smoothing * (lowrank + smoothness + underspecification)
    
    def _regularise_chars(self, embedding_matrix):
        '''Calculate L2 loss of the char embedding weights
        to control for underspecification at zero
        (by interpolating between other embedding vectors).
        '''
        from keras import backend as K
        
        em_dims = embedding_matrix.shape.as_list()
        if em_dims[0] == 0: # voc_size starts with 0 before first training
            return 0
        vec0 = K.slice(embedding_matrix, [0, 0], [1, em_dims[1]])            # zero vector only,
        vec0 = K.repeat_elements(vec0, em_dims[0]-1, axis=0)                 # repeated
        vecs = K.slice(embedding_matrix, [1, 0], [em_dims[0]-1, em_dims[1]]) # all vectors except zero
        # make sure only vec0 is affected, i.e. vecs change only via global loss:
        vecs = K.stop_gradient(vecs)
        # scale to make gradients benign:
        underspecification = K.sum(K.square(0.01 * (vec0 - vecs))) # c='\0' ~ mean of others
        lowrank = K.sum(K.square(0.01 * embedding_matrix)) # generalization/sparsity
        return lowrank + underspecification
    
    def train(self, data, val_data=None):
        '''Train model on text files.
        
        Pass the character sequences in all `data` files to the loop
        training model weights with stochastic gradient descent.
        Derive meta-data for context variables from file names.
        
        It will open file by file, repeating over the complete set (epoch)
        as long as validation error does not increase in between (early stopping).
        Validate on a random fraction of the file set automatically separated before.
        (Data are split by window/file in stateless/stateful mode.)
        
        If `val_data` is given, then do not split, but use those files
        for validation instead (regardless of mode).
        '''
        from keras.callbacks import EarlyStopping, TerminateOnNaN
        # uncomment the following lines to enter tfdbg during training:
        #from keras import backend as K
        #from tensorflow.python import debug as tf_debug
        #K.set_session(tf_debug.LocalCLIDebugWrapperSession(K.get_session()))

        assert self.status > 0 # must be configured already, but incremental training is allowed
        assert self.incremental is False # no explicit state transfer
        
        # extract character mapping and calculate epoch size:
        training_data, validation_data, split, training_epoch_size, validation_epoch_size, total_size, steps = self._split_data(data, val_data)
        self.logger.info('training on %d files / %d batches per epoch / %d character tokens for %d character types',
                         len(training_data), training_epoch_size, total_size, self.voc_size)
        
        # update mapping-specific layers:
        self.reconfigure_for_mapping()
        
        # fit model
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True),
                     TerminateOnNaN(),
                     StopSignalCallback(signal.SIGINT, self.logger)]
        if self.stateful:
            self.reset_cb = ResetStatesCallback(self.logger)
            callbacks.append(self.reset_cb)
        history = self.model.fit_generator(self._gen_data_from_files(training_data, steps, split=split, train=True, repeat=True),
                                 steps_per_epoch=training_epoch_size, epochs=100,
                                 workers=1, use_multiprocessing=False, # True makes communication with reset callback impossible
                                 validation_data=self._gen_data_from_files(validation_data, steps, split=split, train=False, repeat=True),
                                 validation_steps=validation_epoch_size,
                                 verbose=1, callbacks=callbacks)
        # set state
        if np.isnan(history.history['loss'][0]):
            self.logger.critical('training failed (NaN loss)')
            self.status = 1
        else:
            if 'val_loss' in history.history:
                self.logger.info('training finished with val_loss %f', min(history.history['val_loss']))
            self.status = 2
    
    def _split_data(self, data, val_data):
        '''Read text files and split into training vs validation, count batches and update char mapping.'''
        assert self.status >= 1
        
        shuffle(data) # random order of files (because generators cannot shuffle within files)
        
        total_size = 0
        chars = set(self.mapping[0].keys())
        if self.stateful: # we must split file-wise in stateful mode
            steps = self.length
            if val_data:
                training_data = data
                validation_data = val_data
            else:
                split = ceil(len(data)*self.validation_split) # split position in randomized file list
                training_data, validation_data = data[:-split], data[-split:] # reserve last files for validation
            assert training_data, "stateful mode needs at least one file for training"
            assert validation_data, "stateful mode needs at least one file for validation"
            for file in validation_data:
                self.logger.info('using input %s for validation only', file.name)
            training_epoch_size = 0
            with click.progressbar(training_data) as pbar:
                for file in pbar:
                    text, size = _read_normalize_file(file)
                    total_size += size
                    training_epoch_size += ceil((size-self.length)/steps/self.batch_size)
                    chars.update(set(text))
            validation_epoch_size = 0
            with click.progressbar(validation_data) as pbar:
                for file in pbar:
                    text, size = _read_normalize_file(file)
                    total_size += size
                    validation_epoch_size += ceil((size-self.length)/steps/self.batch_size)
                    chars.update(set(text))
            split = None
        else: # we can split window by window in stateless mode
            steps = 3
            max_size = 0
            with click.progressbar(data) as pbar:
                for file in pbar:
                    text, size = _read_normalize_file(file)
                    total_size += size - self.length
                    max_size = max(max_size, size)
                    chars.update(set(text))
            if val_data:
                training_epoch_size = ceil(total_size/steps/self.batch_size)
                with click.progressbar(val_data) as pbar:
                    for file in pbar:
                        text, size = _read_normalize_file(file)
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
                split = np.random.uniform(0, 1, (ceil(max_size/steps),)) # reserve split fraction at random positions
            if self.variable_length:
                training_epoch_size *= 1.1 # training data augmented with partial windows (1+subsampling ratio)
        chars = sorted(list(chars))
        self.voc_size = len(chars) + 1 # reserve 0 for padding
        c_i = dict((c, i) for i, c in enumerate(chars, 1))
        i_c = dict((i, c) for i, c in enumerate(chars, 1))
        self.mapping = (c_i, i_c)

        return training_data, validation_data, split, training_epoch_size, validation_epoch_size, total_size, steps
    
    def reconfigure_for_mapping(self):
        '''Reconfigure character embedding layer after change of mapping (possibly transferring previous weights).'''
        
        assert self.status >= 1
        embedding = self.model.get_layer(name='char_embedding')
        if embedding.input_dim < self.voc_size: # more chars than during last training?
            if self.status >= 2: # weights exist already (i.e. incremental training)?
                self.logger.warning('transferring weights from previous model with only %d character types', embedding.input_dim)
                # get old weights:
                layer_weights = [layer.get_weights() for layer in self.model.layers]
                # reconfigure with new mapping size (and new initializers):
                self.configure()
                # set old weights:
                for layer, weights in zip(self.model.layers, layer_weights):
                    self.logger.debug('transferring weights for layer %s %s', layer.name, str([w.shape for w in weights]))
                    if layer.name == 'char_embedding':
                        # transfer weights from previous Embedding layer to new one:
                        new_weights = layer.get_weights() # freshly initialised
                        #new_weights[0][embedding.input_dim:, 0:embedding.output_dim] = weights[0][0,:] # repeat zero vector instead
                        new_weights[0][0:embedding.input_dim, 0:embedding.output_dim] = weights[0]
                        layer.set_weights(new_weights)
                    else:
                        # use old weights:
                        layer.set_weights(weights)
            else:
                self.configure()
    
    def remove_from_mapping(self, char=None, idx=None):
        '''Remove one character from mapping and reconfigure embedding layer accordingly (transferring previous weights).'''
        
        assert self.status > 1
        assert self.voc_size > 0
        if not char and not idx:
            return False
        if char:
            if char in self.mapping[0]:
                idx = self.mapping[0][char]
            else:
                self.logger.error('unmapped character "%s" cannot be removed', char)
                return False
        else:
            if idx in self.mapping[1]:
                char = self.mapping[1][idx]
            else:
                self.logger.error('unmapped index "%d" cannot be removed', idx)
                return False
        embedding = self.model.get_layer(name='char_embedding').get_weights()[0]
        norm = np.linalg.norm(embedding[idx, :])
        self.logger.warning('pruning character "%s" [%d] with norm %f', char, idx, norm)
        self.mapping[0].pop(char)
        self.mapping[1].pop(idx)
        for i in range(idx + 1, self.voc_size):
            otherchar = self.mapping[1][i]
            self.mapping[0][otherchar] -= 1
            self.mapping[1][i-1] = otherchar
            self.mapping[1].pop(i)
        self.voc_size -= 1
        embedding = np.delete(embedding, idx, 0)
        # get old weights:
        layer_weights = [layer.get_weights() for layer in self.model.layers]
        # reconfigure with new mapping size (and new initializers):
        self.configure()
        # set old weights:
        for layer, weights in zip(self.model.layers, layer_weights):
            if layer.name == 'char_embedding':
                # transfer weights from previous Embedding layer to new one:
                layer.set_weights([embedding])
            else:
                # use old weights:
                layer.set_weights(weights)
        self.status = 2
        return True
    
    def test(self, test_data):
        '''Evaluate model on text files.
        
        Calculate the perplexity of the character sequences in
        all `test_data` files according to the current model.
        Derive meta-data for context variables from file names.
        
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
                text, size = _read_normalize_file(file)
                epoch_size += ceil((size-1)/self.batch_size/steps)
        
        # todo: make iterator thread-safe and then use_multiprocesing=True
        loss, _accuracy = self.model.evaluate_generator(self._gen_data_from_files(test_data, steps), steps=epoch_size, verbose=1)
        return exp(loss)
    
    def rate(self, text, context=None):
        '''Rate a string all at once.
        
        Calculate the probabilities of the character sequence in `text`
        according to the current model (predicting all at once).
        Use the integer list `context` as time-constant context variables,
        or zero-based underspecification.
        
        Return a list of probabilities (one per character/codepoint).
        
        (To be called on (subsequent chunks of) text directly, as a faster
         replacement for `rate`. See also `wrapper.rate`.)
        (Requires the model to be compiled in a non-incremental configuration.)
        '''
        
        assert self.status > 1
        assert self.incremental is False # no explicit state transfer
        if not context:
            context = self.underspecify_contexts()
        text = unicodedata.normalize('NFC', text)
        size = len(text)
        steps = self.length if self.stateful else 1
        epoch_size = ceil((size-1)/self.batch_size/steps)
        preds = self.model.predict_generator(self._gen_data(text, context, steps), steps=epoch_size, verbose=1)
        preds = preds.reshape((-1, self.voc_size))[0:size, :]
        
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
    
    def rate2(self, text, context=None):
        '''Rate a string one by one.
        
        Calculate probabilities (individually) and perplexity (accumulated)
        of the character sequence in `text` according to the current model
        (predicting one by one).
        Use the integer list `context` as time-constant context variables,
        or zero-based underspecification.
        
        Return a list of character-probability tuples, and the overall perplexity.
        '''
        
        # prediction calculation is a lot slower that way than via batched generator, cf. rate() / test()
        # perplexity calculation is a lot slower that way than via tensor metric, cf. test()
        assert self.status > 1
        assert self.incremental is False # no explicit state transfer
        if not context:
            context = self.underspecify_contexts()
        text = unicodedata.normalize('NFC', text)
        x = np.zeros((1, 1 if self.stateful else self.length), dtype=np.uint32)
        zs = [np.zeros((1, 1 if self.stateful else self.length), dtype=np.uint32) for _ in context]
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
                input_ = [x[:, -i:]] + [z[:, -i:] for z in zs] if self.variable_length else [x] + zs
                output = self.model.predict_on_batch(input_).tolist()
                pred = dict(enumerate(output[0][0] if self.stateful else output[0]))
                prob = pred[idx]
                entropy -= log(max(prob, 1e-99), 2)
                result.append((char, prob))
            x = np.roll(x, -1, axis=1) # left-shift by 1 time-step
            zs = [np.roll(z, -1, axis=1) for z in zs]
            x[0, -1] = idx # fill with next char
            for z, idx in zip(zs, context):
                z[0, -1] = idx
        return result, pow(2.0, entropy/len(text))
    
    def predict(self, candidates, initial_states, context=None):
        '''Predict character probabilities, passing initial and final states.
        
        Calculate the output probability distribution for a single input character
        incrementally according to the current model. Do so in parallel for
        any number of hypotheses (i.e. batch size), identified by list position:
        For `candidates` hypotheses with their `initial_states`, return a tuple of
        their probabilities and their final states (for the next run).
        If any of `initial_states` is None, it is treated like reset (zero states).
        
        Use the integer list `context` as time-constant context variables,
        or zero-based underspecification.
        
        Return a list of probability arrays and a list of final states.
        
        (To be called by an adapter tracking history paths and input alternatives,
         combining them up to a maximum number of best running candidates, i.e. beam.
         See `lib.generate` and `wrapper.rate` and `lib.Node`.)
        (Requires the model to be compiled in an incremental configuration.
         Ignores the configured batch size and window length.)
        '''
        
        assert self.status > 1
        assert self.stateful is False # no implicit state transfer
        assert self.incremental is True # only explicit state transfer
        assert len(candidates) == len(initial_states), "number of inputs (%d) and number of states (%d) inconsistent" % (len(candidates), len(initial_states))
        if not context:
            context = self.underspecify_contexts()
        n = len(candidates)
        x = np.zeros((n, 1), dtype=np.uint32)
        zs = [np.zeros((n, 1), dtype=np.uint32) for _ in context]
        for i in range(n):
            char = candidates[i]
            if char not in self.mapping[0]:
                # suppress the input error because it must be an aftereffect
                # of an error already reported by the caller (wrapper.rate)
                # – generative use case does not produce unmapped output (scripts.run.generate):
                #self.logger.error('unmapped character "%s" at input alternative %d', char, i)
                idx = 0
            else:
                idx = self.mapping[0][char]
            x[i, 0] = idx
            for z, idx in zip(zs, context):
                z[i, 0] = idx
        # each initial_states[i] is a layer list (h1,c1,h2,c2,...) of state vectors
        # thus, each layer is a single input (and output) in addition to normal input (and output)
        # for batch processing, all hypotheses must be passed together:
        for i, initial_state in enumerate(initial_states):
            if not initial_state:
                initial_states[i] = [np.zeros((self.width), dtype=np.float) for n in range(0, self.depth*2)] # h+c per layer
        states_inputs = [np.vstack([initial_state[layer] for initial_state in initial_states])
                         for layer in range(0, self.depth*2)] # stack layers across batch (h+c per layer)
        
        outputs = self.model.predict_on_batch([x] + zs + states_inputs)
        probs_outputs = outputs[0]
        states_outputs = list(outputs[1:]) # we need a (layers) list instead of a tuple
        preds = [] # we need a (hypo) list of (score) vectors instead of an array
        final_states = [] # we need a (hypo) list of (layers) list of state vectors
        for i in range(n):
            preds.append(probs_outputs[i, :])
            final_states.append([layer[i:i+1] for layer in states_outputs])
        return preds, final_states

    # todo: also allow specifying suffix
    def generate(self, prefix, number, context=None):
        '''Generate a number of characters after a prefix.
        
        Calculate the hidden layer state after reading the string `prefix`
        according to the current model. Use it as the initial state in the
        following beam search, constructing a first (zero-length) hypothesis:
        
        1. For each current hypothesis, calculate new predictions. Taking only
           the best-scoring candidates for the next character (beam threshold /
           beam width / fan-in), construct new hypotheses (replacing the current)
           by appending the next character candidates and the new HL state,
           respectively.
        2. Score all current hypotheses by adding the log probabilities of their
           individual characters, normalised by their length (norm heuristic).
        3. Separate hypotheses already of length `number` into the list of
           solutions. If enough are available (beam depth / fan-out), then
           terminate and return the best-scoring solution.
        4. Sort remaining hypotheses according to score and select the best ones
           (parallel batch size) to repeat with step 1.
        
        Use the integer list `context` as time-constant context variables,
        or zero-based underspecification.
        
        Return the character string of the best scoring hypothesis.
        
        (Requires the model to be compiled in an incremental configuration.
         Ignores the configured batch size and window length.)
        '''
        
        assert self.status > 1
        assert self.stateful is False # no implicit state transfer
        assert self.incremental is True # only explicit state transfer
        if not context:
            context = self.underspecify_contexts()
        
        # initial state
        prefix_states = [None]
        # prefix (to get correct initial state)
        for char in prefix[:-1]: # all but last character
            _, prefix_states = self.predict([char], prefix_states, context=context)
        next_fringe = [Node(state=prefix_states[0],
                            value=prefix[-1], # last character
                            cost=0.0)]
        
        # beam search
        for _ in range(number): # iterate over number of characters to be generated
            fringe = next_fringe
            preds, states = self.predict([n.value for n in fringe],
                                         [n.state for n in fringe],
                                         context=context)
            next_fringe = []
            for j, n in enumerate(fringe): # iterate over batch
                pred = preds[j]
                pred_best = np.argsort(pred)[-10:] # keep only 10-best alternatives
                pred_best = pred_best[np.searchsorted(pred[pred_best], 0.004):] # keep by absolute threshold
                costs = -np.log(pred[pred_best])
                state = states[j]
                for best, cost in zip(pred_best, costs): # follow up on best predictions
                    if best not in self.mapping[1]: # avoid zero/unmapped
                        continue # ignore this alternative
                    n_new = Node(parent=n, state=state, value=self.mapping[1][best], cost=cost)
                    insort_left(next_fringe, n_new) # add alternative to tree
            next_fringe = next_fringe[:256] # keep 256-best paths (equals batch size)
        best = next_fringe[0] # best-scoring
        result = ''.join([n.value for n in best.to_sequence()])
        
        return result # without prefix
    
    def rate_best(self, graph, start_node, end_node, context=None, lm_weight=0.5,
                  max_length=500, beam_width=100, beam_clustering_dist=0):
        '''Rate a lattice of string alternatives, decoding the best-scoring path.
        
        The lattice `graph` must be a networkx.DiGraph instance (i.e. a unigraph)
        with the following edge attributes:
        - `element`: reference to be kept in the result,
          should have an `id` property for debugging
          (e.g. a TextRegionType, TextLineType, WordType or GlyphType)
        - `alternatives`: list of string alternatives, represented
          as objects with the properties `Unicode` and a `conf`
          (e.g. ocrd_page_generateds.TextEquivType)
        Start with `start_node`, and ensure that `end_node` is its frontier.
        
        Parameters:
        - `lm_weight`: share for language model in the linear combination
          of log probability scores for language model and previous confidence
        - `max_length`: guaranteed boundary on string length of readings
        - `beam_width`: number of hypotheses (histories) to keep between elements
        - `beam_clustering_dist`: maximum distance between HL state vectors
          to form a cluster for pruning
        
        Search for the best path through the graph and its edge string alternatives,
        regarding both previous confidence values in the graph's edges and LM scores
        calculated for the respective path through the graph. Prune paths to avoid
        expanding all possible combinations.
        
        Return the single-best path as a list of tuples of:
        - the element reference of the edge (unchanged),
        - the best alternative object (unchanged),
        - the new score calculated locally.
        '''
        import networkx as nx
        
        # initial state; todo: pass from previous page
        graph.nodes[start_node]['traceback'] = [Node(state=None, value='\n', cost=0.0)]
        out = 0
        for in_, out in nx.bfs_edges(graph, start_node): # breadth-first search
            edge = graph.edges[in_, out]
            element = edge['element']
            textequivs = edge['alternatives']
            in_node = graph.nodes[in_]
            out_node = graph.nodes[out]
            # make sure we have already been at in_node
            assert 'traceback' in in_node, "breadth-first search should have visited %d first in '%s'" % (in_, element.id)
            fringe = in_node['traceback']
            next_fringe = out_node['traceback'] if 'traceback' in out_node else []
            self.logger.debug("Rating '%s', combining %d new inputs with %d existing paths, competing with %d existing paths",
                              element.id if element else "space", len(textequivs), len(fringe), len(next_fringe))
            for node in fringe:
                # make a copy of parent node for each textequiv alternative
                new_nodes = [Node(parent=node,
                                  state=node.state, # changes during character-wise prediction
                                  value=node.value, # changes during character-wise prediction
                                  cost=0.0,         # accumulates during character-wise prediction
                                  extras=(element, textequiv)) for textequiv in textequivs]
                strings_ = [textequiv.Unicode for textequiv in textequivs] # alternative character sequences
                confidences = [textequiv.conf for textequiv in textequivs] # alternative probabilities
                # advance states and accumulate costs of all alternatives/strings_ (of different length) in parallel:
                for position in range(max_length):
                    # indices to update (next batch):
                    updates = [j for j in range(len(textequivs)) if position < len(strings_[j])]
                    if updates == []:
                        break # no characters left for any textequiv alternative
                    preds, states = self.predict([new_nodes[j].value for j in updates],
                                                 [new_nodes[j].state for j in updates],
                                                 context)
                    for alternative, update in enumerate(updates):
                        new_node = new_nodes[update]
                        string_ = strings_[update]
                        conf = confidences[update]
                        char = string_[position]
                        if char not in self.mapping[0]:
                            if not next_fringe: # avoid repeating the input error for all current candidates
                                self.logger.error('unmapped character "%s" at input alternative %d of element %s',
                                                  char, textequivs[updates[alternative]].index, element.id)
                            idx = 0
                        else:
                            idx = self.mapping[0][char]
                        new_node.value = char
                        new_node.state = states[alternative]
                        new_node.cum_cost += -log(max(preds[alternative][idx], 1e-99), 2) * lm_weight # averaged afterwards/below
                        new_node.cum_cost += -log(max(conf, 1e-99), 2) * (1. - lm_weight) # repeat length times (because of average)
                for new_node, string_ in zip(new_nodes, strings_):
                    new_node.value = string_ # not just last char
                    if beam_clustering_dist and self._history_clustering(new_node, next_fringe, beam_clustering_dist):
                        continue
                    insort_left(next_fringe, new_node) # insert sorted by (unnormalised) cumulative costs
            #self.logger.debug("Shrinking %d paths to best %d", len(next_fringe), beam_width)
            # todo: use some variable length beam threshold
            next_fringe = next_fringe[:beam_width] # keep best paths (cardinality pruning)
            out_node['traceback'] = next_fringe
        assert out == end_node, 'breadth-first search failed to reach true end node (%d instead of %d)' % (out, end_node)
        assert 'traceback' in out_node, "breadth-first search failed to reach end node with any result"
        best = out_node['traceback'][0] # best-scoring path
        result = []
        for node in best.to_sequence()[1:]: # ignore root node
            element, textequiv = node.extras
            textequiv_len = len(textequiv.Unicode)
            score = pow(2.0, -(node.cum_cost-node.parent.cum_cost)/textequiv_len) # average probability
            result.append((element, textequiv, score))
        return result, best.cum_cost
    
    #def rate_all(self, graph, start_node, end_node, push_forward_k=1):
    #    '''Rate a lattice of string alternatives, rescoring all edges.'''
    
    def _history_clustering(self, new_node, next_fringe, distance=5):
        '''Determine whether a node may enter the beam, or has redundant history.
        
        Search hypotheses in `next_fringe` for similarities to `new_node`,
        comparing their hidden layer state (history) vectors, considering them
        similar if the vector norm is below `distance`.
        
        If such hypothesis exists, compare scores: If it is worse than `new_node`,
        then remove it from `next_fringe` right away. Otherwise, return True
        (preventing `new_node` from being inserted).
        
        If no such hypothesis exists, then return False (allowing `new_node`
        to be inserted).
        '''
        for old_node in next_fringe:
            if (new_node.value == old_node.value and
                all(np.linalg.norm(new_node.state[layer] - old_node.state[layer]) < distance
                    for layer in range(self.depth))):
                if old_node.cum_cost < new_node.cum_cost:
                    # self.logger.debug("discarding %s in favour of %s due to history clustering",
                    #                   ''.join([prev_node.extras[1].Unicode for prev_node in new_node.to_sequence()[1:]]),
                    #                   ''.join([prev_node.extras[1].Unicode for prev_node in old_node.to_sequence()[1:]]))
                    return True # continue with next new_node
                else:
                    # self.logger.debug("neglecting %s in favour of %s due to history clustering",
                    #                   ''.join([prev_node.extras[1].Unicode for prev_node in old_node.to_sequence()[1:]]),
                    #                   ''.join([prev_node.extras[1].Unicode for prev_node in new_node.to_sequence()[1:]]))
                    next_fringe.remove(old_node)
                    break # immediately proceed to insert new_node
        return False # proceed to insert new_node (no clustering possible)
    
    def save(self, filename):
        '''Save weights of the trained model, and configuration parameters.
        
        Save both the configured parameters and the trained weights
        of the model into `filename`.
        (This preserves weights across CPU/GPU implementations or input shape configurations.)
        '''
        assert self.status > 1
        self.model.save_weights(filename)
        with h5py.File(filename, 'a') as file:
            group = file.create_group('config')
            group.create_dataset('width', data=np.array(self.width))
            group.create_dataset('depth', data=np.array(self.depth))
            group.create_dataset('length', data=np.array(self.length))
            group.create_dataset('stateful', data=np.array(self.stateful))
            group.create_dataset('variable_length', data=np.array(self.variable_length))
            group.create_dataset('mapping', data=np.fromiter((ord(self.mapping[1][i]) if i in self.mapping[1] else 0
                                                              for i in range(self.voc_size)), dtype='uint32'))
    
    def load_config(self, filename):
        '''Load parameters to prepare configuration/compilation.

        Load model configuration from `filename`.
        '''
        assert self.status == 0
        with h5py.File(filename, 'r') as file:
            group = file['config']
            self.width = group['width'][()]
            self.depth = group['depth'][()]
            self.length = group['length'][()]
            self.stateful = group['stateful'][()]
            self.variable_length = group['variable_length'][()]
            c_i = dict((chr(c), i) for i, c in enumerate(group['mapping'][()]) if c > 0)
            i_c = dict((i, chr(c)) for i, c in enumerate(group['mapping'][()]) if c > 0)
            self.mapping = (c_i, i_c)
            self.voc_size = len(c_i) + 1
    
    def load_weights(self, filename):
        '''Load weights into the configured/compiled model.

        Load weights from `filename` into the compiled and configured model.
        (This preserves weights across CPU/GPU implementations or input shape configurations.)
        '''
        assert self.status > 0
        self.model.load_weights(filename)
        self.status = 2
    
    # data preparation
    def _gen_data_from_files(self, files, steps, split=None, train=False, repeat=False):
        '''Generate numpy arrays suitable for batch processing.
        
        Split the character sequences read from `files` into windows (as configured),
        progressing by `steps` at a time. Yield successive batches of
        input and expected output arrays, accordingly.
        Derive meta-data for context variables from file names.
        
        If `split` is given, then omit windows randomly at a rate equal to
        validation_split (as configured).
        '''
        while True:
            for file in files:
                file.seek(0)
                if self.stateful and train:
                    self.reset_cb.reset(file.name)
                name = os.path.basename(file.name).split('.')[0]
                author = name.split('_')[0]
                year = ceil(int(name.split('_')[-1])/10)
                yield from self._gen_data(_read_normalize_file(file)[0], [year], steps, train, split)
            if not repeat:
                break # causes StopIteration exception if calculated epoch size is too large

    # todo: make iterator thread-safe and then use_multiprocesing=True
    def _gen_data(self, text, context, steps, train=False, split=None):
        '''Generate numpy arrays suitable for batch processing.
        
        Split the character sequence `text` into windows (as configured),
        progressing by `steps` at a time. Yield successive batches of
        input and expected output arrays, accordingly.
        Use the integer list `context` as time-constant context variables,
        or zero-based underspecification.
        
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
        i = 0
        for i in range(self.length if self.stateful else 0, size, steps):
            if isinstance(split, np.ndarray):
                if (split[int(i/steps)] < self.validation_split) == train:
                    # data shared between training and validation: belongs to other generator, resp.
                    continue
                # re-use rest of random number:
                rand = (split[int(i/steps)] - self.validation_split) / (1 - self.validation_split)
            else:
                rand = np.random.uniform(0, 1, 1)[0]
            # make windows:
            if i < self.length:
                if train:
                    if self.variable_length:
                        # below, vectorize() will do zero right-padding (suboptimal)
                        sequences.append(text[0:i])
                    else:
                        continue
                else:
                    # partial window (needs interim batch size 1 for interim length i):
                    yield self._vectorize([text[0:i]], [text[i]], context, length=i, batch_size=1)
                    continue
            else:
                sequences.append(text[i - self.length: i])
            if self.stateful:
                next_chars.append(text[i+1 - self.length: i+1])
            else:
                next_chars.append(text[i])
            if (len(sequences) % self.batch_size == 0 or # next full batch or
                i + steps >= size): # last (partially filled) batch?
                x, y = self._vectorize(sequences, next_chars, context)
                # also train for unmapped characters by random degradation,
                #         or for partial windows by random subsampling:
                if train:
                    # zero degradation for character underspecification:
                    rand_max = 0.01 # effective character degradation ratio
                    if rand < rand_max:
                        j = int((self.length-1) * rand / rand_max) # random position in window
                        x[0][:, j] = 0
                    # zero degradation for context underspecification:
                    rand = (rand - rand_max) / (1 - rand_max) # re-use rest of random number
                    rand_max = 0.1 # effective context degradation ratio
                    if rand < rand_max:
                        j = int((len(x) - 1) * rand / rand_max) + 1 # random context
                        x[j][:, :] = 0
                    rand = (rand - rand_max) / (1 - rand_max) # re-use rest of random number
                    if self.variable_length:
                        rand_max = 0.1 # effective subsampling ratio
                        if rand < rand_max:
                            j = int((self.length-1) * rand / rand_max) + 1 # random length
                            # erase complete batch by sublength from the left ...
                            # to simulate running in with zero padding as in rate():
                            # x[0][:, 0:j] = 0
                            # yield (x, y)
                            # shorten complete batch to sublength from the right ...
                            # to simulate running in with short sequences in rate():
                            yield [z[:, -j:] for z in x], y
                        rand = (rand - rand_max) / (1 - rand_max) # re-use rest of random number
                yield x, y
                sequences = []
                next_chars = []
        if i + steps >= size and steps > 1: # last batch: 1 sample with partial length
            next_chars.append(text[i+1: size])
            sequences.append(text[i: size-1])
            yield self._vectorize(sequences, next_chars, context, batch_size=1) # length=size-i-1 crashes predict_generator in stateful mode (return_sequences)
    
    def _vectorize(self, inputs, outputs=None, contexts=None, length=None, batch_size=None):
        '''Convert a sequence of characters into numpy arrays.
        
        Convert the character sequences in `inputs` to index vectors
        of equal (window) length by zero padding.
        Use the integer list `contexts` as time-constant context variables,
        or zero-based underspecification. Concatenate both inputs.
        If given, convert the character sequences in `outputs` to unit vectors
        likewise.
        
        Return a tuple of input array and output array.
        '''
        if not contexts:
            contexts = self.underspecify_contexts()
        if not length:
            length = self.length
        if not batch_size:
            batch_size = self.batch_size
        # vectorization
        x = np.zeros((batch_size, length), dtype=np.uint32)
        if self.stateful:
            y = np.zeros((batch_size, length, self.voc_size), dtype=np.bool)
        else:
            y = np.zeros((batch_size, self.voc_size), dtype=np.bool)
        zs = [np.zeros((batch_size, length), dtype=np.uint32) for _ in contexts]
        for i, sequence in enumerate(inputs):
            assert i < batch_size, 'input sequence %d (%s) exceeds batch size' % (i, sequence)
            for j, char in enumerate(sequence):
                assert j < length, 'input sequence %d (%s) exceeds window length' % (j, sequence)
                if char not in self.mapping[0]:
                    self.logger.error('unmapped character "%s" at input position %d', char, j + i * length)
                    idx = 0
                else:
                    idx = self.mapping[0][char]
                x[i, j] = idx
                for z, idx in zip(zs, contexts):
                    z[i, j] = idx
                if outputs:
                    if self.stateful:
                        char = outputs[i][j]
                    else:
                        char = outputs[i]
                    if char not in self.mapping[0]:
                        self.logger.error('unmapped character "%s" at output position %d', char, j + i * length)
                        idx = 0
                    else:
                        idx = self.mapping[0][char]
                    if self.stateful:
                        y[i, j, idx] = 1
                    else:
                        y[i, idx] = 1
        return [x] + zs, y
    
    def print_charset(self):
        '''Print the mapped characters, newline-separated.'''
        for i, c in self.mapping[1].items():
            print('%d: "%s"' % (i, c))
            char = unicodedata.normalize('NFC', c)
            if c != char:
                self.logger.warning('mapped character "%s" (%d) should have been normalized to "%s", which is %s mapped',
                                    c, i, char, 'also' if char in self.mapping[0] else 'not')
    
    def plot_char_embeddings_similarity(self, filename):
        '''Paint a heat map of character embeddings.
        
        Calculate the autocorrelation matrix of embedding vectors,
        and plot it as PNG with grayscale colors into `filename`.
        
        (Similar characters should have a higher correlation and
        therefore form groups. Rare or unseen characters will be
        darker and appear random.)
        '''
        logging.getLogger('matplotlib').setLevel(logging.WARNING) # workaround
        from matplotlib import pyplot as plt
        from matplotlib import cm
        
        assert self.status == 2
        charlay = self.model.get_layer(name='char_embedding')
        charwgt = charlay.get_weights()[0]
        charcor = np.tensordot(charwgt, charwgt, (1, 1)) # confusion matrix
        plt.imsave(filename, np.log(np.abs(charcor)), cmap=cm.gray)
    
    def plot_context_embeddings_similarity(self, filename, n=1):
        '''Paint a heat map of context embeddings.
        
        Calculate the autocorrelation matrix of embedding vectors,
        and plot it as PNG with grayscale colors into `filename`.
        
        (Similar contexts should have a higher correlation and
        therefore form groups. Rare or unseen contexts will be
        darker and appear random.)
        '''
        logging.getLogger('matplotlib').setLevel(logging.WARNING) # workaround
        from matplotlib import pyplot as plt
        from matplotlib import cm
        
        assert self.status == 2
        ctxtlay = self.model.get_layer(name='context%d_embedding' % n)
        ctxtwgt = ctxtlay.get_weights()[0]
        ctxtcor = np.tensordot(ctxtwgt, ctxtwgt, (1, 1)) # confusion matrix
        plt.imsave(filename, np.log(np.abs(ctxtcor)), cmap=cm.gray)

    def plot_context_embeddings_projection(self, filename, n=1):
        '''Paint a 2-d PCA projection of context embeddings.
        
        Calculate the principal component analysis for only
        2 components of embedding vectors, and scatter plot
        its vectors with blue crosses (and a red circle for the
        zero/underspecified vector) as PNG into `filename`.
        
        (Similar contexts should lie close to each other and
        therefore form groups. The zero vector should be central.)
        '''
        logging.getLogger('matplotlib').setLevel(logging.WARNING) # workaround
        from matplotlib import pyplot as plt
        from sklearn.decomposition import PCA
        
        assert self.status == 2
        ctxtlay = self.model.get_layer(name='context%d_embedding' % n)
        ctxtwgt = ctxtlay.get_weights()[0]
        ctxtprj = PCA(n_components=2).fit_transform(ctxtwgt) # get a 2-d view
        plt.plot(ctxtprj[:, 0], ctxtprj[:, 1], 'bx') # blue crosses (all)
        plt.plot(ctxtprj[0, 0], ctxtprj[0, 1], 'ro') # red circle (zero)
        plt.savefig(filename)

class StopSignalCallback(Callback):
    '''Keras callback for graceful interruption of training.
    
    Halts training prematurely at the end of the current batch
    when the given signal was received once. If the callback
    gets to receive the signal again, exits immediately.
    '''
    
    def __init__(self, sig=signal.SIGINT, logger=None):
        super(StopSignalCallback, self).__init__()
        self.received = False
        self.sig = sig
        self.logger = logger or logging.getLogger(__name__)
        def stopper(sig, frame):
            if sig == self.sig:
                if self.received: # called again?
                    self.logger.critical('interrupting')
                    exit(0)
                else:
                    self.logger.critical('stopping training')
                    self.received = True
        self.action = signal.signal(self.sig, stopper)
    
    def __del__(self):
        signal.signal(self.sig, self.action)
    
    def on_batch_end(self, batch, logs=None):
        if self.received:
            self.model.stop_training = True

class ResetStatesCallback(Callback):
    '''Keras callback for stateful models to reset state between files.
    
    Callback to be called by `fit_generator()` or even `evaluate_generator()`:
    do `model.reset_states()` whenever generator sees EOF (on_batch_begin with self.eof),
    and between training and validation (on_batch_end with batch>=steps_per_epoch-1).
    '''
    def __init__(self, logger=None):
        super(ResetStatesCallback, self).__init__()
        self.eof = False
        self.here = ''
        self.there = ''
        self.logger = logger or logging.getLogger(__name__)
    
    def reset(self, where):
        '''Reset the model after the end of the current batch.'''
        self.eof = True
        self.there = where
    
    def on_batch_begin(self, batch, logs=None):
        if self.eof:
            # between training files
            self.model.reset_states()
            self.eof = False
            self.here = self.there
    
    def on_batch_end(self, batch, logs=None):
        if logs.get('loss') > 25:
            self.logger.warning('huge loss in "%s" at %d', self.here, batch)
        if np.isnan(logs.get('loss')):
            self.logger.critical('NaN loss in "%s" at %d', self.here, batch)
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

def _read_normalize_file(fd):
    text = unicodedata.normalize('NFC', fd.read())
    size = len(text)
    return text, size
    
