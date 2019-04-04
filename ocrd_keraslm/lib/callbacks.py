import logging
import signal
import numpy as np
from keras.callbacks import Callback

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
        def stopper(sig, _frame):
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
