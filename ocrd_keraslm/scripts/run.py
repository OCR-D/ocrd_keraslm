# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import click, sys, json
import numpy, codecs
from bisect import insort_left

from ocrd_keraslm import lib

@click.group()
def cli():
    pass

@cli.command(short_help='train a language model')
@click.option('-m', '--model', default="model.weights.h5", help='model weights file', type=click.Path(dir_okay=False, writable=True))
@click.option('-c', '--config', default="model.config.pkl", help='model config file', type=click.Path(dir_okay=False, writable=True))
@click.option('-w', '--width', default=128, help='number of nodes per hidden layer', type=click.IntRange(min=1, max=9128))
@click.option('-d', '--depth', default=2, help='number of hidden layers', type=click.IntRange(min=1, max=10))
@click.option('-l', '--length', default=256, help='number of previous bytes seen (window size)', type=click.IntRange(min=1, max=1024))
@click.option('-v', '--val-data', default=None, help='directory containing validation data files (no split)', type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument('data', nargs=-1, type=click.File('rb'))
def train(model, config, width, depth, length, val_data, data):
    """Train a language model from DATA files,
       with parameters WIDTH, DEPTH, and LENGTH.

       The files will be randomly split into training and validation data,
       except if VAL_DATA is given.
    """
    
    # train
    rater = lib.Rater()
    incremental = False
    if os.path.isfile(model) and os.path.isfile(config):
        rater.load_config(config)
        if rater.width == width and rater.depth == depth:
            incremental = True
    rater.width = width
    rater.depth = depth
    rater.length = length
    
    rater.configure()
    if incremental:
        print ('loading weights for incremental training')
        rater.load_weights(model)
    if val_data:
        val_files = [os.path.join(val_data, f) for f in os.listdir(val_data)]
        val_data = [open(f, mode='rb') for f in val_files if os.path.isfile(f)]
    rater.train(data, val_data=val_data)
    
    # save model and dicts
    #rater.save(model)
    rater.save_config(config)
    assert rater.status == 2
    if os.path.isfile("model_last.weights.h5"):
        # use best-scoring weights from last checkpoint
        rename("model_last.weights.h5", model) # mv
    else:
        rater.save_weights(model)

@cli.command(short_help='get individual probabilities from language model')
@click.option('-m', '--model', required=True, help='model weights file', type=click.Path(dir_okay=False, exists=True))
@click.option('-c', '--config', required=True, help='model config file', type=click.Path(dir_okay=False, exists=True))
@click.argument('text', type=click.STRING) # todo: create custom click.ParamType for graph/FST input
def apply(model, config, text):
    """Apply a language model to TEXT string and compute its individual probabilities.

       If TEXT is the symbol '-', the string will be read from standard input.
    """
    
    # load model
    rater = lib.Rater()
    #rater.load(model)
    rater.load_config(config)
    rater.configure()
    rater.load_weights(model)
    
    if text:
        if text[0] == u"-":
            text = sys.stdin.read()
    else:
        pass
    
    ratings, perplexity = rater.rate(text)
    click.echo(perplexity)
    click.echo(json.dumps(ratings, ensure_ascii=False))
    # much faster:
    #probs = rater.rate_once(text)
    #click.echo(json.dumps(zip(text, probs)))

@cli.command(short_help='get overall perplexity from language model')
@click.option('-m', '--model', required=True, help='model weights file', type=click.Path(dir_okay=False, exists=True))
@click.option('-c', '--config', required=True, help='model config file', type=click.Path(dir_okay=False, exists=True))
@click.argument('data', nargs=-1, type=click.File('rb'))
def test(model, config, data):
    """Apply a language model to DATA files and compute its overall perplexity."""
    
    # load model
    rater = lib.Rater()
    #rater.load(model)
    rater.load_config(config)
    rater.configure()
    rater.load_weights(model)
    
    # evaluate on files
    perplexity = rater.test(data)
    click.echo(perplexity)

@cli.command(short_help='sample characters from language model')
@click.option('-m', '--model', required=True, help='model weights file', type=click.Path(dir_okay=False, exists=True))
@click.option('-c', '--config', required=True, help='model config file', type=click.Path(dir_okay=False, exists=True))
@click.option('-n', '--number', default=1, help='number of bytes to sample', type=click.IntRange(min=1, max=10000))
@click.argument('context', type=click.STRING)
# todo: also allow specifying follow-up context
def generate(model, config, number, context):
    """Apply a language model, generating the most probable characters (starting with CONTEXT string)."""

    # load model
    rater = lib.Rater()
    rater.load_config(config)
    rater.stateful = False # no implicit state transfer
    rater.incremental = True # but explicit state transfer
    rater.configure()
    rater.load_weights(model)

    # initial state
    context_states = [None]
    # context (to get correct initial state)
    context_bytes = context.encode("utf-8")
    for b in context_bytes[:-1]: # all but last byte
        _, context_states = rater.rate_single([b], context_states)
    decoder = codecs.getincrementaldecoder('utf-8')()
    decoder.decode(context_bytes)
    next_fringe = [lib.Node(parent=None,
                            state=context_states[0],
                            value=context_bytes[-1], # last byte
                            cost=0.0,
                            extras=decoder.getstate())]
    # beam search
    for i in range(0,number): # iterate over number of bytes to be generated
        fringe = next_fringe
        preds, states = rater.rate_single([n.value for n in fringe], [n.state for n in fringe])
        next_fringe = []
        for j, n in enumerate(fringe): # iterate over batch
            pred = preds[j]
            pred_best = numpy.argsort(pred)[-10:] # keep only 10-best alternatives
            pred_best = pred_best[numpy.searchsorted(pred[pred_best], 0.004):] # keep only alternatives better than 1/256 (uniform distribution)
            costs = -numpy.log(pred[pred_best])
            state = states[j]
            for best, cost in zip(pred_best, costs): # follow up on best predictions
                decoder.setstate(n.extras)
                try:
                    decoder.decode(bytes([best]))
                    n_new = lib.Node(parent=n, state=state, value=best, cost=cost, extras=decoder.getstate())
                    insort_left(next_fringe, n_new) # add alternative to tree
                except UnicodeDecodeError:
                    pass # ignore this alternative
        next_fringe = next_fringe[:256] # keep 256-best paths (equals batch size)
    # todo: keep only candidates with clean decoder state (no partial codepoints), then resort
    best = next_fringe[0] # best-scoring
    result = bytes([n.value for n in best.to_sequence()])
    click.echo(context_bytes[:-1] + result)
    

if __name__ == '__main__':
    cli()
