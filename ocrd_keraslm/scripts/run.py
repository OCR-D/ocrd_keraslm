# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import sys
import codecs
import logging
from bisect import insort_left
from math import ceil
import json
import click
import numpy

from ocrd_keraslm import lib

class SortedGroup(click.Group):
    def list_commands(self, ctx):
        commands = set(super(SortedGroup, self).list_commands(ctx))
        commands0 = ['train', 'test', 'apply', 'generate', 'print-charset',
                     'plot-char-embeddings-similarity', 'plot-context-embeddings-similarity',
                     'plot-context-embeddings-projection']
        commands0.extend(list(commands.difference(set(commands0))))
        return commands0

@click.group(cls=SortedGroup)
def cli():
    logging.basicConfig(level=logging.DEBUG)
    #pass

@cli.command(short_help='train a language model')
@click.option('-m', '--model', default="model.h5", help='model file', type=click.Path(dir_okay=False, writable=True))
@click.option('-w', '--width', default=128, help='number of nodes per hidden layer', type=click.IntRange(min=1, max=9128))
@click.option('-d', '--depth', default=2, help='number of hidden layers', type=click.IntRange(min=1, max=10))
@click.option('-l', '--length', default=256, help='number of previous characters seen (window size)', type=click.IntRange(min=1, max=1024))
@click.option('-v', '--val-data', default=None, help='directory containing validation data files (no split)', type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument('data', nargs=-1, type=click.File('r'))
def train(model, width, depth, length, val_data, data):
    """Train a language model from DATA files,
       with parameters WIDTH, DEPTH, and LENGTH.

       The files will be randomly split into training and validation data,
       except if VAL_DATA is given.
    """
    
    # train
    rater = lib.Rater()
    incremental = False
    if os.path.isfile(model):
        rater.load_config(model)
        if rater.width == width and rater.depth == depth:
            incremental = True
            print('loading weights from existing model for incremental training')
        else:
            print('warning: ignoring existing model due to different topology (width=%d, depth=%d)' % (rater.width, rater.depth), file=sys.stderr)
    rater.width = width
    rater.depth = depth
    rater.length = length
    
    rater.configure()
    if incremental:
        rater.load_weights(model)
    if val_data:
        val_files = [os.path.join(val_data, f) for f in os.listdir(val_data)]
        val_data = [open(f, mode='r') for f in val_files if os.path.isfile(f)]
    rater.train(data, val_data=val_data)
    assert rater.status == 2
    
    # save model and config
    rater.save(model)

@cli.command(short_help='get individual probabilities from language model')
@click.option('-m', '--model', required=True, help='model file', type=click.Path(dir_okay=False, exists=True))
@click.option('-c', '--context', default=None, help='constant meta-data input')
@click.argument('text', type=click.STRING) # todo: create custom click.ParamType for graph/FST input
def apply(model, text, context):
    """Apply a language model to TEXT string and compute its individual probabilities.

       If TEXT is the symbol '-', the string will be read from standard input.
    """
    
    # load model
    rater = lib.Rater()
    rater.load_config(model)
    rater.configure()
    rater.load_weights(model)
    
    if text and text[0] == u"-":
        text = sys.stdin.read()
    if context:
        context = list(map(lambda x: ceil(int(x)/10), context.split(' ')))
    
    ratings, perplexity = rater.rate(text, context)
    click.echo(perplexity)
    click.echo(json.dumps(ratings, ensure_ascii=False))
    # much faster:
    #probs = rater.rate_once(text)
    #click.echo(json.dumps(zip(text, probs)))

@cli.command(short_help='get overall perplexity from language model')
@click.option('-m', '--model', required=True, help='model file', type=click.Path(dir_okay=False, exists=True))
@click.argument('data', nargs=-1, type=click.File('r'))
def test(model, data):
    """Apply a language model to DATA files and compute its overall perplexity."""
    
    # load model
    rater = lib.Rater()
    rater.load_config(model)
    rater.configure()
    rater.load_weights(model)
    
    # evaluate on files
    perplexity = rater.test(data)
    click.echo(perplexity)

@cli.command(short_help='sample characters from language model')
@click.option('-m', '--model', required=True, help='model file', type=click.Path(dir_okay=False, exists=True))
@click.option('-n', '--number', default=1, help='number of characters to sample', type=click.IntRange(min=1, max=10000))
@click.option('-c', '--context', default=None, help='constant meta-data input')
@click.argument('prefix', type=click.STRING)
# todo: also allow specifying suffix
def generate(model, number, prefix, context):
    """Apply a language model, generating the most probable characters (starting with PREFIX string)."""

    # load model
    rater = lib.Rater()
    rater.load_config(model)
    rater.stateful = False # no implicit state transfer
    rater.incremental = True # but explicit state transfer
    rater.configure()
    rater.load_weights(model)
    
    if context:
        context = list(map(lambda x: ceil(int(x)/10), context.split(' ')))
    else:
        context = rater.underspecify_contexts()
        
    # initial state
    prefix_states = [None]
    # prefix (to get correct initial state)
    for char in prefix[:-1]: # all but last character
        _, prefix_states = rater.predict([char], prefix_states, context=context)
    next_fringe = [lib.Node(state=prefix_states[0],
                            value=prefix[-1], # last character
                            cost=0.0)]
    # beam search
    for _ in range(0, number): # iterate over number of characters to be generated
        fringe = next_fringe
        preds, states = rater.predict([n.value for n in fringe], [n.state for n in fringe], context=context)
        next_fringe = []
        for j, n in enumerate(fringe): # iterate over batch
            pred = preds[j]
            pred_best = numpy.argsort(pred)[-10:] # keep only 10-best alternatives
            pred_best = pred_best[numpy.searchsorted(pred[pred_best], 0.004):] # keep only alternatives better than 1/256 (uniform distribution)
            costs = -numpy.log(pred[pred_best])
            state = states[j]
            for best, cost in zip(pred_best, costs): # follow up on best predictions
                if best not in rater.mapping[1]: # avoid zero/unmapped
                    continue # ignore this alternative
                n_new = lib.Node(parent=n, state=state, value=rater.mapping[1][best], cost=cost)
                insort_left(next_fringe, n_new) # add alternative to tree
        next_fringe = next_fringe[:256] # keep 256-best paths (equals batch size)
    best = next_fringe[0] # best-scoring
    result = ''.join([n.value for n in best.to_sequence()])
    click.echo(prefix[:-1] + result)

@cli.command(short_help='Print the mapped characters')
@click.option('-m', '--model', required=True, help='model file', type=click.Path(dir_okay=False, exists=True))
def print_charset(model):
    rater = lib.Rater()
    rater.load_config(model)
    rater.print_charset()

@cli.command(short_help='Paint a heat map of character embeddings')
@click.option('-m', '--model', required=True, help='model file', type=click.Path(dir_okay=False, exists=True))
@click.argument('filename', type=click.Path(dir_okay=False, writable=True))
def plot_char_embeddings_similarity(model, filename):
    rater = lib.Rater()
    rater.load_config(model)
    rater.configure()
    rater.load_weights(model)
    rater.plot_char_embeddings_similarity(filename)

@cli.command(short_help='Paint a heat map of context embeddings')
@click.option('-m', '--model', required=True, help='model file', type=click.Path(dir_okay=False, exists=True))
@click.option('-n', '--number', default=1, help='which context variable', type=click.IntRange(min=1, max=100)) # see lib for contexts actually available
@click.argument('filename', type=click.Path(dir_okay=False, writable=True))
def plot_context_embeddings_similarity(model, filename, number):
    rater = lib.Rater()
    rater.load_config(model)
    rater.configure()
    rater.load_weights(model)
    rater.plot_context_embeddings_similarity(filename, n=number)

@cli.command(short_help='Paint a 2-d PCA projection of context embeddings')
@click.option('-m', '--model', required=True, help='model file', type=click.Path(dir_okay=False, exists=True))
@click.option('-n', '--number', default=1, help='which context variable', type=click.IntRange(min=1, max=100)) # see lib for contexts actually available
@click.argument('filename', type=click.Path(dir_okay=False, writable=True))
def plot_context_embeddings_projection(model, filename, number):
    rater = lib.Rater()
    rater.load_config(model)
    rater.configure()
    rater.load_weights(model)
    rater.plot_context_embeddings_projection(filename, n=number)

if __name__ == '__main__':
    cli()
