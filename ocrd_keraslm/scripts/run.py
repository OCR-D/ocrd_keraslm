# -*- coding: utf-8 -*-
from __future__ import absolute_import

import click, sys, json

from ocrd_keraslm import lib

@click.group()
def cli():
    pass

@cli.command(short_help='train a language model')
@click.option('-m', '--model', default="model.weights.h5", help='model weights file', type=click.Path(dir_okay=False, writable=True))
@click.option('-c', '--config', default="model.config.pkl", help='model config file', type=click.Path(dir_okay=False, writable=True))
@click.option('-w', '--width', default=128, help='number of nodes per hidden layer', type=click.IntRange(min=1, max=9128))
@click.option('-d', '--depth', default=2, help='number of hidden layers', type=click.IntRange(min=1, max=10))
@click.option('-l', '--length', default=5, help='number of previous bytes seen (window size)', type=click.IntRange(min=1, max=500))
@click.argument('data', nargs=-1, type=click.File('rb'))
def train(model, config, width, depth, length, data):
    """Train a language model from DATA files,
       with parameters WIDTH, DEPTH, and LENGTH.

       The files will be randomly split into training and validation data.
    """
    
    # train
    rater = lib.Rater()
    rater.width = width
    rater.depth = depth
    rater.length = length
    if rater.stateful: # override necessary before compilation: 
        rater.minibatch_size = rater.length # make sure states are consistent with windows after 1 minibatch
    
    rater.configure()
    rater.train(data)
    
    # save model and dicts
    #rater.save(model)
    rater.save_config(config)
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
    if rater.stateful: # override necessary before compilation: 
        rater.length = 1 # allow single-sample batches
        rater.minibatch_size = rater.length # make sure states are consistent with windows after 1 minibatch
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
    #probs = rater.rate_once(text)
    #click.echo(json.dumps(probs))

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
    if rater.stateful: # override necessary before compilation: 
        rater.minibatch_size = rater.length # make sure states are consistent with windows after 1 minibatch
    rater.configure()
    rater.load_weights(model)
    
    # evaluate on files
    perplexity = rater.test(data)
    click.echo(perplexity)


if __name__ == '__main__':
    cli()
