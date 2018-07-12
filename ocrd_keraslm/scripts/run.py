# -*- coding: utf-8 -*-
from __future__ import absolute_import

import click, sys, json

from ocrd_keraslm import lib

@click.group()
def cli():
    pass

@cli.command()
@click.option('-m', '--model', default="model.h5", type=click.Path(dir_okay=False, writable=True))
@click.option('-w', '--width', default=128, type=click.IntRange(min=1, max=9128))
@click.option('-d', '--depth', default=2, type=click.IntRange(min=1, max=10))
@click.option('-l', '--length', default=5, type=click.IntRange(min=1, max=500))
@click.argument('data', nargs=-1, type=click.File('rb'))
def train(model, width, depth, length, data):
    """Train a language model from `data` files,
       with `width` nodes per hidden layer,
       with `depth` hidden layers,
       with `length` bytes window size.
    """
    
    # train
    rater = lib.Rater()
    rater.train(data, width, depth, length)
    
    # save model and dicts
    rater.save(model)

@cli.command()
@click.option('-m', '--model', required=True, type=click.Path(dir_okay=False, exists=True))
@click.argument('text', type=click.STRING) # todo: create custom click.ParamType for graph/FST input
def apply(model, text):
    """Apply a language model to `text` string"""
    
    # load model
    rater = lib.Rater()
    rater.load(model)
    
    if text:
        if text[0] == u"-":
            text = sys.stdin.read()
    else:
        pass
    
    ratings, perplexity = rater.rate(text)
    click.echo(perplexity)
    click.echo(json.dumps(ratings, ensure_ascii=False))

@cli.command()
@click.option('-m', '--model', required=True, type=click.Path(dir_okay=False, exists=True))
@click.argument('data', nargs=-1, type=click.File('rb'))
def test(model, data):
    """Apply a language model to `data` files"""
    
    # load model
    rater = lib.Rater()
    rater.load(model)
    
    # evaluate on files
    perplexity = rater.test(data)
    click.echo(perplexity)


if __name__ == '__main__':
    cli()
