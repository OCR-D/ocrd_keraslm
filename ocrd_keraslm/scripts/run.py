# -*- coding: utf-8 -*-
from __future__ import absolute_import

import click, sys, json

from ocrd_keraslm import lib

@click.group()
def cli():
    pass

@cli.command()
@click.option('-o', '--output', default="model")
@click.argument('data', nargs=-1, type=click.File('rb'))
def train(output, data):
    """Train a language model from `data` files"""
    
    # train
    rater = lib.Rater()
    rater.train(data)
    
    # save model and dicts
    rater.save(output)

@cli.command()
@click.option('-m', '--model', required=True)
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
@click.option('-m', '--model', required=True)
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
