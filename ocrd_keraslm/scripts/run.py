# -*- coding: utf-8 -*-
from __future__ import absolute_import

import click, sys, json

from ocrd_keraslm import lib

@click.group()
def cli():
    pass

@cli.command()
@click.option('-o', '--output', default="model")
@click.argument('data')
def train(output, data):
    """Train a language model"""

    # train
    rater = lib.Rater()
    with open(str(data),"r") as f:
        training_data = f.read()

        rater.train(training_data)

    # save model and dicts
    rater.save(output)

@cli.command()
@click.option('-m', '--model', required=True)
@click.option('-M', '--mapping', required=True)
@click.argument('TEXT')
def apply(model, mapping, text):
    """Apply a language model"""

    # load model
    rater = lib.Rater()
    rater.load(mapping,model)

    # read input
    in_string = u""
    if text:
        if text[0] == u"-":
            text = sys.stdin
        for line in text:
            in_string += line
    else:
        pass

    ratings = rater.rate(in_string)
    click.echo(json.dumps(ratings))

if __name__ == '__main__':
    cli()
