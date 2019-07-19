import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from . import KerasRate

@click.command()
@ocrd_cli_options
def ocrd_keraslm_rate(*args, **kwargs):
    return ocrd_cli_wrap_processor(KerasRate, *args, **kwargs)
