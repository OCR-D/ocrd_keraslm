'''wrapper for OCR-D conformance (CLI and workspace processor)

ocrd_keraslm_rate - an ocrd.cli command-line interface
KerasRate - an ocrd.Processor for METS/PAGE-XML workspace data
'''

from .rate import KerasRate
from .cli import ocrd_keraslm_rate
