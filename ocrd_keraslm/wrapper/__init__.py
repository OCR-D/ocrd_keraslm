'''wrapper for OCR-D conformance (CLI and workspace processor)

OCRD_TOOL - JSON description of the processor tool
ocrd_keraslm_rate - an ocrd.cli command-line interface
KerasRate - an ocrd.Processor for METS/PAGE-XML workspace data
'''

from .rate import KerasRate
from .cli import ocrd_keraslm_rate
from .config import OCRD_TOOL
