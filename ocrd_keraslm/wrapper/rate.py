from __future__ import absolute_import

from ocrd import Processor
from ocrd.utils import getLogger
from ocrd_keraslm.wrapper.config import OCRD_TOOL

log = getLogger('processor.KerasRate')

class KerasRate(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-keraslm-rate']
        kwargs['version'] = OCRD_TOOL['version']
        super(KerasRate, self).__init__(*args, **kwargs)

    def process(self):
        """
        Performs the rating.
        """
        pass
