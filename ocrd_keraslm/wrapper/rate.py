from __future__ import absolute_import

from ocrd import Processor
from ocrd.utils import getLogger
from ocrd_keraslm.wrapper.config import OCRD_TOOL
import ocrd.model.ocrd_page as ocrd_page

log = getLogger('processor.KerasRate')

class KerasRate(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-keraslm-rate']
        kwargs['version'] = OCRD_TOOL['version']
        kwargs['input_file_grp'] = 'OCR-D-GT-SEG-WORD'
        super(KerasRate, self).__init__(*args, **kwargs)

    def process(self):
        """
        Performs the rating.
        """
        for (n, input_file) in enumerate(self.input_files):
            log.info("INPUT FILE %i / %s", n, input_file)
            pcgts = ocrd_page.from_file(self.workspace.download_file(input_file))
