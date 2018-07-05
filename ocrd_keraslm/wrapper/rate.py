from __future__ import absolute_import

from ocrd import Processor, MIMETYPE_PAGE
from ocrd.utils import getLogger, concat_padded
from ocrd_keraslm.wrapper.config import OCRD_TOOL
from ocrd.model.ocrd_page import from_file, to_xml

from ocrd_keraslm import lib

log = getLogger('processor.KerasRate')

class KerasRate(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-keraslm-rate']
        kwargs['version'] = OCRD_TOOL['version']
        super(KerasRate, self).__init__(*args, **kwargs)
        self.rater = lib.Rater()
        self.rater.load(self.parameter['mapping'],self.parameter['model'])

    def process(self):
        """
        Performs the rating.
        """
        for (n, input_file) in enumerate(self.input_files):
            log.info("INPUT FILE %i / %s", n, input_file)
            pcgts = from_file(self.workspace.download_file(input_file))
            for region in pcgts.get_Page().get_TextRegion():
                for line in region.get_TextLine():
                    for word in line.get_Word():
                        context = u""
                        for glyph in word.get_Glyph():
                            if glyph.get_TextEquiv():
                                context += glyph.get_TextEquiv()[0].Unicode
                                glyph.set_custom("%.8f" % self.rater.rate_single(context))
            ID = concat_padded(self.output_file_grp, n)
            self.add_output_file(
                ID=ID,
                file_grp=self.output_file_grp,
                basename=ID + '.xml',
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts),
            )
