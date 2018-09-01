from __future__ import absolute_import
from math import log

from ocrd import Processor, MIMETYPE_PAGE
from ocrd.utils import getLogger, concat_padded, xywh_from_points, points_from_xywh
from ocrd_keraslm.wrapper.config import OCRD_TOOL
from ocrd.model.ocrd_page import from_file, to_xml, GlyphType, CoordsType, TextEquivType
from ocrd.model.ocrd_page_generateds import MetadataItemType, LabelsType, LabelType

from ocrd_keraslm import lib

logger = getLogger('processor.KerasRate')

class KerasRate(Processor):
    
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-keraslm-rate']
        kwargs['version'] = OCRD_TOOL['version']
        super(KerasRate, self).__init__(*args, **kwargs)
        if not hasattr(self, 'workspace') or not self.workspace: # no parameter/workspace for --dump-json or --version (no processing)
            return
        
        self.rater = lib.Rater()
        self.rater.load_config(self.parameter['config_file'])
        if self.rater.stateful: # override necessary before compilation: 
            self.rater.length = 1 # allow single-sample batches
            self.rater.minibatch_size = self.rater.length # make sure states are consistent with windows after 1 minibatch
        self.rater.configure()
        self.rater.load_weights(self.parameter['weight_file'])
    
    def process(self):
        """
        Performs the rating.
        """
        level = self.parameter['textequiv_level']
        for (n, input_file) in enumerate(self.input_files):
            logger.info("INPUT FILE %i / %s", n, input_file)
            pcgts = from_file(self.workspace.download_file(input_file))
            logger.info("Scoring text in page '%s' at the %s level", pcgts.get_pcGtsId(), level)
            metadata = pcgts.get_Metadata() # ensured by from_file()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=OCRD_TOOL['tools']['ocrd-keraslm-rate']['steps'][0],
                                 value='ocrd-keraslm-rate',
                                 Labels=[LabelsType(externalRef="parameters",
                                                    Label=[LabelType(type_=name,
                                                                     value=self.parameter[name])
                                                           for name in self.parameter.keys()])]))
            text = []
            # white space at word boundaries, newline at line/region boundaries: required for LM input
            # FIXME: tokenization ambiguity (i.e. segmenting punctuation into Word suffixes or extra elements)
            #        is handled very differently between GT and ocrd_tesserocr annotation...
            #        we can only reproduce segmentation if Word boundaries exactly coincide with white space!
            regions = pcgts.get_Page().get_TextRegion()
            if not regions:
                logger.warn("Page contains no text regions")
            first_region = True
            for region in regions:
                if level == 'region':
                    logger.debug("Getting text in region '%s'", region.id)
                    if not first_region:
                        text.append(TextEquivType(Unicode=u'\n')) # LM output will not appear in annotation (conf cannot be combined to accurate perplexity from output)
                    first_region = False
                    textequivs = region.get_TextEquiv()
                    if textequivs:
                        text.append(textequivs[0]) # only 1-best for now (otherwise we need to do beam search and return paths)
                    else:
                        logger.warn("Region '%s' contains no text results", region.id)
                    continue
                lines = region.get_TextLine()
                if not lines:
                    logger.warn("Region '%s' contains no text lines", region.id)
                first_line = True
                for line in lines:
                    if level == 'line':
                        logger.debug("Getting text in line '%s'", line.id)
                        if not first_line or not first_region:
                            text.append(TextEquivType(Unicode=u'\n')) # LM output will not appear in annotation (conf cannot be combined to accurate perplexity from output)
                        first_line = False
                        textequivs = line.get_TextEquiv()
                        if textequivs:
                            text.append(textequivs[0]) # only 1-best for now (otherwise we need to do beam search and return paths)
                        else:
                            logger.warn("Line '%s' contains no text results", line.id)
                        continue
                    words = line.get_Word()
                    if not words:
                        logger.warn("Line '%s' contains no words", line.id)
                    first_word = True
                    for word in words:
                        if level == 'word':
                            logger.debug("Getting text in word '%s'", word.id)
                            if not first_word:
                                text.append(TextEquivType(Unicode=u' ')) # LM output will not appear in annotation (conf cannot be combined to accurate perplexity from output)
                            elif not first_line or not first_region:
                                text.append(TextEquivType(Unicode=u'\n')) # LM output will not appear in annotation (conf cannot be combined to accurate perplexity from output)
                            first_word = False
                            textequivs = word.get_TextEquiv()
                            if textequivs:
                                text.append(textequivs[0]) # only 1-best for now (otherwise we need to do beam search and return paths)
                            else:
                                logger.warn("Word '%s' contains no text results", word.id)
                            continue
                        space_char = None
                        if not first_word:
                            space_char = u' '
                        elif not first_line or not first_region:
                            space_char = u'\n'
                        if space_char: # space required for LM input
                            space_textequiv = TextEquivType(Unicode=space_char)
                            if self.parameter['add_space_glyphs']:
                                xywh = xywh_from_points(word.get_Coords().points)
                                xywh['w'] = 0
                                xywh['h'] = 0
                                space_glyph = GlyphType(id='%s_space' % word.id, 
                                                        TextEquiv=[space_textequiv], 
                                                        Coords=CoordsType(points_from_xywh(xywh))) # empty box
                                word.insert_Glyph_at(0, space_glyph) # add a pseudo glyph in annotation
                            else:
                                text.append(space_textequiv) # LM output will not appear in annotation (conf cannot be combined to accurate perplexity from output)
                        glyphs = word.get_Glyph()
                        if not glyphs:
                            logger.warn("Word '%s' contains no glyphs", word.id)
                        for glyph in glyphs:
                            logger.debug("Getting text in glyph '%s'", glyph.id)
                            textequivs = glyph.get_TextEquiv()
                            if textequivs:
                                text.append(textequivs[0]) # only 1-best for now (otherwise we need to do beam search and return paths)
                            else:
                                logger.warn("Glyph '%s' contains no text results", glyph.id)
                        first_word = False
                    first_line = False
                first_region = False
            textstring = u''.join(te.Unicode for te in text) # same length as text
            logger.debug("Rating %d characters", len(textstring))
            confidences = self.rater.rate_once(textstring)
            avg = sum(confidences)/len(confidences)
            ent = sum([-log(p, 2) for p in confidences])/len(confidences)
            ppl = pow(2.0, ent) # character level
            ppll = pow(2.0, ent * len(confidences)/len(text)) # textequiv level
            logger.debug("avg: %.3f, char ppl: %.3f, %s ppl: %.3f", avg, ppl, level, ppll) # char need not always equal glyph!
            i = 0
            for textequiv in text:
                j = len(textequiv.Unicode)
                conf = sum(confidences[i:i+j])/j
                textequiv.set_conf(conf) # or multiply to existing value?
                i += j
            if i != len(confidences):
                logger.err("Input text length and output scores length are off by %d characters", i-len(confidences))
            ID = concat_padded(self.output_file_grp, n)
            self.workspace.add_file(
                ID=ID,
                file_grp=self.output_file_grp,
                basename=ID + '.xml',
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts),
            )
