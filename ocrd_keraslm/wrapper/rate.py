from __future__ import absolute_import
from math import log, ceil

from ocrd import Processor, MIMETYPE_PAGE
from ocrd.utils import getLogger, concat_padded, xywh_from_points, points_from_xywh
from ocrd.model.ocrd_page import from_file, to_xml, GlyphType, CoordsType, TextEquivType
from ocrd.model.ocrd_page_generateds import MetadataItemType, LabelsType, LabelType

from ocrd_keraslm.wrapper.config import OCRD_TOOL
from ocrd_keraslm import lib

LOG = getLogger('processor.KerasRate')

CHOICE_THRESHOLD_NUM = 4 # maximum number of choices to try per element
CHOICE_THRESHOLD_CONF = 0.1 # maximum score drop from best choice to try per element
#beam_width = 100 # maximum number of best partial paths to consider during search with alternative_decoding
BEAM_CLUSTERING_ENABLE = True # enable pruning partial paths by history clustering
BEAM_CLUSTERING_DIST = 5 # maximum distance between state vectors to form a cluster
MAX_LENGTH = 500 # maximum string length of TextEquiv alternatives

class KerasRate(Processor):
    
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-keraslm-rate']
        kwargs['version'] = OCRD_TOOL['version']
        super(KerasRate, self).__init__(*args, **kwargs)
        if not hasattr(self, 'workspace') or not self.workspace: # no parameter/workspace for --dump-json or --version (no processing)
            return
        
        self.rater = lib.Rater(logger=LOG)
        self.rater.load_config(self.parameter['model_file'])
        # overrides necessary before compilation:
        if self.parameter['alternative_decoding']:
            self.rater.stateful = False # no implicit state transfer
            self.rater.incremental = True # but explicit state transfer
        elif self.rater.stateful:
            self.rater.batch_size = 1 # make sure states are consistent with windows after 1 batch
        self.rater.configure()
        self.rater.load_weights(self.parameter['model_file'])
    
    def process(self):
        """
        Performs the rating.
        """
        level = self.parameter['textequiv_level']
        beam_width = self.parameter['beam_width']
        for (n, input_file) in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file)
            pcgts = from_file(self.workspace.download_file(input_file))
            LOG.info("Scoring text in page '%s' at the %s level", pcgts.get_pcGtsId(), level)
            metadata = pcgts.get_Metadata() # ensured by from_file()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=OCRD_TOOL['tools']['ocrd-keraslm-rate']['steps'][0],
                                 value='ocrd-keraslm-rate',
                                 Labels=[LabelsType(externalRef="parameters",
                                                    Label=[LabelType(type_=name,
                                                                     value=self.parameter[name])
                                                           for name in self.parameter.keys()])]))
            # todo: as soon as we have true MODS meta-data in METS (dmdSec/mdWrap/xmlData/mods),
            #       get global context variables from there (e.g. originInfo/dateIssued/@text for year)
            ident = self.workspace.mets.unique_identifier # at least try to get purl
            context = [0]
            if ident:
                name = ident.split('/')[-1]
                year = name.split('_')[-1]
                if year.isnumeric():
                    year = ceil(int(year)/10)
                    context = [year]
                    # todo: author etc
            text = []
            # white space at word boundaries, newline at line/region boundaries: required for LM input
            # FIXME: tokenization ambiguity (i.e. segmenting punctuation into Word suffixes or extra elements)
            #        is handled very differently between GT and ocrd_tesserocr annotation...
            #        we can only reproduce segmentation if Word boundaries exactly coincide with white space!
            regions = pcgts.get_Page().get_TextRegion()
            if not regions:
                LOG.warning("Page contains no text regions")
            first_region = True
            for region in regions:
                if level == 'region':
                    LOG.debug("Getting text in region '%s'", region.id)
                    if not first_region:
                        text.append((None, [TextEquivType(Unicode=u'\n')])) # LM output will not appear in annotation (conf cannot be combined to accurate perplexity from output)
                    first_region = False
                    textequivs = region.get_TextEquiv()
                    if textequivs:
                        text.append((region, filter_choices(textequivs)))
                    else:
                        LOG.warning("Region '%s' contains no text results", region.id)
                    continue
                lines = region.get_TextLine()
                if not lines:
                    LOG.warning("Region '%s' contains no text lines", region.id)
                first_line = True
                for line in lines:
                    if level == 'line':
                        LOG.debug("Getting text in line '%s'", line.id)
                        if not first_line or not first_region:
                            text.append((None, [TextEquivType(Unicode=u'\n')])) # LM output will not appear in annotation (conf cannot be combined to accurate perplexity from output)
                        first_line = False
                        textequivs = line.get_TextEquiv()
                        if textequivs:
                            text.append((line, filter_choices(textequivs)))
                        else:
                            LOG.warning("Line '%s' contains no text results", line.id)
                        continue
                    words = line.get_Word()
                    if not words:
                        LOG.warning("Line '%s' contains no words", line.id)
                    first_word = True
                    for word in words:
                        if level == 'word':
                            LOG.debug("Getting text in word '%s'", word.id)
                            if not first_word:
                                text.append((None, [TextEquivType(Unicode=u' ')])) # LM output will not appear in annotation (conf cannot be combined to accurate perplexity from output)
                            elif not first_line or not first_region:
                                text.append((None, [TextEquivType(Unicode=u'\n')])) # LM output will not appear in annotation (conf cannot be combined to accurate perplexity from output)
                            first_word = False
                            textequivs = word.get_TextEquiv()
                            if textequivs:
                                text.append((word, filter_choices(textequivs)))
                            else:
                                LOG.warning("Word '%s' contains no text results", word.id)
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
                                text.append((None, [space_textequiv])) # LM output will not appear in annotation (conf cannot be combined to accurate perplexity from output)
                        glyphs = word.get_Glyph()
                        if not glyphs:
                            LOG.warning("Word '%s' contains no glyphs", word.id)
                        for glyph in glyphs:
                            LOG.debug("Getting text in glyph '%s'", glyph.id)
                            textequivs = glyph.get_TextEquiv()
                            if textequivs:
                                text.append((glyph, filter_choices(textequivs)))
                            else:
                                LOG.warning("Glyph '%s' contains no text results", glyph.id)
                        first_word = False
                    first_line = False
                first_region = False
            # apply language model to (TextEquiv path in) sequence of elements,
            # remove non-path TextEquivs, modify confidences:
            if self.parameter['alternative_decoding']:
                LOG.info("Rating %d elements including its alternatives", len(text))
                path, entropy = self.rater.rate_best(text, context,
                                                     max_length=MAX_LENGTH,
                                                     beam_width=beam_width,
                                                     beam_clustering_dist=BEAM_CLUSTERING_DIST if BEAM_CLUSTERING_ENABLE else 0)
                strlen = 0
                for element, textequiv, score in path:
                    if element: # not just space
                        element.set_TextEquiv([textequiv]) # delete others
                        strlen += len(textequiv.Unicode)
                        textequiv.set_conf(score)
                        #print(textequiv.Unicode, end='')
                    else:
                        strlen += 1
                        #print(''.join([node.value]), end='')
                #print('')
                ent = entropy/strlen
                avg = pow(2.0, -ent)
                ppl = pow(2.0, ent) # character level
                ppll = pow(2.0, ent * strlen/len(path)) # textequiv level (including spaces/newlines)
                LOG.info("avg: %.3f, char ppl: %.3f, %s ppl: %.3f", avg, ppl, level, ppll) # character need not always equal glyph!
            else:
                textstring = u''.join(textequivs[0].Unicode for element, textequivs in text) # same length as text
                LOG.info("Rating %d elements with a total of %d characters", len(text), len(textstring))
                confidences = self.rater.rate(textstring, context, verbose=0) # much faster
                i = 0
                for element, textequivs in text:
                    textequiv = textequivs[0] # 1st choice only
                    if element:
                        element.set_TextEquiv([textequiv]) # delete others
                    textequiv_len = len(textequiv.Unicode)
                    conf = sum(confidences[i:i+textequiv_len])/textequiv_len # mean probability
                    textequiv.set_conf(conf) # todo: incorporate input confidences, too (weighted product or as input into LM)
                    i += textequiv_len
                if i != len(confidences):
                    LOG.critical("Input text length and output scores length are off by %d characters", i-len(confidences))
                avg = sum(confidences)/len(confidences)
                ent = sum([-log(max(p, 1e-99), 2) for p in confidences])/len(confidences)
                ppl = pow(2.0, ent) # character level
                ppll = pow(2.0, ent * len(confidences)/len(text)) # textequiv level (including spaces/newlines)
                LOG.info("avg: %.3f, char ppl: %.3f, %s ppl: %.3f", avg, ppl, level, ppll) # character need not always equal glyph!
            # ensure parent textequivs are up to date:
            regions = pcgts.get_Page().get_TextRegion()
            if level != 'region':
                for region in regions:
                    lines = region.get_TextLine()
                    if level != 'line':
                        for line in lines:
                            words = line.get_Word()
                            if level != 'word':
                                for word in words:
                                    glyphs = word.get_Glyph()
                                    word_unicode = u''.join(glyph.get_TextEquiv()[0].Unicode for glyph in glyphs)
                                    word.set_TextEquiv([TextEquivType(Unicode=word_unicode)]) # remove old
                            line_unicode = u' '.join(word.get_TextEquiv()[0].Unicode for word in words)
                            line.set_TextEquiv([TextEquivType(Unicode=line_unicode)]) # remove old
                    region_unicode = u'\n'.join(line.get_TextEquiv()[0].Unicode for line in lines)
                    region.set_TextEquiv([TextEquivType(Unicode=region_unicode)]) # remove old
            file_id = concat_padded(self.output_file_grp, n)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                basename=file_id + '.xml', # with suffix or bare?
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts),
            )

def filter_choices(textequivs):
    '''assuming `textequivs` are already sorted by input confidence (conf attribute), ensure maximum number and maximum relative threshold'''
    textequivs = textequivs[:min(CHOICE_THRESHOLD_NUM, len(textequivs))]
    if textequivs:
        conf0 = textequivs[0].conf
        if conf0:
            return [te for te in textequivs if conf0 - te.conf < CHOICE_THRESHOLD_CONF]
        else:
            return textequivs
    else:
        return []
