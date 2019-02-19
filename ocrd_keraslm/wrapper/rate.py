from __future__ import absolute_import
from math import log, ceil

from ocrd import Processor, MIMETYPE_PAGE
from ocrd.validator.page_validator import PageValidator, ConsistencyError
from ocrd.utils import getLogger, concat_padded, xywh_from_points, points_from_xywh
from ocrd.model.ocrd_page import from_file, to_xml, GlyphType, CoordsType, TextEquivType
from ocrd.model.ocrd_page_generateds import MetadataItemType, LabelsType, LabelType

import networkx as nx

from ocrd_keraslm.wrapper.config import OCRD_TOOL
from ocrd_keraslm import lib

LOG = getLogger('processor.KerasRate')

CHOICE_THRESHOLD_NUM = 4 # maximum number of choices to try per element
CHOICE_THRESHOLD_CONF = 0.1 # maximum score drop from best choice to try per element
#beam_width = 100 # maximum number of best partial paths to consider during search with alternative_decoding
BEAM_CLUSTERING_ENABLE = True # enable pruning partial paths by history clustering
BEAM_CLUSTERING_DIST = 5 # maximum distance between state vectors to form a cluster
MAX_LENGTH = 500 # maximum string length of TextEquiv alternatives

# similar to ocrd.validator.page_validator._HIERARCHY:
_HIERARCHY = {'Page': 'region', 'TextRegion': 'line', 'TextLine': 'word', 'Word': 'glyph', 'Glyph': ''}

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
            # create a graph from the linear sequence of elements at the given level
            graph = nx.DiGraph(level=level) # initialise directed unigraph with global attribute level
            graph.add_node(0)
            start_node = 0
            # white space IFF between words, newline IFF between lines/regions: required for LM input
            # as a minor mitigation, try to guess consistency a text annotation on multiple levels
            # (i.e. infer wrong tokenisation when mother node has TextEquiv deviating from
            #  concatenated child node TextEquivs only w.r.t. white-space):
            report = PageValidator.validate(ocrd_page=pcgts, strictness='strict')
            problems = {}
            if not report.is_valid:
                LOG.warning("Page validation failed: %s", report.to_xml())
                if report.errors:
                    for err in report.errors:
                        if (isinstance(err, ConsistencyError) and
                            _HIERARCHY[err.tag] == level and # relevant for current processing level
                            err.actual and # not just a missing TextEquiv on super level
                            len(err.actual.split()) != len(err.expected.split())): # only tokenisation
                            problems[err.ID] = err
            regions = pcgts.get_Page().get_TextRegion()
            if not regions:
                LOG.warning("Page contains no text regions")
            page_start_node = start_node
            first_region = True
            for region in regions:
                if level == 'region':
                    graph.add_node(start_node + 1)
                    LOG.debug("Getting text in region '%s'", region.id)
                    textequivs = region.get_TextEquiv()
                    # fixme: this won't happen as long as ocrd.validator.page_validator
                    #        does not use the pcgts id (as there is no page.id)!
                    if textequivs and repair_tokenisation(problems, pcgts.get_pcGtsId(), graph, 0, textequivs[0].Unicode, first_region):
                        # likely cases: no newlines, or line feed
                        pass # skip all rules for concatenation joints
                    elif not first_region:
                        graph.add_edge(start_node, start_node + 1, element=None, alternatives=[TextEquivType(Unicode=u'\n', conf=1.0)]) # LM output will not appear in annotation (conf cannot be combined to accurate perplexity from output)
                        start_node += 1
                        graph.add_node(start_node + 1)
                    if textequivs:
                        graph.add_edge(start_node, start_node + 1, element=region, alternatives=filter_choices(textequivs))
                        start_node += 1
                    else:
                        LOG.warning("Region '%s' contains no text results", region.id)
                    first_region = False
                    continue
                lines = region.get_TextLine()
                if not lines:
                    LOG.warning("Region '%s' contains no text lines", region.id)
                region_start_node = start_node
                first_line = True
                for line in lines:
                    if level == 'line':
                        graph.add_node(start_node + 1)
                        LOG.debug("Getting text in line '%s'", line.id)
                        textequivs = line.get_TextEquiv()
                        if textequivs and repair_tokenisation(problems, region.id, graph, region_start_node, textequivs[0].Unicode, first_line):
                            # likely cases: extra trailing newlines, or carriage-return
                            pass # skip all rules for concatenation joints
                        elif not first_line or not first_region:
                            graph.add_edge(start_node, start_node + 1, element=None, alternatives=[TextEquivType(Unicode=u'\n', conf=1.0)]) # LM output will not appear in annotation (conf cannot be combined to accurate perplexity from output)
                            start_node += 1
                            graph.add_node(start_node + 1)
                        if textequivs:
                            graph.add_edge(start_node, start_node + 1, element=line, alternatives=filter_choices(textequivs))
                            start_node += 1
                        else:
                            LOG.warning("Line '%s' contains no text results", line.id)
                        first_line = False
                        continue
                    words = line.get_Word()
                    if not words:
                        LOG.warning("Line '%s' contains no words", line.id)
                    line_start_node = start_node
                    first_word = True
                    for word in words:
                        space_char = None
                        textequivs = word.get_TextEquiv()
                        if textequivs and repair_tokenisation(problems, line.id, graph, line_start_node, textequivs[0].Unicode, first_word):
                            # likely cases: no spaces
                            pass # skip all rules for concatenation joints
                        elif not first_word:
                            space_char = u' '
                        elif not first_line or not first_region:
                            space_char = u'\n'
                        if space_char: # space required for LM input
                            graph.add_node(start_node + 1)
                            graph.add_edge(start_node, start_node + 1, element=None, alternatives=[TextEquivType(Unicode=space_char, conf=1.0)]) # LM output will not appear in annotation (conf cannot be combined to accurate perplexity from output)
                            start_node += 1
                        if level == 'word':
                            graph.add_node(start_node + 1)
                            LOG.debug("Getting text in word '%s'", word.id)
                            if textequivs:
                                graph.add_edge(start_node, start_node + 1, element=word, alternatives=filter_choices(textequivs))
                                start_node += 1
                            else:
                                LOG.warning("Word '%s' contains no text results", word.id)
                            first_word = False
                            continue
                        glyphs = word.get_Glyph()
                        if not glyphs:
                            LOG.warning("Word '%s' contains no glyphs", word.id)
                        for glyph in glyphs:
                            graph.add_node(start_node + 1)
                            LOG.debug("Getting text in glyph '%s'", glyph.id)
                            textequivs = glyph.get_TextEquiv()
                            if textequivs:
                                graph.add_edge(start_node, start_node + 1, element=glyph, alternatives=filter_choices(textequivs))
                                start_node += 1
                            else:
                                LOG.warning("Glyph '%s' contains no text results", glyph.id)
                        first_word = False
                    first_line = False
                first_region = False
            # apply language model to (TextEquiv path in) graph,
            # remove non-path TextEquivs, modify confidences:
            pathlen = start_node - page_start_node
            if self.parameter['alternative_decoding']:
                LOG.info("Rating %d elements including its alternatives", pathlen)
                path, entropy = self.rater.rate_best(graph, page_start_node, start_node,
                                                     context=context,
                                                     lm_weight=1.0, # no influence of input confidence
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
                ppll = pow(2.0, ent * strlen/pathlen) # textequiv level (including spaces/newlines)
                LOG.info("avg: %.3f, char ppl: %.3f, %s ppl: %.3f", avg, ppl, level, ppll) # character need not always equal glyph!
            else:
                text = [(graph.edges[in_, out]['element'], graph.edges[in_, out]['alternatives']) for in_, out in nx.bfs_edges(graph, 0)] # graph's path
                textstring = u''.join(textequivs[0].Unicode for element, textequivs in text) # same length as text
                LOG.info("Rating %d elements with a total of %d characters", len(text), len(textstring))
                confidences = self.rater.rate(textstring, context) # much faster
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
                                    word_unicode = u''.join(glyph.get_TextEquiv()[0].Unicode if glyph.get_TextEquiv() else u'' for glyph in glyphs)
                                    word.set_TextEquiv([TextEquivType(Unicode=word_unicode)]) # remove old
                            line_unicode = u' '.join(word.get_TextEquiv()[0].Unicode if word.get_TextEquiv() else u'' for word in words)
                            line.set_TextEquiv([TextEquivType(Unicode=line_unicode)]) # remove old
                    region_unicode = u'\n'.join(line.get_TextEquiv()[0].Unicode if line.get_TextEquiv() else u'' for line in lines)
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

def repair_tokenisation(tokenisation_problems, identifier, running_hypothesis, start_node, next_token, first):
    # tokenisation inconsistency does not apply if:
    # - element id not contained in detected problem set
    # - there is no TextEquiv to compare with at the next token
    # - the element is first of its kind (i.e. must not start with white space anyway)
    if identifier in tokenisation_problems and next_token and not first: 
        # invariant: text should contain a representation that concatenates into actual tokenisation
        tokenisation = tokenisation_problems[identifier].actual
        concatenation = u''.join(map(lambda x: running_hypothesis.edges[x[0], x[1]]['alternatives'][0].Unicode, nx.bfs_edges(running_hypothesis, start_node)))
        # ideally, both overlap (concatenation~tokenisation)
        i = 0
        for i in range(min(len(tokenisation), len(concatenation)), -1, -1):
            if concatenation[-i:] == tokenisation[:i]:
                break
        if i > 0 and tokenisation[i:].startswith(next_token): # without white space?
            LOG.warning('Repairing tokenisation between "%s" and "%s"', concatenation[-i:], next_token)
            return True # repair by skipping space/newline here
    return False
