from __future__ import absolute_import
import os
from math import log, ceil

from ocrd import Processor
from ocrd_validators.page_validator import PageValidator, ConsistencyError
from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    to_xml,
    MetadataItemType, LabelsType, LabelType,
    TextEquivType
)

import networkx as nx

from .config import OCRD_TOOL
from .. import lib

CHOICE_THRESHOLD_NUM = 4 # maximum number of choices to try per element
CHOICE_THRESHOLD_CONF = 0.1 # maximum score drop from best choice to try per element
#beam_width = 100 # maximum number of best partial paths to consider during search with alternative_decoding
BEAM_CLUSTERING_ENABLE = True # enable pruning partial paths by history clustering
BEAM_CLUSTERING_DIST = 5 # maximum distance between state vectors to form a cluster

# similar to ocrd.validator.page_validator._HIERARCHY:
_HIERARCHY = {'Page': 'region', 'TextRegion': 'line', 'TextLine': 'word', 'Word': 'glyph', 'Glyph': ''}

class KerasRate(Processor):
    
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-keraslm-rate']
        kwargs['version'] = OCRD_TOOL['version']
        super(KerasRate, self).__init__(*args, **kwargs)
        if not hasattr(self, 'workspace') or not self.workspace: # no parameter/workspace for --dump-json or --version (no processing)
            return
        
        LOG = getLogger('processor.KerasRate')
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
        """Rates textual annotation of PAGE input files, producing output files with LM scores (and choices).
        
        ... explain incremental page-wise processing here ...
        """
        LOG = getLogger('processor.KerasRate')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        level = self.parameter['textequiv_level']
        beam_width = self.parameter['beam_width']
        lm_weight = self.parameter['lm_weight']

        prev_traceback = None
        prev_pcgts = None
        prev_file = None
        for (n, input_file) in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            LOG.info("Scoring text in page '%s' at the %s level", pcgts.get_pcGtsId(), level)
            
            # annotate processing metadata:
            metadata = pcgts.get_Metadata() # ensured by page_from_file()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=OCRD_TOOL['tools']['ocrd-keraslm-rate']['steps'][0],
                                 value='ocrd-keraslm-rate',
                                 Labels=[LabelsType(externalRef="parameters",
                                                    Label=[LabelType(type_=name,
                                                                     value=self.parameter[name])
                                                           for name in self.parameter.keys()])]))
            
            # context preprocessing:
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
            
            # create a graph for the linear sequence of elements at the given level:
            graph, start_node, end_node = page_get_linear_graph_at(level, pcgts)
            
            # apply language model to (TextEquiv path in) graph,
            # remove non-path TextEquivs, modify confidences:
            if not self.parameter['alternative_decoding']:
                text = [(edge['element'], edge['alternatives']) for edge in _get_edges(graph, 0)] # graph's path
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
                    conf2 = textequiv.conf
                    textequiv.set_conf(conf * lm_weight + conf2 * (1. - lm_weight))
                    i += textequiv_len
                if i != len(confidences):
                    LOG.critical("Input text length and output scores length are off by %d characters", i-len(confidences))
                avg = sum(confidences)/len(confidences)
                ent = sum([-log(max(p, 1e-99), 2) for p in confidences])/len(confidences)
                ppl = pow(2.0, ent) # character level
                ppll = pow(2.0, ent * len(confidences)/len(text)) # textequiv level (including spaces/newlines)
                LOG.info("avg: %.3f, char ppl: %.3f, %s ppl: %.3f", avg, ppl, level, ppll) # character need not always equal glyph!
                
                # ensure parent textequivs are up to date:
                page_update_higher_textequiv_levels(level, pcgts)
            
                # write back result
                file_id = make_file_id(input_file, self.output_file_grp)
                pcgts.set_pcGtsId(file_id)
                self.workspace.add_file(
                    ID=file_id,
                    pageId=input_file.pageId,
                    file_grp=self.output_file_grp,
                    local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                    mimetype=MIMETYPE_PAGE,
                    content=to_xml(pcgts),
                )
            else:
                LOG.info("Rating %d elements including its alternatives", end_node - start_node)
                path, entropy, traceback = self.rater.rate_best(
                    graph, start_node, end_node,
                    start_traceback=prev_traceback,
                    context=context,
                    lm_weight=lm_weight,
                    beam_width=beam_width,
                    beam_clustering_dist=BEAM_CLUSTERING_DIST if BEAM_CLUSTERING_ENABLE else 0)

                if prev_pcgts:
                    _page_update_from_path(level, path, entropy)
                    
                    # ensure parent textequivs are up to date:
                    page_update_higher_textequiv_levels(level, prev_pcgts)

                    # write back result
                    file_id = make_file_id(prev_file, self.output_file_grp)
                    prev_pcgts.set_pcGtsId(file_id)
                    self.workspace.add_file(
                        ID=file_id,
                        pageId=prev_file.pageId,
                        file_grp=self.output_file_grp,
                        local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                        mimetype=MIMETYPE_PAGE,
                        content=to_xml(prev_pcgts),
                    )

                prev_file = input_file
                prev_pcgts = pcgts
                prev_traceback = traceback
        
        if prev_pcgts:
            path, entropy, _ = self.rater.next_path(prev_traceback[0], ([], prev_traceback[1]))
            _page_update_from_path(level, path, entropy)
            
            # ensure parent textequivs are up to date:
            page_update_higher_textequiv_levels(level, prev_pcgts)

            # write back result
            file_id = make_file_id(input_file, self.output_file_grp)
            prev_pcgts.set_pcGtsId(file_id)
            self.workspace.add_file(
                ID=file_id,
                pageId=input_file.pageId,
                file_grp=self.output_file_grp,
                local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                mimetype=MIMETYPE_PAGE,
                content=to_xml(prev_pcgts),
            )

def page_get_linear_graph_at(level, pcgts):
    LOG = getLogger('processor.KerasRate')
    problems = _page_get_tokenisation_problems(level, pcgts)
    
    graph = nx.DiGraph(level=level) # initialise directed unigraph
    graph.add_node(0)
    start_node = 0
    regions = pcgts.get_Page().get_TextRegion()
    if not regions:
        LOG.warning("Page contains no text regions")
    page_start_node = start_node
    first_region = True
    for region in regions:
        if level == 'region':
            LOG.debug("Getting text in region '%s'", region.id)
            textequivs = region.get_TextEquiv()
            if not first_region:
                start_node = _add_space(graph, start_node, '\n',
                                        page_start_node, problems.get(pcgts.get_pcGtsId()),
                                        textequivs)
            if textequivs:
                start_node = _add_element(graph, start_node, region, textequivs)
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
                LOG.debug("Getting text in line '%s'", line.id)
                textequivs = line.get_TextEquiv()
                if not first_line or not first_region:
                    start_node = _add_space(graph, start_node, '\n',
                                            region_start_node, not first_line and problems.get(region.id),
                                            textequivs)
                if textequivs:
                    start_node = _add_element(graph, start_node, line, textequivs)
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
                textequivs = word.get_TextEquiv()
                if not first_word or not first_line or not first_region:
                    start_node = _add_space(graph, start_node, '\n' if first_word else ' ',
                                            line_start_node, not first_word and problems.get(line.id),
                                            textequivs)
                if level == 'word':
                    LOG.debug("Getting text in word '%s'", word.id)
                    if textequivs:
                        start_node = _add_element(graph, start_node, word, textequivs)
                    else:
                        LOG.warning("Word '%s' contains no text results", word.id)
                    first_word = False
                    continue
                glyphs = word.get_Glyph()
                if not glyphs:
                    LOG.warning("Word '%s' contains no glyphs", word.id)
                for glyph in glyphs:
                    LOG.debug("Getting text in glyph '%s'", glyph.id)
                    textequivs = glyph.get_TextEquiv()
                    if textequivs:
                        start_node = _add_element(graph, start_node, glyph, textequivs)
                    else:
                        LOG.warning("Glyph '%s' contains no text results", glyph.id)
                first_word = False
            first_line = False
        first_region = False
    return graph, page_start_node, start_node

def _page_update_from_path(level, path, entropy):
    LOG = getLogger('processor.KerasRate')
    strlen = 0
    for element, textequiv, score in path:
        if element: # not just space
            element.set_TextEquiv([textequiv]) # delete others
            strlen += len(textequiv.Unicode)
            textequiv.set_conf(score)
        else:
            strlen += 1
    ent = entropy/strlen
    avg = pow(2.0, -ent)
    ppl = pow(2.0, ent) # character level
    ppll = pow(2.0, ent * strlen/len(path)) # textequiv level (including spaces/newlines)
    LOG.info("avg: %.3f, char ppl: %.3f, %s ppl: %.3f", avg, ppl, level, ppll) # character need not always equal glyph!

def page_update_higher_textequiv_levels(level, pcgts):
    '''Update the TextEquivs of all PAGE-XML hierarchy levels above `level` for consistency.
    
    Starting with the hierarchy level chosen for processing,
    join all first TextEquiv (by the rules governing the respective level)
    into TextEquiv of the next higher level, replacing them.
    '''
    LOG = getLogger('processor.KerasRate')
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

def _page_get_tokenisation_problems(level, pcgts):
    LOG = getLogger('processor.KerasRate')
    # white space IFF between words, newline IFF between lines/regions: required for LM input
    # as a minor mitigation, try to guess consistency a text annotation on multiple levels
    # (i.e. infer wrong tokenisation when mother node has TextEquiv deviating from
    #  concatenated child node TextEquivs only w.r.t. white-space):
    report = PageValidator.validate(ocrd_page=pcgts, page_textequiv_consistency='strict')
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
    return problems

def _add_element(graph, start_node, element, textequivs):
    graph.add_node(start_node + 1)
    graph.add_edge(start_node, start_node + 1,
                   element=element,
                   alternatives=_filter_choices(textequivs))
    return start_node + 1

def _add_space(graph, start_node, space, last_start_node, problem, textequivs):
    """add a pseudo-element edge for the white-space string `space` to `graph`,
    between `start_node` and new node `start_node`+1, except if there is a
    tokenisation `problem` involving the first textequiv in the graph's current tip"""
    # tokenisation inconsistency does not apply if:
    # - element id not contained in detected problem set
    # - there is no TextEquiv to compare with at the next token
    # - the element is first of its kind (i.e. must not start with white space anyway)
    if (textequivs and textequivs[0].Unicode and problem and
        _repair_tokenisation(problem.actual,
                             u''.join(map(lambda x: x['alternatives'][0].Unicode, _get_edges(graph, last_start_node))),
                             textequivs[0].Unicode)):
        pass # skip all rules for concatenation joins
    else: # joining space required for LM input here?
        start_node = _add_element(graph, start_node, None, [TextEquivType(Unicode=space, conf=1.0)])
        # LM output will not appear in annotation
        # (so conf cannot be combined to accurate perplexity from output)
    return start_node

def _repair_tokenisation(tokenisation, concatenation, next_token):
    LOG = getLogger('processor.KerasRate')
    # invariant: text should contain a representation that concatenates into actual tokenisation
    # ideally, both overlap (concatenation~tokenisation)
    i = 0
    for i in range(min(len(tokenisation), len(concatenation)), -1, -1):
        if concatenation[-i:] == tokenisation[:i]:
            break
    if i > 0 and tokenisation[i:].startswith(next_token): # without white space?
        LOG.warning('Repairing tokenisation between "%s" and "%s"', concatenation[-i:], next_token)
        return True # repair by skipping space/newline here
    return False

def _get_edges(graph, start_node):
    return [graph.edges[in_, out] for in_, out in nx.bfs_edges(graph, start_node)]

def _filter_choices(textequivs):
    '''assuming `textequivs` are already sorted by input confidence (conf attribute), ensure maximum number and maximum relative threshold'''
    if textequivs:
        textequivs = textequivs[:min(CHOICE_THRESHOLD_NUM, len(textequivs))]
        for te in textequivs:
            # generateDS does not convert simpleType for attributes (yet?)
            if te.conf:
                te.set_conf(float(te.conf))
            else:
                te.set_conf(1.0)
        conf0 = textequivs[0].conf
        return [te for te in textequivs
                if conf0 - te.conf < CHOICE_THRESHOLD_CONF]
    else:
        return []

def _get_conf(textequiv, default=1.0):
    '''get float value of conf attribute with default'''
    return float(textequiv.conf or str(default))
