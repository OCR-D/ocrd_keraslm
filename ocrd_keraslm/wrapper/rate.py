from __future__ import absolute_import
import os
from typing import List, Optional, Tuple
from math import log, ceil
from collections import defaultdict
from dataclasses import dataclass

from ocrd import Processor, Workspace, OcrdPageResult
from ocrd_validators.page_validator import PageValidator, ConsistencyError
from ocrd_modelfactory import page_from_file
from ocrd_models import OcrdFile
from ocrd_models.ocrd_page import (
    OcrdPage,
    MetadataItemType, LabelsType, LabelType,
    RegionRefType,
    RegionRefIndexedType,
    OrderedGroupType,
    OrderedGroupIndexedType,
    UnorderedGroupType,
    UnorderedGroupIndexedType,
    TextEquivType,
    to_xml
)
from ocrd_models.ocrd_page_generateds import (
    ReadingDirectionSimpleType,
    TextLineOrderSimpleType,
    TextTypeSimpleType
)
from ocrd_utils import (
    MIMETYPE_PAGE,
    config,
    pushd_popd,
    make_file_id,
)

import networkx as nx
from requests import HTTPError

from .. import lib

CHOICE_THRESHOLD_NUM = 4 # maximum number of choices to try per element
CHOICE_THRESHOLD_CONF = 0.1 # maximum score drop from best choice to try per element
#beam_width = 100 # maximum number of best partial paths to consider during search with alternative_decoding
BEAM_CLUSTERING_ENABLE = True # enable pruning partial paths by history clustering
BEAM_CLUSTERING_DIST = 5 # maximum distance between state vectors to form a cluster

# similar to ocrd.validator.page_validator._HIERARCHY:
_HIERARCHY = {
    'Page': 'region',
    'TextRegion': 'line',
    'TextLine': 'word',
    'Word': 'glyph',
    'Glyph': ''
}


@dataclass
class RateState:
    traceback : Tuple[List[lib.rating.Node], lib.rating.Node]
    pcgts : OcrdPage
    file_id : str
    page_id : str
    
class KerasRate(Processor):
    max_workers = 1 # TF/Keras context cannot be shared across forked workers
    # also, our processing shares context from pa

    @property
    def executable(self):
        return 'ocrd-keraslm-rate'

    @property
    def metadata_filename(self) -> str:
        return os.path.join('wrapper', 'ocrd-tool.json')

    def setup(self):
        if not 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # error
        model = self.parameter['model_file']
        model = self.resolve_resource(model)
        self.rater = lib.Rater(logger=self.logger)
        self.rater.load_config(model)
        # overrides necessary before compilation:
        if self.parameter['alternative_decoding']:
            self.rater.stateful = False # no implicit state transfer
            self.rater.incremental = True # but explicit state transfer
        elif self.rater.stateful:
            self.rater.batch_size = 1 # make sure states are consistent with windows after 1 batch
        self.rater.configure()
        self.rater.load_weights(model)
        self.logger.debug("Loaded model_file '%s'", model)
    
    def process_workspace(self, workspace: Workspace) -> None:
        """Rate text with the language model, either for scoring or finding the best path across alternatives.
        
        Open and deserialise PAGE input files, then iterate over the segment hierarchy
        down to the requested `textequiv_level`, making sequences of first TextEquiv objects
        (if `alternative_decoding` is false), or of lists of all TextEquiv objects (otherwise)
        as a linear graph for input to the LM. If the level is above glyph, then insert
        artificial whitespace TextEquiv where implicit tokenisation rules require it.
        
        Next, if `alternative_decoding` is false, then pass the concatenated string of the
        page text to the LM and map the returned sequence of probabilities to the substrings
        in the input TextEquiv. For each TextEquiv, calculate the average character probability
        (LM score) and combine that with the input confidence (OCR score) by applying `lm_weight`.
        Assign the resulting probability as new confidence to the TextEquiv, and ensure no other
        TextEquiv remain on the segment. Finally, calculate the overall average LM probability, 
        and the character and segment-level perplexity, and print it on the logger.
        
        Otherwise (i.e with `alternative_decoding=true`), search for the best paths through
        the input graph of the page (with TextEquiv string alternatives as edges) by applying
        the LM successively via beam search using `beam_width` (keeping a traceback of LM
        state history at each node, passing and updating LM state explicitly). As in the above
        trivial case without `alternative_decoding`, then combine LM scores weighted by `lm_weight`
        with input confidence on the graph's edges. Also, prune worst paths and apply LM state
        history clustering to avoid expanding all possible combinations. Finally, look into the
        current best overall path, traversing back to the last node of the previous page's graph.
        Lock into that node by removing all current paths that do not derive from it, and making
        its history path the final decision for the previous page: Apply that path by removing
        all but the chosen TextEquiv alternatives, assigning the resulting confidences, and
        making the levels above `textequiv_level` consistent with that textual result (via
        concatenation joined by whitespace). Also, calculate the overall average LM probability,
        and the character and segment-level perplexity, and print it on the logger. Moreover,
        at the last page at the end of the document, lock into the current best path analogously.
        
        Produce new output files by serialising the resulting hierarchy for each page.
        """
        if not self.parameter['alternative_decoding']:
            # use conventional API
            return super().process_workspace(workspace)
        self.process_workspace_stateful(workspace)

    def process_workspace_stateful(self, workspace: Workspace) -> None:
        # override, loop over pages without executor, while trying to heed config vars
        level = self.parameter['textequiv_level']
        # ...from process_workspace:
        with pushd_popd(workspace.directory):
            self.workspace = workspace
            self.verify()
            # ...from process_workspace_handle_tasks:
            # aggregate info for logging:
            nr_succeeded = 0
            nr_failed = 0
            nr_errors = defaultdict(int) # count causes
            if config.OCRD_MISSING_OUTPUT == 'SKIP':
                reason = "skipped"
            elif config.OCRD_MISSING_OUTPUT == 'COPY':
                reason = "fallback-copied"
            # ...instead of process_workspace_submit_task(s)
            prev = None # RateState()
            for input_file in self.input_files:
                page_id = input_file.pageId
                self._base_logger.info(f"preparing page {page_id}")
                if self.download:
                    try:
                        input_file = self.workspace.download_file(input_file)
                    except (ValueError, FileNotFoundError, HTTPError) as e:
                        self._base_logger.error(repr(e))
                        self._base_logger.warning(f"failed downloading file {input_file} for page {page_id}")
                # ...instead of process_page_file:
                if input_file.local_filename is None:
                    self._base_logger.debug(f"ignoring missing file for page {page_id}")
                    continue
                self._base_logger.info("processing page %s", page_id)
                self._base_logger.debug(f"parsing file {input_file.ID} for page {page_id}")
                try:
                    pcgts = page_from_file(input_file)
                    assert isinstance(pcgts, OcrdPage)
                except ValueError as err:
                    # not PAGE and not an image to generate PAGE for
                    self._base_logger.error(f"non-PAGE input for page {page_id}: {err}")
                    continue
                output_file_id = make_file_id(input_file, self.output_file_grp)
                if input_file.fileGrp == self.output_file_grp:
                    # input=output fileGrp: re-use ID exactly
                    output_file_id = input_file.ID
                output_file = next(self.workspace.mets.find_files(ID=output_file_id), None)
                if output_file and config.OCRD_EXISTING_OUTPUT != 'OVERWRITE':
                    # short-cut avoiding useless computation:
                    self._base_logger.error(f"A file with ID=={output_file_id} already exists "
                                            f"{output_file} and neither force nor ignore are set")
                    continue
                # ...instead of process_workspace_handle_task(s):
                try:
                    # ...instead of process_page_pcgts:
                    prev = self.process_page_pcgts_stateful(pcgts, prev, output_file_id, page_id)
                    nr_succeeded += 1
                # handle input failures separately
                except FileExistsError as err:
                    if config.OCRD_EXISTING_OUTPUT == 'ABORT':
                        raise err
                    if config.OCRD_EXISTING_OUTPUT == 'SKIP':
                        pass
                    if config.OCRD_EXISTING_OUTPUT == 'OVERWRITE':
                        # too late here, must not happen
                        raise Exception(f"got {err} despite OCRD_EXISTING_OUTPUT==OVERWRITE")
                except KeyboardInterrupt:
                    raise
                # broad coverage of output failures (including TimeoutError)
                except Exception as err:
                    # FIXME: add re-usable/actionable logging
                    if config.OCRD_MISSING_OUTPUT == 'ABORT':
                        self._base_logger.error(f"Failure on page {page_id}: {str(err) or err.__class__.__name__}")
                        raise err
                    self._base_logger.exception(f"Failure on page {page_id}: {str(err) or err.__class__.__name__}")
                    if config.OCRD_MISSING_OUTPUT == 'SKIP':
                        pass
                    elif config.OCRD_MISSING_OUTPUT == 'COPY':
                        self._copy_page_file(input_file)
                    else:
                        desc = config.describe('OCRD_MISSING_OUTPUT', wrap_text=False, indent_text=False)
                        raise ValueError(f"unknown configuration value {config.OCRD_MISSING_OUTPUT} - {desc}")
                    nr_errors[err.__class__.__name__] += 1
                    nr_failed += 1
                    # FIXME: this is just prospective, because len(tasks)==nr_failed+nr_succeeded is not guaranteed
                    if config.OCRD_MAX_MISSING_OUTPUTS > 0 and nr_failed / len(tasks) > config.OCRD_MAX_MISSING_OUTPUTS:
                        # already irredeemably many failures, stop short
                        nr_errors = dict(nr_errors)
                        raise Exception(f"too many failures with {reason} output ({nr_failed} of {nr_failed+nr_succeeded}, {str(nr_errors)})")

            if prev:
                path, entropy, _ = self.rater.next_path(prev.traceback[0], ([], prev.traceback[1]))
                _page_update_from_path(level, path, entropy, logger=self.logger)

                # ensure parent textequivs are up to date:
                page_update_higher_textequiv_levels(level, prev.pcgts)

                # write back result
                prev.pcgts.set_pcGtsId(prev.file_id)
                self.add_metadata(prev.pcgts)
                self.workspace.add_file(
                    ID=prev.file_id,
                    pageId=prev.page_id,
                    file_grp=self.output_file_grp,
                    local_filename=os.path.join(self.output_file_grp, prev.file_id + '.xml'),
                    mimetype=MIMETYPE_PAGE,
                    content=to_xml(prev.pcgts),
                )

            # ...from process_workspace_handle_tasks:
            nr_errors = dict(nr_errors)
            nr_all = nr_succeeded + nr_failed
            if nr_failed > 0:
                if config.OCRD_MAX_MISSING_OUTPUTS > 0 and nr_failed / nr_all > config.OCRD_MAX_MISSING_OUTPUTS:
                    raise Exception(f"too many failures with {reason} output ({nr_failed} of {nr_all}, {str(nr_errors)})")
                self._base_logger.warning("%s %d of %d pages due to %s", reason, nr_failed, nr_all, str(nr_errors))
            self._base_logger.debug("succeeded %d, missed %d of %d pages due to %s", nr_succeeded, nr_failed, nr_all, str(nr_errors))

    def process_page_pcgts_stateful(self, pcgts: OcrdPage, prev: Optional[RateState], file_id: str, page_id: str) -> RateState:
        level = self.parameter['textequiv_level']
        beam_width = self.parameter['beam_width']
        lm_weight = self.parameter['lm_weight']
        self.rater.logger.info("Scoring text in page '%s' at the %s level", pcgts.get_pcGtsId(), level)

        context = mets_get_context(self.workspace.mets)
        # create a graph for the linear sequence of elements at the given level:
        graph, start_node, end_node = page_get_linear_graph_at(level, pcgts, logger=self.logger)

        # apply language model to (TextEquiv path in) graph,
        # remove non-path TextEquivs, modify confidences:

        self.rater.logger.info("Rating %d elements including its alternatives", end_node - start_node)
        path, entropy, traceback = self.rater.rate_best(
            graph, start_node, end_node,
            start_traceback=prev and prev.traceback,
            context=context,
            lm_weight=lm_weight,
            beam_width=beam_width,
            beam_clustering_dist=BEAM_CLUSTERING_DIST if BEAM_CLUSTERING_ENABLE else 0)

        if prev:
            _page_update_from_path(level, path, entropy, logger=self.logger)

            # ensure parent textequivs are up to date:
            page_update_higher_textequiv_levels(level, prev.pcgts)

            # write back result
            prev.pcgts.set_pcGtsId(prev.file_id)
            self.add_metadata(prev.pcgts)
            self.workspace.add_file(
                ID=prev.file_id,
                pageId=prev.page_id,
                file_grp=self.output_file_grp,
                local_filename=os.path.join(self.output_file_grp, prev.file_id + '.xml'),
                mimetype=MIMETYPE_PAGE,
                content=to_xml(prev.pcgts),
            )

        # next
        prev = RateState(traceback=traceback, pcgts=pcgts, file_id=file_id, page_id=page_id)
        return prev

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        pcgts = input_pcgts[0]

        level = self.parameter['textequiv_level']
        beam_width = self.parameter['beam_width']
        lm_weight = self.parameter['lm_weight']
        self.rater.logger.info("Scoring text in page '%s' at the %s level", pcgts.get_pcGtsId(), level)

        context = mets_get_context(self.workspace.mets)
        # create a graph for the linear sequence of elements at the given level:
        graph, start_node, end_node = page_get_linear_graph_at(level, pcgts, logger=self.logger)

        text = [(edge['element'], edge['alternatives']) for edge in _get_edges(graph, 0)] # graph's path
        textstring = ''.join(textequivs[0].Unicode for element, textequivs in text) # same length as text
        self.logger.info("Rating %d elements with a total of %d characters", len(text), len(textstring))
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
            self.logger.critical("Input text length and output scores length are off by %d characters", i-len(confidences))
        avg = sum(confidences)/len(confidences)
        ent = sum([-log(max(p, 1e-99), 2) for p in confidences])/len(confidences)
        ppl = pow(2.0, ent) # character level
        ppll = pow(2.0, ent * len(confidences)/len(text)) # textequiv level (including spaces/newlines)
        self.logger.info("avg: %.3f, char ppl: %.3f, %s ppl: %.3f", avg, ppl, level, ppll) # character need not always equal glyph!
        return OcrdPageResult(pcgts)

def mets_get_context(mets):
    """context preprocessing"""
    # todo: as soon as we have true MODS meta-data in METS (dmdSec/mdWrap/xmlData/mods),
    #       get global context variables from there (e.g. originInfo/dateIssued/@text for year)
    ident = mets.unique_identifier # at least try to get purl
    context = [0]
    if ident:
        name = ident.split('/')[-1]
        year = name.split('_')[-1]
        if year.isnumeric():
            year = ceil(int(year)/10)
            context = [year]
            # todo: author etc
    return context

def page_get_linear_graph_at(level, pcgts, logger=None):
    if logger is None:
        logger = getLogger('ocrd.processor.KerasRate')
    problems = _page_get_tokenisation_problems(level, pcgts, logger=logger)
    
    graph = nx.DiGraph(level=level) # initialise directed unigraph
    graph.add_node(0)
    start_node = 0
    regions = pcgts.get_Page().get_TextRegion()
    if not regions:
        logger.warning("Page contains no text regions")
    page_start_node = start_node
    first_region = True
    for region in regions:
        if level == 'region':
            logger.debug("Getting text in region '%s'", region.id)
            textequivs = region.get_TextEquiv()
            if not first_region:
                start_node = _add_space(graph, start_node, '\n',
                                        page_start_node, problems.get(pcgts.get_pcGtsId()),
                                        textequivs,
                                        logger=logger)
            if textequivs:
                start_node = _add_element(graph, start_node, region, textequivs)
            else:
                logger.warning("Region '%s' contains no text results", region.id)
            first_region = False
            continue
        lines = region.get_TextLine()
        if not lines:
            logger.warning("Region '%s' contains no text lines", region.id)
        region_start_node = start_node
        first_line = True
        for line in lines:
            if level == 'line':
                logger.debug("Getting text in line '%s'", line.id)
                textequivs = line.get_TextEquiv()
                if not first_line or not first_region:
                    start_node = _add_space(graph, start_node, '\n',
                                            region_start_node, not first_line and problems.get(region.id),
                                            textequivs,
                                            logger=logger)
                if textequivs:
                    start_node = _add_element(graph, start_node, line, textequivs)
                else:
                    logger.warning("Line '%s' contains no text results", line.id)
                first_line = False
                continue
            words = line.get_Word()
            if not words:
                logger.warning("Line '%s' contains no words", line.id)
            line_start_node = start_node
            first_word = True
            for word in words:
                textequivs = word.get_TextEquiv()
                if not first_word or not first_line or not first_region:
                    start_node = _add_space(graph, start_node, '\n' if first_word else ' ',
                                            line_start_node, not first_word and problems.get(line.id),
                                            textequivs,
                                            logger=logger)
                if level == 'word':
                    logger.debug("Getting text in word '%s'", word.id)
                    if textequivs:
                        start_node = _add_element(graph, start_node, word, textequivs)
                    else:
                        logger.warning("Word '%s' contains no text results", word.id)
                    first_word = False
                    continue
                glyphs = word.get_Glyph()
                if not glyphs:
                    logger.warning("Word '%s' contains no glyphs", word.id)
                for glyph in glyphs:
                    logger.debug("Getting text in glyph '%s'", glyph.id)
                    textequivs = glyph.get_TextEquiv()
                    if textequivs:
                        start_node = _add_element(graph, start_node, glyph, textequivs)
                    else:
                        logger.warning("Glyph '%s' contains no text results", glyph.id)
                first_word = False
            first_line = False
        first_region = False
    return graph, page_start_node, start_node

def _page_update_from_path(level, path, entropy, logger=None):
    if logger is None:
        logger = getLogger('ocrd.processor.KerasRate')
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
    logger.info("avg: %.3f, char ppl: %.3f, %s ppl: %.3f", avg, ppl, level, ppll) # character need not always equal glyph!

def page_element_unicode0(element):
    """Get Unicode string of the first text result."""
    if element.get_TextEquiv():
        return element.get_TextEquiv()[0].Unicode or ''
    else:
        return ''

def page_element_conf0(element):
    """Get confidence (as float value) of the first text result."""
    if element.get_TextEquiv():
        # generateDS does not convert simpleType for attributes (yet?)
        return float(element.get_TextEquiv()[0].conf or "1.0")
    return 1.0

def page_get_reading_order(ro, rogroup):
    """Add all elements from the given reading order group to the given dictionary.
    
    Given a dict ``ro`` from layout element IDs to ReadingOrder element objects,
    and an object ``rogroup`` with additional ReadingOrder element objects,
    add all references to the dict, traversing the group recursively.
    """
    regionrefs = list()
    if isinstance(rogroup, (OrderedGroupType, OrderedGroupIndexedType)):
        regionrefs = (rogroup.get_RegionRefIndexed() +
                      rogroup.get_OrderedGroupIndexed() +
                      rogroup.get_UnorderedGroupIndexed())
    if isinstance(rogroup, (UnorderedGroupType, UnorderedGroupIndexedType)):
        regionrefs = (rogroup.get_RegionRef() +
                      rogroup.get_OrderedGroup() +
                      rogroup.get_UnorderedGroup())
    for elem in regionrefs:
        ro[elem.get_regionRef()] = elem
        if not isinstance(elem, (RegionRefType, RegionRefIndexedType)):
            page_get_reading_order(ro, elem)

def page_update_higher_textequiv_levels(level, pcgts, overwrite=True):
    """Update the TextEquivs of all PAGE-XML hierarchy levels above ``level`` for consistency.
    
    Starting with the lowest hierarchy level chosen for processing,
    join all first TextEquiv.Unicode (by the rules governing the respective level)
    into TextEquiv.Unicode of the next higher level, replacing them.
    If ``overwrite`` is false and the higher level already has text, keep it.
    
    When two successive elements appear in a ``Relation`` of type ``join``,
    then join them directly (without their respective white space).
    
    Likewise, average all first TextEquiv.conf into TextEquiv.conf of the next higher level.
    
    In the process, traverse the words and lines in their respective ``readingDirection``,
    the (text) regions which contain lines in their respective ``textLineOrder``, and
    the (text) regions which contain text regions in their ``ReadingOrder``
    (if they appear there as an ``OrderedGroup``).
    Where no direction/order can be found, use XML ordering.
    
    Follow regions recursively, but make sure to traverse them in a depth-first strategy.
    """
    page = pcgts.get_Page()
    relations = page.get_Relations() # get RelationsType
    if relations:
        relations = relations.get_Relation() # get list of RelationType
    else:
        relations = []
    joins = list() # 
    for relation in relations:
        if relation.get_type() == 'join': # ignore 'link' type here
            joins.append((relation.get_SourceRegionRef().get_regionRef(),
                          relation.get_TargetRegionRef().get_regionRef()))
    reading_order = dict()
    ro = page.get_ReadingOrder()
    if ro:
        page_get_reading_order(reading_order, ro.get_OrderedGroup() or ro.get_UnorderedGroup())
    if level != 'region':
        for region in page.get_AllRegions(classes=['Text']):
            # order is important here, because regions can be recursive,
            # and we want to concatenate by depth first;
            # typical recursion structures would be:
            #  - TextRegion/@type=paragraph inside TextRegion
            #  - TextRegion/@type=drop-capital followed by TextRegion/@type=paragraph inside TextRegion
            #  - any region (including TableRegion or TextRegion) inside a TextRegion/@type=footnote
            #  - TextRegion inside TableRegion
            subregions = region.get_TextRegion()
            if subregions: # already visited in earlier iterations
                # do we have a reading order for these?
                # TODO: what if at least some of the subregions are in reading_order?
                if (all(subregion.id in reading_order for subregion in subregions) and
                    isinstance(reading_order[subregions[0].id], # all have .index?
                               (OrderedGroupType, OrderedGroupIndexedType))):
                    subregions = sorted(subregions, key=lambda subregion:
                                        reading_order[subregion.id].index)
                region_unicode = page_element_unicode0(subregions[0])
                for subregion, next_subregion in zip(subregions, subregions[1:]):
                    if (subregion.id, next_subregion.id) not in joins:
                        region_unicode += '\n' # or '\f'?
                    region_unicode += page_element_unicode0(next_subregion)
                region_conf = sum(page_element_conf0(subregion) for subregion in subregions)
                region_conf /= len(subregions)
            else: # TODO: what if a TextRegion has both TextLine and TextRegion children?
                lines = region.get_TextLine()
                if ((region.get_textLineOrder() or
                     page.get_textLineOrder()) ==
                    TextLineOrderSimpleType.BOTTOMTOTOP):
                    lines = list(reversed(lines))
                if level != 'line':
                    for line in lines:
                        words = line.get_Word()
                        if ((line.get_readingDirection() or
                             region.get_readingDirection() or
                             page.get_readingDirection()) ==
                            ReadingDirectionSimpleType.RIGHTTOLEFT):
                            words = list(reversed(words))
                        if level != 'word':
                            for word in words:
                                glyphs = word.get_Glyph()
                                if ((word.get_readingDirection() or
                                     line.get_readingDirection() or
                                     region.get_readingDirection() or
                                     page.get_readingDirection()) ==
                                    ReadingDirectionSimpleType.RIGHTTOLEFT):
                                    glyphs = list(reversed(glyphs))
                                word_unicode = ''.join(page_element_unicode0(glyph) for glyph in glyphs)
                                word_conf = sum(page_element_conf0(glyph) for glyph in glyphs)
                                if glyphs:
                                    word_conf /= len(glyphs)
                                if not word.get_TextEquiv() or overwrite:
                                    word.set_TextEquiv( # replace old, if any
                                        [TextEquivType(Unicode=word_unicode, conf=word_conf)])
                        line_unicode = ' '.join(page_element_unicode0(word) for word in words)
                        line_conf = sum(page_element_conf0(word) for word in words)
                        if words:
                            line_conf /= len(words)
                        if not line.get_TextEquiv() or overwrite:
                            line.set_TextEquiv( # replace old, if any
                                [TextEquivType(Unicode=line_unicode, conf=line_conf)])
                region_unicode = ''
                region_conf = 0
                if lines:
                    region_unicode = page_element_unicode0(lines[0])
                    for line, next_line in zip(lines, lines[1:]):
                        words = line.get_Word()
                        next_words = next_line.get_Word()
                        if not (words and next_words and (words[-1].id, next_words[0].id) in joins):
                            region_unicode += '\n'
                        region_unicode += page_element_unicode0(next_line)
                    region_conf = sum(page_element_conf0(line) for line in lines)
                    region_conf /= len(lines)
            if not region.get_TextEquiv() or overwrite:
                region.set_TextEquiv( # replace old, if any
                    [TextEquivType(Unicode=region_unicode, conf=region_conf)])

def _page_get_tokenisation_problems(level, pcgts, logger=None):
    if logger is None:
        logger = getLogger('ocrd.processor.KerasRate')
    # white space IFF between words, newline IFF between lines/regions: required for LM input
    # as a minor mitigation, try to guess consistency a text annotation on multiple levels
    # (i.e. infer wrong tokenisation when mother node has TextEquiv deviating from
    #  concatenated child node TextEquivs only w.r.t. white-space):
    report = PageValidator.validate(ocrd_page=pcgts, page_textequiv_consistency='strict')
    problems = {}
    if not report.is_valid:
        logger.warning("Page validation failed: %s", report.to_xml())
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

def _add_space(graph, start_node, space, last_start_node, problem, textequivs, logger=None):
    """add a pseudo-element edge for the white-space string `space` to `graph`,
    between `start_node` and new node `start_node`+1, except if there is a
    tokenisation `problem` involving the first textequiv in the graph's current tip"""
    if logger is None:
        logger = getLogger('ocrd.processor.KerasRate')
    # tokenisation inconsistency does not apply if:
    # - element id not contained in detected problem set
    # - there is no TextEquiv to compare with at the next token
    # - the element is first of its kind (i.e. must not start with white space anyway)
    if (textequivs and textequivs[0].Unicode and problem and
        _repair_tokenisation(problem.actual,
                             ''.join(map(lambda x: x['alternatives'][0].Unicode, _get_edges(graph, last_start_node))),
                             textequivs[0].Unicode,
                             logger=logger)):
        pass # skip all rules for concatenation joins
    else: # joining space required for LM input here?
        start_node = _add_element(graph, start_node, None, [TextEquivType(Unicode=space, conf=1.0)])
        # LM output will not appear in annotation
        # (so conf cannot be combined to accurate perplexity from output)
    return start_node

def _repair_tokenisation(tokenisation, concatenation, next_token, logger=None):
    if logger is None:
        logger = getLogger('ocrd.processor.KerasRate')
    # invariant: text should contain a representation that concatenates into actual tokenisation
    # ideally, both overlap (concatenation~tokenisation)
    i = 0
    for i in range(min(len(tokenisation), len(concatenation)), -1, -1):
        if concatenation[-i:] == tokenisation[:i]:
            break
    if i > 0 and tokenisation[i:].startswith(next_token): # without white space?
        logger.warning('Repairing tokenisation between "%s" and "%s"', concatenation[-i:], next_token)
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
