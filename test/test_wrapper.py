import os, sys
import shutil
from unittest import TestCase, main

from ocrd.resolver import Resolver
from ocrd_models.ocrd_page import to_xml
from ocrd_modelfactory import page_from_file
from ocrd_utils import MIMETYPE_PAGE
from ocrd_tesserocr.recognize import TesserocrRecognize
from ocrd_keraslm.wrapper import KerasRate

WORKSPACE_DIR = '/tmp/pyocrd-test-ocrd_keraslm'
PWD = os.path.dirname(os.path.realpath(__file__))

class TestKerasRate(TestCase):

    def setUp(self):
        if os.path.exists(WORKSPACE_DIR):
            shutil.rmtree(WORKSPACE_DIR)
        os.makedirs(WORKSPACE_DIR)

    def runTest(self):
        resolver = Resolver()
        workspace = resolver.workspace_from_url('test/assets/kant_aufklaerung_1784/data/mets.xml', dst_dir=WORKSPACE_DIR, download=True)
        self.assertIsNotNone(workspace)
        #
        # rate text alternative 1 on the word level:
        #
        KerasRate(
            workspace,
            input_file_grp='OCR-D-GT-PAGE', # has wrong tokenisation but that's ok now
            output_file_grp='OCR-D-LM-WORD',
            parameter={'textequiv_level': 'word',
                       'alternative_decoding': False,
                       'model_file': PWD + '/../model_dta_test.h5'}
            ).process()
        workspace.save_mets()
        for file in workspace.mets.find_files(fileGrp='OCR-D-LM-WORD'):
            pcgts = page_from_file(workspace.download_file(file))
            metadata = pcgts.get_Metadata()
            self.assertIsNotNone(metadata)
            metadataitems = metadata.get_MetadataItem()
            self.assertIsNotNone(metadataitems)
            rated = any([i for i in metadataitems if i.get_value() == 'ocrd-keraslm-rate'])
            self.assertTrue(rated)
        # 
        # rate and viterbi-decode all text alternatives on the glyph level:
        # 
        TesserocrRecognize( # we need this to get alternatives to decode
            workspace,
            input_file_grp='OCR-D-GT-PAGE', # has wrong tokenisation but that's ok now
            output_file_grp='OCR-D-OCR-TESS-GLYPH',
            parameter={'textequiv_level': 'glyph',
                       'overwrite_words': True,
                       'model': 'deu-frak'} # old model for alternatives
            ).process()
        workspace.save_mets()
        KerasRate(
            workspace,
            input_file_grp='OCR-D-OCR-TESS-GLYPH',
            output_file_grp='OCR-D-LM-GLYPH',
            parameter={'textequiv_level': 'glyph',
                       'alternative_decoding': True,
                       'beam_width': 10, # not too slow
                       'model_file': PWD + '/../model_dta_test.h5'}
            ).process()
        workspace.save_mets()
        for file in workspace.mets.find_files(fileGrp='OCR-D-LM-GLYPH'):
            pcgts = page_from_file(workspace.download_file(file))
            metadata = pcgts.get_Metadata()
            self.assertIsNotNone(metadata)
            metadataitems = metadata.get_MetadataItem()
            self.assertIsNotNone(metadataitems)
            rated = any([i for i in metadataitems if i.get_value() == 'ocrd-keraslm-rate'])
            self.assertTrue(rated)
            page = pcgts.get_Page()
            for region in page.get_TextRegion():
                for line in region.get_TextLine():
                    for word in line.get_Word():
                        for glyph in word.get_Glyph():
                            self.assertEqual(len(glyph.get_TextEquiv()), 1) # only 1-best results

if __name__ == '__main__':
    main()
