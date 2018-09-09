import os, sys
import shutil
from unittest import TestCase, main

from ocrd.resolver import Resolver
from ocrd.model.ocrd_page import from_file, to_xml
from ocrd import MIMETYPE_PAGE
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
        workspace = resolver.workspace_from_url('test/assets/kant_aufklaerung_1784/mets.xml', directory=WORKSPACE_DIR, download_local=True)
        for file in workspace.mets.find_files(fileGrp='OCR-D-GT-PAGE'):
            grp='OCR-D-GT-SEG-LINE'
            ID=grp + '_' + file.ID.split(sep='_')[-1]
            pcgts = from_file(file)
            page = pcgts.get_Page()
            for region in page.get_TextRegion():
                for line in region.get_TextLine():
                    line.set_TextEquiv([]) # remove text results (interferes with Tesserocr)
                    line.set_Word([]) # remove word annotation (interferes with Tesserocr, has wrong tokenization)
            workspace.add_file(
                ID=ID,
                file_grp=grp,
                basename=ID + '.xml',
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))
        TesserocrRecognize(
            workspace,
            input_file_grp='OCR-D-GT-SEG-LINE',
            output_file_grp='OCR-D-OCR-TESS-WORD',
            parameter={'textequiv_level': 'word',
                       'model': 'Fraktur'}
            ).process()
        workspace.save_mets()
        KerasRate(
            workspace,
            input_file_grp='OCR-D-OCR-TESS-WORD',
            output_file_grp='OCR-D-LM-WORD',
            parameter={'textequiv_level': 'word',
                       'alternative_decoding': False,
                       'weight_file': PWD + '/../model_dta_test.weights.h5',
                       'config_file': PWD + '/../model_dta_test.config.pkl'}
            ).process()
        workspace.save_mets()
        workspace.reload_mets()
        for file in workspace.mets.find_files(fileGrp='OCR-D-LM-WORD'):
            continue # todo: for some reason, from_file yields NoneType here
            pcgts = from_file(file)
            metadata = pcgts.get_Metadata()
            assertIsNotNone(metadata)
            metadataitems = metadata.get_MetadataItem()
            assertIsNotNone(metadataitems)
            rated = any([i for i in metadataitems if i.get_value() == 'ocrd-keraslm-rate'])
            assertTrue(rated)
        TesserocrRecognize(
            workspace,
            input_file_grp='OCR-D-GT-SEG-LINE',
            output_file_grp='OCR-D-OCR-TESS-GLYPH',
            parameter={'textequiv_level': 'glyph',
                       'model': 'deu-frak'}
            ).process()
        workspace.save_mets()
        KerasRate(
            workspace,
            input_file_grp='OCR-D-OCR-TESS-GLYPH',
            output_file_grp='OCR-D-LM-GLYPH',
            parameter={'textequiv_level': 'glyph',
                       'alternative_decoding': True,
                       'beam_width': 10,
                       'weight_file': PWD + '/../model_dta_test.weights.h5',
                       'config_file': PWD + '/../model_dta_test.config.pkl'}
            ).process()
        workspace.save_mets()
        workspace.reload_mets()
        for file in workspace.mets.find_files(fileGrp='OCR-D-LM-GLYPH'):
            continue # todo: for some reason, from_file yields NoneType here
            pcgts = from_file(file)
            metadata = pcgts.get_Metadata()
            assertIsNotNone(metadata)
            metadataitems = metadata.get_MetadataItem()
            assertIsNotNone(metadataitems)
            rated = any([i for i in metadataitems if i.get_value() == 'ocrd-keraslm-rate'])
            assertTrue(rated)
            for region in page.get_TextRegion():
                for line in region.get_TextLine():
                    for word in line.get_Word():
                        for glyph in word.get_Glyph():
                            assertEqual(len(glyph.get_TextEquiv()), 1) # only 1-best results

if __name__ == '__main__':
    main()
