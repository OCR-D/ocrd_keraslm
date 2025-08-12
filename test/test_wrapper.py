import os
import logging

from ocrd import run_processor
from ocrd_modelfactory import page_from_file
from ocrd_tesserocr.recognize import TesserocrRecognize
from ocrd_keraslm.wrapper import KerasRate

MODEL = os.getenv('OCRD_KERASLM_MODEL', 'model_dta_full.h5')

def test_scoring(processor_kwargs, caplog):
    """rate text alternative 1 on the word level"""
    caplog.set_level(logging.DEBUG)
    ws = processor_kwargs['workspace']
    run_processor(
        KerasRate,
        input_file_grp='OCR-D-GT-PAGE', # has wrong tokenisation but that's ok now
        output_file_grp='OCR-D-LM-WORD',
        parameter={'textequiv_level': 'word',
                   'alternative_decoding': False,
                   'model_file': MODEL},
        **processor_kwargs
    )
    ws.save_mets()
    n_pages = 0
    n_words = 0
    for file in ws.mets.find_files(fileGrp='OCR-D-LM-WORD'):
        pcgts = page_from_file(ws.download_file(file))
        metadata = pcgts.get_Metadata()
        assert metadata
        metadataitems = metadata.get_MetadataItem()
        assert metadataitems
        assert any([i for i in metadataitems
                    if i.get_value() == 'ocrd-keraslm-rate'])
        for line in pcgts.Page.get_AllTextLines():
            for word in line.get_Word():
                assert len(word.get_TextEquiv()) == 1 # only 1-best results
                #for glyph in word.get_Glyph():
                #    assert len(glyph.get_TextEquiv()) == 1 # only 1-best results
                n_words += 1
        n_pages += 1
    assert n_pages > 1
    assert n_words > 100
    messages = [logrec.message for logrec in caplog.records]
    assert "are off by" not in messages
    assert len([msg for msg in messages if msg.startswith("Scoring text in page")]) == n_pages
    ppls = [float(dict([v.strip() for v in p.split(':')] for p in msg.split(','))['char ppl'])
            for msg in messages if msg.startswith("avg:")]
    refq = 6.0 if MODEL.endswith('full.h5') else 11.5
    assert all(ppl < refq for ppl in ppls), ppls

def test_decoding(processor_kwargs, caplog):
    """rate and viterbi-decode all text alternatives on the glyph level"""
    caplog.set_level(logging.DEBUG)
    ws = processor_kwargs['workspace']
    run_processor(
        TesserocrRecognize, # we need this to get alternatives to decode
        input_file_grp='OCR-D-GT-PAGE', # has wrong tokenisation but that's ok now
        output_file_grp='OCR-D-OCR-TESS-GLYPH',
        parameter={'textequiv_level': 'glyph',
                   'overwrite_segments': True,
                   'model': 'deu-frak'}, # old model for alternatives
        **processor_kwargs
    )
    ws.save_mets()
    run_processor(
        KerasRate,
        input_file_grp='OCR-D-OCR-TESS-GLYPH',
        output_file_grp='OCR-D-LM-GLYPH',
        parameter={'textequiv_level': 'glyph',
                   'alternative_decoding': True,
                   'beam_width': 10, # not too slow
                   'model_file': MODEL},
        **processor_kwargs
    )
    ws.save_mets()
    n_pages = 0
    n_chars = 0
    for file in ws.mets.find_files(fileGrp='OCR-D-LM-GLYPH'):
        pcgts = page_from_file(ws.download_file(file))
        metadata = pcgts.get_Metadata()
        assert metadata
        metadataitems = metadata.get_MetadataItem()
        assert metadataitems
        assert any([i for i in metadataitems
                    if i.get_value() == 'ocrd-keraslm-rate'])
        for line in pcgts.Page.get_AllTextLines():
            for word in line.get_Word():
                for glyph in word.get_Glyph():
                    assert len(glyph.get_TextEquiv()) == 1 # only 1-best results
                    n_chars += 1
        n_pages += 1
    assert n_pages > 1
    assert n_chars > 1000
    messages = [logrec.message for logrec in caplog.records]
    assert "AssertionError" not in messages
    assert len([msg for msg in messages if msg.startswith("Scoring text in page")]) == n_pages
    #assert len([msg for msg in messages if msg.endswith("existing paths") and not msg.endswith("0 existing paths")]) > 0
    ppls = [float(dict([v.strip() for v in p.split(':')] for p in msg.split(','))['char ppl'])
            for msg in messages if msg.startswith("avg:")]
    refq = 3.5 if MODEL.endswith('full.h5') else 5.0
    assert all(ppl < refq for ppl in ppls)
