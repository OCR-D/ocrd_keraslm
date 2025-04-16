from multiprocessing import Process
import logging
import os
import pytest
from time import sleep

from ocrd import Resolver, Workspace, OcrdMetsServer
from ocrd_utils import (
    MIMETYPE_PAGE,
    pushd_popd,
    disableLogging,
    initLogging,
    setOverrideLogLevel,
    config
)

DEFAULT_WS = "kant_aufklaerung_1784"
WORKSPACES = {
    DEFAULT_WS: os.path.join(os.path.dirname(__file__), 'assets', DEFAULT_WS, 'mets.xml'),
}

@pytest.fixture(params=WORKSPACES.keys())
def workspace(tmpdir, request, pytestconfig):
    initLogging()
    logging.getLogger('ocrd.processor').setLevel(logging.DEBUG)
    #if pytestconfig.getoption('verbose') > 0:
    #    setOverrideLogLevel('DEBUG')
    config.OCRD_MISSING_OUTPUT = "ABORT"
    with pushd_popd(tmpdir):
        directory = str(tmpdir)
        resolver = Resolver()
        url = WORKSPACES[request.param]
        workspace = resolver.workspace_from_url(url, dst_dir=directory, download=True)
        workspace.name = request.param # for debugging
        yield workspace
    config.reset_defaults()
    disableLogging()

CONFIGS = ['', 'pageparallel+metscache']

@pytest.fixture(params=CONFIGS)
def processor_kwargs(request, workspace):
    config.OCRD_DOWNLOAD_INPUT = False # only 4 pre-downloaded pages
    config.OCRD_MISSING_OUTPUT = "ABORT"
    if 'metscache' in request.param:
        config.OCRD_METS_CACHING = True
        #print("enabled METS caching")
    if 'pageparallel' in request.param:
        config.OCRD_MAX_PARALLEL_PAGES = 4
        #print("enabled page-parallel processing")
        def _start_mets_server(*args, **kwargs):
            #print("running with METS server")
            server = OcrdMetsServer(*args, **kwargs)
            server.startup()
        process = Process(target=_start_mets_server,
                          kwargs={'workspace': workspace, 'url': 'mets.sock'})
        process.start()
        sleep(1)
        # instantiate client-side workspace
        asset = workspace.name
        workspace = Workspace(workspace.resolver, workspace.directory,
                              mets_server_url='mets.sock',
                              mets_basename=os.path.basename(workspace.mets_target))
        workspace.name = asset
        yield {'workspace': workspace, 'mets_server_url': 'mets.sock'}
        process.terminate()
    else:
        yield {'workspace': workspace}
    config.reset_defaults()
