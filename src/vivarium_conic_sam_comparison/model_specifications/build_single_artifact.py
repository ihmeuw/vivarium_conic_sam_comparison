from bdb import BdbQuit
import sys
from pathlib import Path

import click
from loguru import logger

from vivarium_gbd_access.gbd import ARTIFACT_FOLDER
from vivarium_inputs.data_artifact import utilities
from vivarium_inputs.data_artifact.cli import build_artifact

PROJECT_NAME='vivarium_conic_sam_comparison'
OUTPUT_ROOT= ARTIFACT_FOLDER / PROJECT_NAME


@click.command()
@click.argument('model_specification')
@click.option('--pdb', 'debugger', is_flag=True, help='Drop the debugger if an error occurs')
def build_sam_artifact(model_specification, debugger):
    """
    build_artifact is a program for building data artifacts.

    MODEL_SPECIFICATION should be the name of the model specification you wish
    to build an artifact for, e.g., bangladesh.yaml and should be available in
    the model_specifications folder of this repository.

    It requires access to the J drive and /ihme. If you are running this job
    from a qlogin on the cluster, you must specifically request J drive access
    when you qlogin by adding "-l archive=TRUE" to your command.

    Please have at least 50GB of memory on your qlogin."""

    utilities.setup_logging(OUTPUT_ROOT, Path(model_specification).parent)

    try:
        build(model_specification, OUTPUT_ROOT, append)
    except (BdbQuit, KeyboardInterrupt):
        raise
    except Exception as e:
        logger.exception("Uncaught exception: %s", e)
        if debugger:
            import pdb
            import traceback
            traceback.print_exc()
            pdb.post_mortem()
        else:
            raise
    

def build(model_spec, output_root, append):
    logger.info(f'Starting build for "{model_spec}"')
    build_artifact(str(model_spec), output_root, None, append)
    logger.info(f'Completed build for "{model_spec}"')
    

if __name__ == '__main__':
    build_sam_artifact()    