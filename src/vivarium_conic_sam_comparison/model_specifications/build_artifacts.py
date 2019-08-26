import sys
from bdb import BdbQuit
from pathlib import Path

import click
from loguru import logger

from vivarium_gbd_access.gbd import ARTIFACT_FOLDER
from vivarium_inputs.data_artifact import utilities
from vivarium_cluster_tools.psimulate.utilities import get_drmaa

drmaa = get_drmaa()

PROJECT_NAME='vivarium_conic_sam_comparison'
OUTPUT_ROOT= ARTIFACT_FOLDER / PROJECT_NAME

JOB_MEMORY_NEEDED=50
JOB_TIME_NEEDED='24:00:00'

RUNNER_SCRIPT='{}/{}'.format(Path.cwd(), 'build_single_artifact.py')


def create_and_run_job(model_spec_path):
    #print(f'Running {model_spec_path}')
    with drmaa.Session() as s:
        jt = s.createJobTemplate()
        jt.remoteCommand = sys.executable
        jt.nativeSpecification = '-l m_mem_free={}G,fthread=1,h_rt={} -q all.q -P proj_cost_effect'.format(
            JOB_MEMORY_NEEDED, JOB_TIME_NEEDED)
        jt.args = [RUNNER_SCRIPT, model_spec_path]
        jt.jobName = f'conic_sam_comparison_build_artifact'
        s.runJob(jt)

@click.command()
@click.option('--model_spec', '-m', multiple=True, help='Multiple model spec files can be provided. Each requires the option switch.')
def build_model_spec(model_spec):
    for m in model_spec:
        p = Path(m)
        if p.exists():
            # send a full path to the model spec file
            create_and_run_job(p.resolve())
        else:
            logger.error(f'The file "{p}" does not exist')
        
        
if __name__ == '__main__':
    build_model_spec()