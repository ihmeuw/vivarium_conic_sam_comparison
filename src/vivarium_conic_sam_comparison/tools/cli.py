from pathlib import Path

from jinja2 import Template
import click

from vivarium_gbd_access.gbd import ARTIFACT_FOLDER
from vivarium_cluster_tools.psimulate.utilities import get_drmaa

JOB_MEMORY_NEEDED = 50
JOB_TIME_NEEDED = '24:00:00'

drmaa = get_drmaa()  # safe if not on cluster


def create_and_run_job(model_spec_path: Path, output_root: Path):
    with drmaa.Session() as s:
        jt = s.createJobTemplate()
        jt.remoteCommand = "build_artifact"
        jt.nativeSpecification = '-V -l m_mem_free={}G,fthread=1,h_rt={} -q all.q -P proj_cost_effect'.format(
            JOB_MEMORY_NEEDED, JOB_TIME_NEEDED)
        jt.args = [model_spec_path, '-o', output_root]
        jt.jobName = f'build_artifact_{model_spec_path.name}'
        result = s.runJob(jt)
        print(f"Submitted job for {model_spec_path.name}. Job id: {result}")


@click.command()
@click.option('--model_spec', '-m', multiple=True, type=click.Path(dir_okay=False, exists=True),
              help='Multiple model spec files can be provided. Each requires the option switch.')
@click.option('--project_name', default='vivarium_conic_sam_comparison',
              help='The name of the research project. Used if no output root is provided.')
@click.option('--output-root', '-o', type=click.Path(file_okay=False, exists=True),
              help='A directory root to store artifact results in.')
def pbuild_artifacts(model_spec, project_name, output_root):
    """Build artifacts in parallel from model specifications. Supports multiple
    model specification files with the -m flag.
    """
    output_root = output_root if output_root else ARTIFACT_FOLDER / project_name
    for m in model_spec:
        p = Path(m)
        create_and_run_job(p.resolve(), output_root)


def validate_locations(locations):
    """Locations in model specifications should be capitalized. There are other
    validations should could be added."""
    for location in locations:
        for word in location.split(' '):
            if not word[0].isupper():
                raise ValueError(f"Locations must be upper case. See {location}")


@click.command()
@click.argument('template', type=click.Path(dir_okay=False, exists=True))
@click.argument('locations', nargs=-1)
@click.option('--project_name', default='vivarium_conic_sam_comparison',
              help='The name of the research project.')
def generate_spec_from_template(template, locations, project_name):
    """Generate model specifications based on TEMPLATE for LOCATIONS. The
    locations should be specified as a comma separated list.

    TEMPLATE is a model specification file. It should be a jinja2 template with
    a keyword for location.

    LOCATIONS is a list of locations, space separated. For
    vivarium_conic_sam_comparison, this should be India, Bangladesh, Pakistan,
    Malawi, Tanzania, and Mali.
    """
    template = Path(template)
    with template.open() as infile:
        temp = Template(infile.read())

        # loc_list = coerce_loc_list(loc_list)
        validate_locations(locations)
        for loc in locations:
            # TODO: handle locations with spaces/odd characters
            with open(f'{template.stem}_{loc}.yaml', 'w+') as outfile:
                outfile.write(temp.render(
                    location=loc
                ))
