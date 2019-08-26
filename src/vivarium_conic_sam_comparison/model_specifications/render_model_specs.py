import sys
from pathlib import Path
from jinja2 import Template
from re import split
import click

# jinja2 template for generating model specifications
TEMPLATE='vivarium_sam_comparison.in'

# name/location for the generated artifacts
LOCATIONS=['India', 'Bangladesh', 'Pakistan', 'Malawi', 'Tanzania', 'Mali']
HELP_STR_LOC='Specify a list of locations/names for which to generate model specifications. The default list is \n' \
                            '{}'.format("\n".join(LOCATIONS))

def coerce_loc_list(locs):
    # if the user specifies locations on the command line, the result is a string. It needs to be
    #  a list. Allow commas or spaces to delimit the list elements
    if isinstance(locs, str):
        locs = [i for i in split('[ ,]',  locs) if len(i)]
    return locs

@click.command()
@click.option('--template', default=TEMPLATE, help=f'Specify the model specification template file. The default is "{TEMPLATE}"')
@click.option('--loc_list', default=LOCATIONS, help=HELP_STR_LOC)
def generate_model_specs(template, loc_list):
    with open(template, 'r') as infile:
        temp = Template(infile.read())
        
        loc_list = coerce_loc_list(loc_list)
        for loc in loc_list:
            with open(f'{loc}.yaml', 'w+') as outfile:
                outfile.write(temp.render(
                    location=loc
                ))

if __name__ == '__main__':
    generate_model_specs()