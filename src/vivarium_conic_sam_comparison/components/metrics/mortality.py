from collections import Counter

import pandas as pd
import numpy as np

from vivarium_public_health.metrics.mortality import MortalityObserver
from vivarium_public_health.metrics.utilities import (QueryString, get_output_template, get_group_counts,
                                                      get_deaths, get_years_of_life_lost)

from vivarium_conic_sam_comparison.components.metrics.utilities import convert_whz_to_categorical


class WHZMortalityObserver(MortalityObserver):

    MortalityObserver.configuration_defaults.update(
            {'metrics': {
                'mortality': {
                    'by_whz': False
            }}})


    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'whz_mortality_observer'

    def setup(self, builder):
        super().setup(builder)
        # FIXME: does this need to be done with timedeltas?
        self.step_size = builder.configuration.time.step_size / 365.25
        # NOTE: Abie wants un-intervened on exposure to govern the groupings.
        self.raw_whz_exposure = builder.value.get_value('child_wasting.exposure').source

        builder.value.register_value_modifier('metrics', self.metrics)

        if self.config.by_whz:
            # We want to categorize based on WHZ at death, so we need to track that.
            # We want to track whz 
            self.whz_at_death_view = builder.population.get_view(['alive', 'whz_at_death'])
            builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=['whz_at_death'])
            builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)
            self.person_time = Counter()

    def on_initialize_simulants(self, pop_data):
        pop = self.whz_at_death_view.subview(['alive']).get(pop_data.index)
        pop['whz_at_death'] = np.nan

        # FIXME: Can peole die immediately? Don't think so, but check
        dead = pop.loc[pop['alive'] == 'dead']
        pop.loc[dead.index, 'whz_at_death'] = self.raw_whz_exposure(dead.index) 
        self.whz_at_death_view.update(pop)

    def on_time_step_prepare(self, event):
        # we count person time each time step if we are tracking WHZ
        pop = self.population_view.get(event.index)
        raw_whz_exposure = self.raw_whz_exposure(event.index)
        whz_exposure = convert_whz_to_categorical(raw_whz_exposure)
        for cat in whz_exposure.unique():
            in_cat = pop.loc[whz_exposure == cat]
            base_filter = QueryString('alive == "alive"')
            base_key = get_output_template(**self.config)
            # base_key += f'_in_whz_group_{cat}'
            base_key.substitute(measure='person_time', year=self.clock().year)
            counts = get_group_counts(in_cat, base_filter, base_key, self.config.to_dict(), self.age_bins)
            counts = {str(key) + f'_in_whz_group_{cat}': value * self.step_size for key, value in counts.items()}
            # counts *= self.step_size()
            self.person_time.update(counts)

    def metrics(self, index, metrics):
        if not self.config.by_whz:
            return super().metrics(index, metrics)

        pop = self.population_view.get(index)
        pop.loc[pop.exit_time.isnull(), 'exit_time'] = self.clock()

        the_living = pop[(pop.alive == 'alive') & pop.tracked]
        the_dead = pop[pop.alive == 'dead']
        metrics['years_of_life_lost'] = self.life_expectancy(the_dead.index).sum()
        metrics['total_population_living'] = len(the_living)

        # Ylls and Deaths are 'point' estimates at the time of death.
        # We can count them after-the-fact since we tracked WHZ at death.
        raw_whz_exposure = self.raw_whz_exposure(pop.index)
        whz_exposure = convert_whz_to_categorical(raw_whz_exposure)
        for cat in whz_exposure.unique():
            pop_for_cat = pop.loc[whz_exposure == cat]
            deaths = get_deaths(pop_for_cat, self.config.to_dict(), self.start_time, self.clock(),
                                self.age_bins, self.causes)
            ylls = get_years_of_life_lost(pop_for_cat, self.config.to_dict(), self.start_time, self.clock(),
                                          self.age_bins, self.life_expectancy, self.causes)

            deaths = {key + f'_in_whz_{cat}': value for key, value in deaths.items()}
            ylls = {key + f'_in_whz_{cat}': value for key, value in ylls.items()}

            # metrics.update(person_time)
            metrics.update(deaths)
            metrics.update(ylls)

        # toss in the person time we accrued each step
        metrics.update(self.person_time)
        return metrics

