from vivarium_public_health.metrics.mortality import MortalityObserver


class WHZMortalityObserver(MortalityObserver):

    MortalityObserver.configuration_defaults.update(
            {'metrics': {
                'mortality': {
                    'by_whz': False
            }}})


    def __init__(self):
        super().__init__()
        self.readable_cats = {
            'cat1': 'lt_-3',
            'cat2': '-3_to_-2',
            'cat3': '-2_to_-1',
            'cat4': 'unexposed'
        }
    
    @property
    def name(self):
        return 'whz_mortality_observer'

    def setup(self, builder):
        super().setup(builder)
        self.whz_exposure = builder.value.get_value('child_wasting.exposure')

    def metrics(self, index, metrics):
        if not self.config.by_whz:
            return super().metrics(index, metrics)
       
        pop = self.population_view.get(index)
        pop.loc[pop.exit_time.isnull(), 'exit_time'] = self.clock()
        
        the_living = pop[(pop.alive == 'alive') & pop.tracked]
        the_dead = pop[pop.alive == 'dead']
        metrics['years_of_life_lost'] = self.life_expectancy(the_dead.index).sum()
        metrics['total_population_living'] = len(the_living)

        whz_exposure = self.whz_exposure(pop.index)
        for cat in whz_exposure.unique():
            pop_for_cat = pop.loc[whz_exposure == cat]

            person_time = get_person_time(pop_for_cat, self.config.to_dict(), self.start_time, self.clock(),
                                          self.age_bins)
            deaths = get_deaths(pop_for_cat, self.config.to_dict(), self.start_time, self.clock(),
                                self.age_bins, self.causes)
            ylls = get_years_of_life_lost(pop_for_cat, self.config.to_dict(), self.start_time, self.clock(),
                                          self.age_bins, self.life_expectancy, self.causes)
            
            person_time = {key + f'_in_whz_{self.readable_cats[cat]}': value for key, value in person_time.keys()}
            deaths = {key + f'_in_whz_{self.readable_cats[cat]}': value for key, value in deaths.items()}
            ylls = {key + f'_in_whz_{self.readable_cats[cats]}': value for key, value in ylls.items()}

            metrics.update(person_time)
            metrics.update(deaths)
            metrics.update(ylls)

        return metrics

