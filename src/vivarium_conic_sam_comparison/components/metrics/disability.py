from vivarium_public_health.metrics.disability import Disability
from vivarium_public_health.metrics.utilities import get_years_lived_with_disability


class WHZDisabilityObserver(Disability):

    Disability.configuration_defaults.update(
        {'metrics': {
            'disability': {
                'by_whz': False
        }}})

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'whz_disability_observer'

    def setup(self, builder):
        super().setup(builder)
        self.whz_exposure = builder.value.get_value('child_stunting.exposure')

    def on_time_step_prepare(self, event):
        # Almost the same process, just additionally subset by WHZ.
        whz_exposure = self.whz_exposure(event.index)
        pop = self.population_view.get(event.index, query='tracked == True and alive == "alive"')
        for cat in whz_exposure.unique():
            cat_idx = (whz_exposure == cat).index
            pop_for_cat = pop.loc[cat_idx]

            ylds_this_step = get_years_lived_with_disability(pop_for_cat, self.config.to_dict(),
                                                             self.clock().year, self.step_size(),
                                                             self.age_bins, self.disability_weight_pipelines, self.causes)
            ylds_this_step = {key + f'_in_whz_cat_{cat}': value for key, value in ylds_this_step.items()}
            self.years_lived_with_disability.update(ylds_this_step)

            pop.loc[cat_idx, 'years_lived_with_disability'] += self.disability_weight(pop_for_cat.index)
            self.population_view.update(pop)
