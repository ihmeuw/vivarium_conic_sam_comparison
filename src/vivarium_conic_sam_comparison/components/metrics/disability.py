from vivarium_public_health.metrics.disability import Disability
from vivarium_public_health.metrics.utilities import get_years_lived_with_disability
from vivarium_conic_sam_comparison.components.metrics import convert_whz_to_categorical


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
        self.raw_whz_exposure = builder.value.get_value('child_stunting.exposure').source

    def on_time_step_prepare(self, event):
        # Almost the same process, just additionally subset by WHZ cat before using utilities.
        if not self.config.by_whz:
            super().on_time_step_prepare(event)
            return

        pop = self.population_view.get(event.index, query='tracked == True and alive == "alive"')
        raw_whz_exposure = self.raw_whz_exposure(pop.index)
        whz_exposure = convert_whz_to_categorical(raw_whz_exposure)
        for cat in whz_exposure.unique():
            pop_for_cat = pop.loc[whz_exposure == cat]

            ylds_this_step = get_years_lived_with_disability(pop_for_cat, self.config.to_dict(),
                                                             self.clock().year, self.step_size(),
                                                             self.age_bins, self.disability_weight_pipelines, self.causes)
            ylds_this_step = {key + f'_in_whz_{cat}': value for key, value in ylds_this_step.items()}
            self.years_lived_with_disability.update(ylds_this_step)

            pop.loc[pop_for_cat.index, 'years_lived_with_disability'] += self.disability_weight(pop_for_cat.index)
            self.population_view.update(pop)

