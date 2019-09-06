from collections import Counter
import pandas as pd

from vivarium_public_health.utilities import EntityString
from vivarium_public_health.metrics.disability import Disability
from vivarium_public_health.metrics.utilities import get_years_lived_with_disability
from vivarium_public_health.metrics.utilities import get_age_sex_filter_and_iterables, get_age_bins, get_output_template


class WHZDisabilityObserver(Disability):

    Disability.configuration_defaults.update(
        {'metrics': {
            'disability': {
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
        return 'whz_disability_observer'

    def setup(self, builder):
        super().setup(builder)
        self.whz_exposure = builder.value.get_value('child_wasting.exposure')

    def on_time_step_prepare(self, event):
        # Almost the same process, just additionally subset by WHZ cat before using utilities.
        if not self.config.by_whz:
            super().on_time_step_prepare(event)
            return

        pop = self.population_view.get(event.index, query='tracked == True and alive == "alive"')
        whz_exposure = self.whz_exposure(pop.index)
        for cat in whz_exposure.unique():
            pop_for_cat = pop.loc[whz_exposure == cat]

            ylds_this_step = get_years_lived_with_disability(pop_for_cat, self.config.to_dict(),
                                                             self.clock().year, self.step_size(),
                                                             self.age_bins, self.disability_weight_pipelines, self.causes)
            ylds_this_step = {key + f'_in_whz_{self.readable_cats[cat]}': value for key, value in ylds_this_step.items()}
            self.years_lived_with_disability.update(ylds_this_step)

            pop.loc[pop_for_cat.index, 'years_lived_with_disability'] += self.disability_weight(pop_for_cat.index)
            self.population_view.update(pop)


class CatStratRiskObserver:
    """ An observer for a categorical risk factor also stratified by age, sex, and year.
    This component observes the number of simulants in each age
    group who are alive and in each category of risk at the specified sample date each year
    (the sample date defaults to July, 1, and can be set in the configuration).
    Here is an example configuration to change the sample date to Dec. 31:
    .. code-block:: yaml
        {risk_name}_observer:
            sample_date:
                month: 12
                day: 31
    """
    configuration_defaults = {
        'metrics': {
            'risk_observer': {
                'categories': ['cat1', 'cat2', 'cat3', 'cat4'],
                'sample_date': {
                    'month': 7,
                    'day': 1
                },
                'by_age': True,
                'by_sex': True,
                'by_year': True,
            }
        }
    }

    def __init__(self, risk: str):
        """
        Parameters
        ----------
        risk :
        the type and name of a risk, specified as "type.name". Type is singular.
        """
        self.risk = EntityString(risk)
        self.configuration_defaults = {'metrics': {
            f'{self.risk.name}_observer': CatStratRiskObserver.configuration_defaults['metrics']['risk_observer']
        }}

    @property
    def name(self):
        return f'categorical_stratified_risk_observer.{self.risk}'

    def setup(self, builder):
        self.data = {}
        self.config = builder.configuration[f'metrics'][f'{self.risk.name}_observer']
        self.clock = builder.time.clock()
        self.categories = self.config.categories
        self.age_bins = get_age_bins(builder)
        self.category_counts = Counter()

        self.population_view = builder.population.get_view(['alive', 'age', 'sex'], query='alive == "alive"')

        self.exposure = builder.value.get_value(f'{self.risk.name}.exposure')
        builder.value.register_value_modifier('metrics', self.metrics)

        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

    def on_collect_metrics(self, event):
        """Records counts of risk exposed by category."""
        pop = self.population_view.get(event.index)

        if self.should_sample(event.time):
            age_sex_filter, (ages, sexes) = get_age_sex_filter_and_iterables(self.config, self.age_bins)
            group_counts = {}
            exposure = self.exposure(pop.index)

            for group, age_group in ages:
                start, end = age_group.age_group_start, age_group.age_group_end
                for sex in sexes:
                    filter_kwargs = {'age_group_start': start, 'age_group_end': end, 'sex': sex, 'age_group': group}
                    group_filter = age_sex_filter.format(**filter_kwargs)
                    in_group = pop.query(group_filter) if group_filter and not pop.empty else pop

                    for cat in self.categories:
                        base_key = get_output_template(**self.config).substitute(
                            measure=f'{self.risk.name}_{cat}_exposed', year=self.clock().year)
                        group_key = base_key.substitute(**filter_kwargs)
                        group_counts[group_key] = (exposure.loc[in_group.index] == cat).sum()

            self.category_counts.update(group_counts)

    def should_sample(self, event_time: pd.Timestamp) -> bool:
        """Returns true if we should sample on this time step."""
        sample_date = pd.Timestamp(event_time.year, self.config.sample_date.month, self.config.sample_date.day)
        return self.clock() <= sample_date < event_time

    def generate_sampling_frame(self) -> pd.DataFrame:
        """Generates an empty sampling data frame."""
        sample = pd.DataFrame({f'{cat}': 0 for cat in self.categories}, index=self.age_bins.index)
        return sample

    def metrics(self, index, metrics):
        metrics.update(self.category_counts)
        return metrics

    def __repr__(self):
        return f"CategoricalStratifiedRiskObserver({self.risk})"