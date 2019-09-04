from vivarium.framework.event import Event

import pandas as pd


class MaternalTreatmentAlgorithm:

    configuration_defaults = {
        "maternal_intervention": {
            "proportion": 0.8,
            "start_date": {
                "year": 2020,
                "month": 1,
                "day": 1
            },
        }
    }

    def __init__(self, intervention_name: str):
        self.intervention_name = intervention_name
        self.configuration_defaults = {self.intervention_name:
                                       MaternalTreatmentAlgorithm.configuration_defaults['maternal_intervention']}

    @property
    def name(self):
        return f'{self.intervention_name}_maternal_intervention'

    def setup(self, builder):
        config = builder.configuration[self.intervention_name]
        self.clock = builder.time.clock()
        self.start_date = pd.Timestamp(**config['start_date'].to_dict())
        self.proportion = config.proportion

        self.enrollment_randomness = builder.randomness.get_stream(f'{self.intervention_name}_enrollment')
        self.effect_randomness = builder.randomness.get_stream('effect_draw')

        columns_created = [f'{self.intervention_name}_treatment_start']
        self.population_view = builder.population.get_view(columns_created)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created)

    def on_initialize_simulants(self, pop_data):
        # Check that start date isn't set before sim start
        if pop_data.user_data['sim_state'] == 'setup' and pop_data.creation_time >= self.start_date:
            raise NotImplementedError(f"{self.intervention_name} intervention must begin strictly "
                                      f"after the intervention start date.")

        pop = pd.DataFrame({f'{self.intervention_name}_treatment_start': pd.NaT}, index=pop_data.index)
        if pop_data.creation_time >= self.start_date:
            treatment_probability = self.proportion
            treated = self.enrollment_randomness.filter_for_probability(pop.index, treatment_probability)
            # This is really a maternal treatment. To signify the mother was treated, treatment
            # start is initialized to simulant creation time
            pop.loc[treated, f'{self.intervention_name}_treatment_start'] = pop_data.creation_time
        self.population_view.update(pop)


class NeonatalTreatmentAlgorithm:

    configuration_defaults = {
        "neonatal_intervention": {
            "whz_target": "all",  # Z-score float or 'all'. Sims at or below eligible
            "proportion": 0.8,
            "start_date": {
                "year": 2020,
                "month": 1,
                "day": 1
            },
            "treatment_age": {
                "start": 0.5,
                "end": 1.0
            },
        }
    }

    def __init__(self, intervention_name: str):
        self.intervention_name = intervention_name
        self.configuration_defaults = {self.intervention_name:
                                       NeonatalTreatmentAlgorithm.configuration_defaults['neonatal_intervention']}

    @property
    def name(self):
        return f"{self.intervention_name}_treatment_algorithm"

    def setup(self, builder):
        config = builder.configuration[self.intervention_name]
        self.whz_target = config.whz_target
        self.start_date = pd.Timestamp(**config['start_date'].to_dict())
        self.treatment_age = config['treatment_age']
        self.coverage = config['proportion']

        self.clock = builder.time.clock()

        self.rand = builder.randomness.get_stream(f"{self.intervention_name}_enrollment")

        self.wasting_exposure = builder.value.get_value(f'child_wasting.exposure')

        created_columns = [f'{self.intervention_name}_treatment_start']
        required_columns = ['age']
        self.pop_view = builder.population.get_view(created_columns + required_columns)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=created_columns,
                                                 requires_columns=required_columns)

        builder.event.register_listener('time_step', self.on_time_step)

    def on_initialize_simulants(self, pop_data):
        # Check that start date isn't set before sim start
        if pop_data.user_data['sim_state'] == 'setup' and pop_data.creation_time >= self.start_date:
            raise NotImplementedError(f"{self.intervention_name} intervention must begin strictly "
                                      f"after the intervention start date.")

        pop = pd.DataFrame({f'{self.intervention_name}_treatment_start': pd.NaT},
                           index=pop_data.index)
        self.pop_view.update(pop)

    def on_time_step(self, event):
        pop = self.pop_view.get(event.index, query="alive == 'alive'")
        treated_idx = self.get_treated_idx(pop, event)

        pop.loc[treated_idx, f'{self.intervention_name}_treatment_start'] = event.time
        self.pop_view.update(pop)

    def get_treated_idx(self, pop: pd.DataFrame, event: Event):
        # Eligible by age
        pop_age_at_event = pop.age + (event.step_size / pd.Timedelta(days=365.25))
        if self.clock() < self.start_date <= event.time:
            # mass treatment when intervention starts
            eligible_idx = pop.loc[(self.treatment_age['start'] <= pop['age']) &
                                   (pop['age'] <= self.treatment_age['end'])].index
            treated_idx = self.rand.filter_for_probability(eligible_idx, self.coverage)
        elif self.start_date <= self.clock():
            # continuous enrollment of those crossing the boundary
            eligible_pop = pop.loc[(pop.age < self.treatment_age['start']) &
                                   (self.treatment_age['start'] <= pop_age_at_event)]
            treated_idx = self.rand.filter_for_probability(eligible_pop.index, self.coverage)
        else:
            # Intervention hasn't started.
            treated_idx = pd.Index([])

        # Eligible by target
        if self.whz_target != 'all':
            treated_idx &= (self.wasting_exposure(pop.index) <= self.whz_target + 10)

        # Filter already treated
        treated_idx &= (pd.isnull(pop[f'{self.intervention_name}_treatment_start']))

        return treated_idx
