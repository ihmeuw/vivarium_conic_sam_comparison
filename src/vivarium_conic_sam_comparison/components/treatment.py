from vivarium.framework.event import Event

import pandas as pd


class MaternalTreatmentAlgorithm:
    configuration_defaults = {
        "interventions": {
            "maternal_intervention": {
                "coverage_proportion": 0.8,
                "start_date": {
                    "year": 2020,
                    "month": 1,
                    "day": 1
                },
            }
        }
    }

    def __init__(self, intervention_name: str):
        self.intervention_name = intervention_name
        self.configuration_defaults = {"interventions": {f"{self.intervention_name}_intervention":
            MaternalTreatmentAlgorithm.configuration_defaults['interventions']['maternal_intervention']}}

    @property
    def name(self):
        return f'{self.intervention_name}_treatment_algorithm'

    def setup(self, builder):
        config = builder.configuration["interventions"][f"{self.intervention_name}_intervention"]
        self.clock = builder.time.clock()
        self.start_date = pd.Timestamp(**config['start_date'].to_dict())
        self.proportion = config['coverage_proportion']

        self.enrollment_randomness = builder.randomness.get_stream(f'{self.intervention_name}_enrollment')

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
        "interventions": {
            "neonatal_intervention": {
                "whz_target": "all",  # Z-score float or 'all'. Sims at or below eligible
                "coverage_proportion": 0.8,
                "treatment_duration": 365.25,  # days
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
    }

    def __init__(self, intervention_name: str):
        self.intervention_name = intervention_name
        self.configuration_defaults = {"interventions": {f"{self.intervention_name}_intervention":
            NeonatalTreatmentAlgorithm.configuration_defaults['interventions']['neonatal_intervention']}}

    @property
    def name(self):
        return f"{self.intervention_name}_treatment_algorithm"

    def setup(self, builder):
        config = builder.configuration["interventions"][f"{self.intervention_name}_intervention"]
        self.whz_target = config['whz_target']
        self.start_date = pd.Timestamp(**config['start_date'].to_dict())
        self.treatment_age = config['treatment_age']
        self.coverage = config['coverage_proportion']
        self.treatment_duration = pd.Timedelta(days=config['treatment_duration'])

        self.clock = builder.time.clock()

        self.enrollment_randomness = builder.randomness.get_stream(f"{self.intervention_name}_enrollment")

        self.wasting_exposure = builder.value.get_value(f'child_wasting.exposure')

        created_columns = [f'{self.intervention_name}_treatment_start',
                           f'{self.intervention_name}_treatment_end']
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

        pop = pd.DataFrame({f'{self.intervention_name}_treatment_start': pd.NaT,
                            f'{self.intervention_name}_treatment_end': pd.NaT},
                           index=pop_data.index)
        self.pop_view.update(pop)

    def on_time_step(self, event):
        pop = self.pop_view.get(event.index, query="alive == 'alive'")
        treated_idx = self.get_treated_idx(pop, event)

        pop.loc[treated_idx, f'{self.intervention_name}_treatment_start'] = event.time
        pop.loc[treated_idx, f'{self.intervention_name}_treatment_end'] = event.time + self.treatment_duration
        self.pop_view.update(pop)

    def get_treated_idx(self, pop: pd.DataFrame, event: Event):
        # Intervention hasn't started
        if event.time < self.start_date:
            return pd.Index([])

        # Eligible by age
        pop_age_at_event = pop.age + (event.step_size / pd.Timedelta(days=365.25))
        if self.clock() < self.start_date:  # Treatment available this time_step
            # mass treatment of anyone in age range when intervention starts
            eligible_mask = (self.treatment_age['start'] <= pop['age']) & (pop['age'] <= self.treatment_age['end'])
        else:  # past treatment start
            # continuous enrollment of those crossing the age threshold
            eligible_mask = (pop.age < self.treatment_age['start']) & (self.treatment_age['start'] <= pop_age_at_event)

        # Eligible by target
        if self.whz_target != 'all':
            eligible_mask &= self.wasting_exposure(pop.index, skip_post_processor=True) <= self.whz_target + 10

        # Filter already treated
        eligible_mask &= pd.isnull(pop[f'{self.intervention_name}_treatment_start'])

        return self.enrollment_randomness.filter_for_probability(pop.loc[eligible_mask].index, self.coverage)
