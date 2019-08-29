import pandas as pd


class SAMIntervention:

    configuration_defaults = {
        'sam_intervention': {
            'proportion': 1.0,
            'birth_weight_shift': 0,  # grams
            'gestation_time_shift': 0,  # weeks
            'stunting_shift': 0,  # z-score
            'wasting_shift': 0,  # z-score
            'underweight_shift': 0,  # z-score
        }
    }

    def __init__(self):
        self.name = 'sam_intervention'

    def setup(self, builder):
        self.start_time = pd.Timestamp(**builder.configuration.time.start.to_dict())
        self.config = builder.configuration['sam_intervention']
        validate_configuration(self.config.to_dict())
        self.randomness = builder.randomness.get_stream('sam_intervention_enrollment')
        columns_created = ['sam_treatment_status']
        self.population_view = builder.population.get_view(columns_created)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created)
        builder.value.register_value_modifier('low_birth_weight_and_short_gestation.raw_exposure',
                                              self.adjust_lbwsg)
        builder.value.register_value_modifier('child_stunting.exposure',
                                              self.adjust_stunting)
        builder.value.register_value_modifier('child_wasting.exposure',
                                              self.adjust_wasting)
        builder.value.register_value_modifier('child_underweight.exposure',
                                              self.adjust_underweight)

    def on_initialize_simulants(self, pop_data):
        pop = pd.DataFrame({'sam_treatment_status': 'not_treated'}, index=pop_data.index)
        if pop_data.creation_time > self.start_time:
            treatment_probability = self.config.proportion
            treated = self.randomness.filter_for_probability(pop.index, treatment_probability)
            pop.loc[treated, 'sam_treatment_status'] = 'treated'

        self.population_view.update(pop)

    def adjust_lbwsg(self, index, exposure):
        pop = self.population_view.get(index)
        exposure['birth_weight'] += self.config.birth_weight_shift * (pop.sam_treatment_status == 'treated')
        exposure['gestation_time'] += self.config.gestation_time_shift * (pop.sam_treatment_status == 'treated')
        return exposure

    def adjust_stunting(self, index, exposure):
        pop = self.population_view.get(index)
        return exposure + self.config.stunting_shift * (pop.sam_treatment_status == 'treated')

    def adjust_wasting(self, index, exposure):
        pop = self.population_view.get(index)
        return exposure + self.config.wasting_shift * (pop.sam_treatment_status == 'treated')

    def adjust_underweight(self, index, exposure):
        pop = self.population_view.get(index)
        return exposure + self.config.underweight_shift * (pop.sam_treatment_status == 'treated')


def validate_configuration(config):
    if not (0 <= config['proportion'] <= 1):
        raise ValueError(f'The proportion for SAM intervention must be between 0 and 1.'
                         f'You specified {config.proportion}.')
    for key in config:
        if 'shift' in key and config[key] < 0:
            raise ValueError(f'Additive shift for {key} must be positive.')
