import numpy as np
import pandas as pd
import scipy.stats

from vivarium_public_health.utilities import TargetString


class InterventionEffect:
    configuration_defaults = {
        "neonatal_intervention": {
            "effect": {
                "population": {
                    "mean": 0.0,
                    "sd": 0.0
                },
                "individual": {
                    "sd": 0.0
                },
                "ramp_up_duration": 0,  # Length of logistic ramp up/down in days.
                "ramp_down_duration": 0,
                "permanent": False  # will ignore ramp down if true
            }
        }
    }

    def __init__(self, intervention_name: str, target: str):
        self.intervention_name = intervention_name
        self.target = TargetString(target)
        self.configuration_defaults = {self.intervention_name:
                                       {f'effect_on_{self.target.name}':
                                        InterventionEffect.configuration_defaults['neonatal_intervention']['effect']}}

    @property
    def name(self):
        return f'{self.intervention_name}_effect_on_{self.target.name}'

    def setup(self, builder):
        self.config = builder.configuration[self.intervention_name][f'effect_on_{self.target.name}']
        self.clock = builder.time.clock()

        self._effect_size = pd.Series()

        self.randomness = builder.randomness.get_stream(self.name)

        builder.value.register_value_modifier(f'{self.target.name}.{self.target.measure}', self.adjust_exposure)

        builder.population.initializes_simulants(self.on_initialize_simulants)
        self.pop_view = builder.population.get_view([f'{self.intervention_name}_treatment_start',
                                                     f'{self.intervention_name}_treatment_end'])

        self.population_effect = self.get_population_effect_size(self.config.population.mean,
                                                                 self.config.population.sd,
                                                                 'population_effect')

    def on_initialize_simulants(self, pop_data):
        individual_effect = self.get_individual_effect_size(pop_data.index, self.population_effect,
                                                            self.config.individual.sd,
                                                            'individual_effect')
        self._effect_size = self._effect_size.append(individual_effect)

    def get_population_effect_size(self, mean, sd, key):
        r = np.random.RandomState(self.randomness.get_seed(additional_key=key))
        draw = r.uniform()
        effect = scipy.stats.norm(mean, sd).ppf(draw)
        effect = 0 if effect < 0 else effect  # NOTE: Not allowing negative effect
        return effect

    def get_individual_effect_size(self, index, mean, sd, key):
        draw = self.randomness.get_draw(index, additional_key=key)
        effect_size = scipy.stats.norm(mean, sd).ppf(draw)
        effect_size[effect_size < 0] = 0.0  # NOTE: Not allowing negative effect
        return pd.Series(effect_size, index=index)

    def adjust_exposure(self, index, exposure):
        effect_size = pd.Series(0, index=index)
        untreated, ramp_up, full_treatment, ramp_down, post_treatment = self.get_treatment_groups(index)

        effect_size.loc[untreated] = 0
        effect_size.loc[ramp_up] = self.ramp_efficacy(ramp_up)
        effect_size.loc[full_treatment] = self._effect_size.loc[full_treatment]
        if self.config.permanent:
            effect_size.loc[ramp_down] = self._effect_size[ramp_down]
            effect_size.loc[post_treatment] = self._effect_size[post_treatment]
        else:
            effect_size.loc[ramp_down] = self.ramp_efficacy(ramp_down, invert=True)
            effect_size.loc[post_treatment] = 0

        return exposure + effect_size

    def get_treatment_groups(self, index):
        ramp_up_duration = pd.Timedelta(days=self.config.ramp_up_duration)
        ramp_down_duration = pd.Timedelta(days=self.config.ramp_down_duration)

        pop = self.pop_view.get(index)
        untreated = pop.loc[(pop[f'{self.intervention_name}_treatment_start'].isnull())
                            | (self.clock() <= pop[f'{self.intervention_name}_treatment_start'])].index

        ramp_up = pop.loc[(pop[f'{self.intervention_name}_treatment_start'] < self.clock())
                          & (self.clock() < pop[f'{self.intervention_name}_treatment_start'] + ramp_up_duration)].index

        full_treatment = pop.loc[(pop[f'{self.intervention_name}_treatment_start'] + ramp_up_duration <= self.clock())
                                 & (self.clock() <= pop[f'{self.intervention_name}_treatment_end'] - ramp_down_duration)].index

        ramp_down = pop.loc[(pop[f'{self.intervention_name}_treatment_end'] - ramp_down_duration < self.clock())
                            & (self.clock() < pop[f'{self.intervention_name}_treatment_end'])].index

        post_treatment = pop.loc[pop[f'{self.intervention_name}_treatment_end'] <= self.clock()].index

        return untreated, ramp_up, full_treatment, ramp_down, post_treatment

    def ramp_efficacy(self, index, ramp_duration, invert=False):
        """Logistic growth/decline of effect size.

        We're using a logistic function here to give a smooth treatment ramp.
        A logistic function has the form L/(1 - e**(-k * (t - t0))
        Where
        L  : function maximum
        t0 : center of the function
        k  : growth rate

        We want the function to be 0 for times below the treatment start,
        then to ramp up to the maximum over sum duration, stay there until
        the treatment stops, then ramp back down over the same duration.
        This means we effectively want to squeeze a logistic function into
        the discontinuities between a step function.  Making a function
        that smoothly transitions would be more math than I want to do right
        now.  Making a function that almost smoothly transitions is pretty
        easy and involves picking a growth rate that gets us very close
        to 0 and the maximum effect when we transition between constant
        effect sizes and the growth periods.

        I've parameterized in terms of the inverse of the  proportion of the
        maximum effect size, p, so that the jump between the different
        sections of the function is equal to (1 / p) * L.

        """
        if index.empty:
            return pd.Series()

        pop = self.pop_view.get(index)
        # 1/p is the proportion of the maximum effect.
        # Size of the discontinuity between constant and logistic functions.
        p = 10_000
        growth_rate = 2 / ramp_duration * np.log(p)
        ramp_days = pd.Timedelta(days=ramp_duration)

        if invert:
            ramp_position = ((pop[f'{self.intervention_name}_treatment_end'] + ramp_days / 2) - self.clock()) / pd.Timedelta(days=1)
        else:
            ramp_position = (self.clock() - (pop[f'{self.intervention_name}_treatment_start'] + ramp_days / 2)) / pd.Timedelta(days=1)

        scale = 1 / (1 + np.exp(-growth_rate * ramp_position))
        return scale * self._effect_size[index]
