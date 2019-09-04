import numpy as np
import pandas as pd
import scipy.stats

from vivarium_public_health.utilities import TargetString


class InterventionEffect:
    configuration_defaults = {
        "intervention": {
            "effect": {
                "duration": 365.25,  # Float days or 'permanent'.  Inclusive of ramp up/down.
                                     # Permanent overrides ramp down.
                "population": {
                    "mean": 0.0,
                    "sd": 0.0
                },
                "individual": {
                    "sd": 0.0
                },
                "ramp_up_duration": 0,  # Length of logistic ramp up in days.
                "ramp_down_duration": 0,  # """
            }
        }
    }

    def __init__(self, intervention_name: str, target: str):
        self.intervention_name = intervention_name
        self.target = TargetString(target)
        self.configuration_defaults = {self.intervention_name:
                                       {f'effect_on_{self.target.name}':
                                        InterventionEffect.configuration_defaults['intervention']['effect']}}

    @property
    def name(self):
        return f'{self.intervention_name}_effect_on_{self.target.name}'

    def setup(self, builder):
        self.config = builder.configuration[self.intervention_name][f'effect_on_{self.target.name}']
        self.duration = pd.Timedelta(days=self.config['duration']) if isinstance(self.config['duration'], float) else self.config['duration']
        self.clock = builder.time.clock()

        self._effect_size = pd.Series()

        self.randomness = builder.randomness.get_stream(self.name)

        builder.value.register_value_modifier(f'{self.target.name}.{self.target.measure}', self.adjust_exposure)

        created_columns = [f'{self.intervention_name}_effect_end']
        required_columns = [f'{self.intervention_name}_treatment_start']
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=created_columns,
                                                 requires_columns=required_columns)
        self.pop_view = builder.population.get_view(created_columns + required_columns)

        self.population_effect = self.get_population_effect_size(self.config.population.mean,
                                                                 self.config.population.sd,
                                                                 'population_effect')

    def on_initialize_simulants(self, pop_data):
        individual_effect = self.get_individual_effect_size(pop_data.index, self.population_effect,
                                                            self.config.individual.sd,
                                                            'individual_effect')
        self._effect_size = self._effect_size.append(individual_effect)

        if self.duration == 'permanent':
            pop = pd.DataFrame({f'{self.intervention_name}_effect_end': pd.NaT})
        else:
            pop = self.pop_view.get(pop_data.index)
            pop[f'{self.intervention_name}_effect_end'] = (pop[f'{self.intervention_name}_treatment_start'] +
                                                           self.duration)
        self.pop_view.update(pop)

    def get_population_effect_size(self, mean, sd, key):
        if sd == 0:
            return mean
        r = np.random.RandomState(self.randomness.get_seed(additional_key=key))
        draw = r.uniform()
        effect = scipy.stats.norm(mean, sd).ppf(draw)
        effect = 0 if effect < 0 else effect  # NOTE: Not allowing negative effect
        return effect

    def get_individual_effect_size(self, index, mean, sd, key):
        if sd == 0:
            return pd.Series(mean, index=index)
        draw = self.randomness.get_draw(index, additional_key=key)
        effect_size = scipy.stats.norm(mean, sd).ppf(draw)
        effect_size[effect_size < 0] = 0.0  # NOTE: Not allowing negative effect
        return pd.Series(effect_size, index=index)

    def adjust_exposure(self, index, exposure):
        effect_size = pd.Series(0, index=index)
        untreated, ramp_up, full_effect, ramp_down, post_effect = self.get_treatment_groups(index)

        effect_size.loc[untreated] = 0
        effect_size.loc[ramp_up] = self.ramp_efficacy(ramp_up)
        effect_size.loc[full_effect] = self._effect_size.loc[full_effect]
        if self.duration == 'permanent':
            effect_size.loc[ramp_down] = self._effect_size[ramp_down]
            effect_size.loc[post_effect] = self._effect_size[post_effect]
        else:
            effect_size.loc[ramp_down] = self.ramp_efficacy(ramp_down, invert=True)
            effect_size.loc[post_effect] = 0

        # FIXME: Hack for lbwsg weirdness for now
        if self.target.name == 'low_birth_weight_and_short_gestation':
            exposure['birth_weight'] += effect_size
        else:
            exposure += effect_size
        return exposure

    def get_treatment_groups(self, index):
        ramp_up_duration = pd.Timedelta(days=self.config.ramp_up_duration)
        ramp_down_duration = pd.Timedelta(days=self.config.ramp_down_duration)

        pop = self.pop_view.get(index)
        untreated = pop.loc[(pop[f'{self.intervention_name}_treatment_start'].isnull())
                            | (self.clock() <= pop[f'{self.intervention_name}_treatment_start'])].index

        ramp_up = pop.loc[(pop[f'{self.intervention_name}_treatment_start'] < self.clock())
                          & (self.clock() < pop[f'{self.intervention_name}_treatment_start'] + ramp_up_duration)].index

        full_effect = pop.loc[(pop[f'{self.intervention_name}_treatment_start'] + ramp_up_duration <= self.clock())
                              & (self.clock() <= pop[f'{self.intervention_name}_effect_end'] - ramp_down_duration)].index

        ramp_down = pop.loc[(pop[f'{self.intervention_name}_effect_end'] - ramp_down_duration < self.clock())
                            & (self.clock() < pop[f'{self.intervention_name}_effect_end'])].index

        post_effect = pop.loc[pop[f'{self.intervention_name}_effect_end'] <= self.clock()].index

        return untreated, ramp_up, full_effect, ramp_down, post_effect

    def ramp_efficacy(self, index, invert=False):
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

        if invert:
            ramp_days = pd.Timedelta(days=self.config.ramp_down_duration)
            growth_rate = 2 / self.config.ramp_down_duration * np.log(p)
            ramp_position = ((pop[f'{self.intervention_name}_effect_end'] + ramp_days / 2) - self.clock()) / pd.Timedelta(days=1)
        else:
            ramp_days = pd.Timedelta(days=self.config.ramp_up_duration)
            growth_rate = 2 / self.config.ramp_up_duration * np.log(p)
            ramp_position = (self.clock() - (pop[f'{self.intervention_name}_treatment_start'] + ramp_days / 2)) / pd.Timedelta(days=1)

        scale = 1 / (1 + np.exp(-growth_rate * ramp_position))
        return scale * self._effect_size[index]

