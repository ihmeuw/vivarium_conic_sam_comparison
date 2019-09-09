import pandas as pd


class SampleHistoryObserver:

    configuration_defaults = {
        'metrics': {
            'sample_history_observer': {
                'sample_size': 1000,
                'fraction_initial_pop': 0.75,  # complement taken from born-in pop
                'path': f'/share/costeffectiveness/results/vivarium_conic_sam_comparison/sample_history.hdf'
            }
        }
    }

    @property
    def name(self):
        return "sample_history_observer"

    def __init__(self):
        self.history_snapshots = []
        self.sample_index = pd.Index([])

    def setup(self, builder):
        self.clock = builder.time.clock()
        self.sample_size = builder.configuration.metrics.sample_history_observer['sample_size']
        self.fraction_initial_pop = builder.configuration.metrics.sample_history_observer['fraction_initial_pop']
        # assume relatively constant births over time.
        sim_start = pd.Timestamp(**builder.configuration.time.start)
        sim_end = pd.Timestamp(**builder.configuration.time.end)
        step_size = pd.Timedelta(days=builder.configuration.time.step_size)
        self.num_samples_each_step = (self.sample_size
                                      * (1. - self.fraction_initial_pop) / ((sim_start - sim_end) / step_size))

        self.path = builder.configuration.metrics.sample_history_observer['path']
        self.randomness = builder.randomness.get_stream("sample_history")

        self.sample_index = pd.Index([])

        # sample from the initial pool and people born in to sim
        builder.population.initializes_simulants(self.on_initialize_simulants)

        columns_required = ['alive', 'age', 'sex', 'entrance_time', 'exit_time',
                            'cause_of_death',
                            'years_of_life_lost',
                            'BEP_treatment_start',
                            'SQ_LNS_treatment_start',
                            'TF_SAM_treatment_start',
                            'neonatal_preterm_birth_event_time',
                            'diarrheal_diseases_event_time',
                            'lower_respiratory_infections_event_time',
                            'measles_event_time',
                            'neonatal_sepsis_and_other_neonatal_infections_event_time',
                            'neonatal_encephalopathy_due_to_birth_asphyxia_and_trauma_event_time',
                            'hemolytic_disease_and_other_neonatal_jaundice_event_time']
        self.population_view = builder.population.get_view(columns_required)

        # keys will become column names in the output
        self.pipelines = {'mortality_rate': builder.value.get_value('mortality_rate'),
                          'disability_weight': builder.value.get_value('disability_weight'),

                          'child_wasting_exposure': builder.value.get_value('child_wasting.exposure'),
                          'child_wasting_raw_exposure': lambda pop_index: builder.value.get_value('child_wasting.exposure')(pop_index, skip_post_processor=True),
                          'child_wasting_raw_exposure_baseline': lambda pop_index: builder.value.get_value('child_wasting.exposure').source(pop_index),

                          'child_stunting_exposure': builder.value.get_value('child_stunting.exposure'),
                          'child_stunting_raw_exposure': lambda pop_index: builder.value.get_value('child_stunting.exposure')(pop_index, skip_post_processor=True),
                          'child_stunting_raw_exposure_baseline': lambda pop_index: builder.value.get_value('child_stunting.exposure').source(pop_index),

                          'low_birth_weight_and_short_gestation_exposure': builder.value.get_value('low_birth_weight_and_short_gestation.exposure'),

                          'low_birth_weight_and_short_gestation_raw_bw_raw_exposure':
                              lambda pop_index: builder.value.get_value('low_birth_weight_and_short_gestation.raw_exposure')(pop_index)['birth_weight'],
                          'low_birth_weight_and_short_gestation_raw_bw_raw_exposure_baseline':
                              lambda pop_index: builder.value.get_value('low_birth_weight_and_short_gestation.raw_exposure').source(pop_index)['birth_weight'],

                          'low_birth_weight_and_short_gestation_raw_gt_raw_exposure':
                              lambda pop_index: builder.value.get_value('low_birth_weight_and_short_gestation.raw_exposure')(pop_index)['gestation_time'],
                          'low_birth_weight_and_short_gestation_raw_gt_raw_exposure_baseline':
                              lambda pop_index: builder.value.get_value('low_birth_weight_and_short_gestation.raw_exposure').source(pop_index)['gestation_time'],

                          'diarrheal_diseases_incidence_rate':
                              builder.value.get_value('diarrheal_diseases.incidence_rate'),
                          'lower_respiratory_infections_incidence_rate':
                              builder.value.get_value('lower_respiratory_infections.incidence_rate'),
                          'measles_incidence_rate': builder.value.get_value('measles.incidence_rate'),
                          }

        builder.event.register_listener('collect_metrics', self.record)
        builder.event.register_listener('simulation_end', self.dump)

        self.builder = builder

    def on_initialize_simulants(self, pop_data):
        """Sample from the initial pop and those born in the sim."""
        draw = self.randomness.get_draw(pop_data.index)
        priority_index = [i for d, i in sorted(zip(draw, pop_data.index), key=lambda x:x[0])]
        if len(self.sample_index) == 0:  # sampling initial population
            initial_sample_size = (self.sample_size * self.fraction_initial_pop)
            self.sample_index = self.sample_index.append(pd.Index(priority_index[:initial_sample_size]))
        else:
            self.sample_index = self.sample_index.append(pd.Index(priority_index[:self.num_samples_each_step]))

    def record(self, event):
        pop = self.population_view.get(self.sample_index)

        pipeline_results = []
        for name, pipeline in self.pipelines.items():
            values = pipeline(pop.index)
            if name == 'mortality_rate':
                values = values.sum(axis=1)
            values = values.rename(name)
            pipeline_results.append(values)

        record = pd.concat(pipeline_results + [pop], axis=1)
        record['time'] = self.clock()
        record.index.rename("simulant", inplace=True)
        record.set_index('time', append=True, inplace=True)

        self.history_snapshots.append(record)

    def dump(self, event):
        sample_history = pd.concat(self.history_snapshots, axis=0)
        sample_history.to_hdf(self.path, key='histories')
