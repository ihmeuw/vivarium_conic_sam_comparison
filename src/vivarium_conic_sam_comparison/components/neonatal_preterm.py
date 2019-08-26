from vivarium_public_health.disease import RiskAttributableDisease


class NeonatalPreterm(RiskAttributableDisease):

    @property
    def name(self):
        return "risk_attributable_neonatal_preterm"

    def __init__(self):
        super().__init__('cause.neonatal_preterm_birth', 'risk_factor.low_birth_weight_and_short_gestation')

    def get_exposure_filter(self, distribution, exposure_pipeline, threshold):
        max_weeks_for_preterm = 38

        def exposure_filter(index):
            exposure = exposure_pipeline(index, skip_post_processor=True)
            return exposure.gestation_time <= max_weeks_for_preterm
        return exposure_filter
