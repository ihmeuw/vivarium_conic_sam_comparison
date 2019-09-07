import numpy as np
import pandas as pd


def convert_whz_to_categorical(whz_series):
    # whz z-score has 10 added to it.
    whz_categorical = pd.Series('', index=whz_series.index)
    categories = {
            # Note that actual expsure is z-score + 10
            'not_eligible': (-np.inf, 0),
            'lt_-3': (0, 7),
            '-3_to_-2': (7, 8),
            '-2_to_-1': (8, 9),
            'unexposed': (9, np.inf)
    }
    for cat in categories.keys():
        in_cat_mask = ((categories[cat][0]) < whz_series) & (whz_series <= (categories[cat][1]))
        whz_categorical.loc[in_cat_mask] = cat
    assert sum(whz_categorical == '') == 0
    return whz_categorical

