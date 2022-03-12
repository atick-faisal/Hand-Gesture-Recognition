import os
import numpy as np
import pandas as pd

from .dir_utils import clean_dir
from .filters import LowPassFilter


def preposses_data(
    data: pd.DataFrame,
    window_len: int = 150
) -> np.ndarray:
    data.fillna(inplace=True)

    # ... smoothing
    data['flex_1'] = data['flex_1'].rolling(3).median()
    data['flex_2'] = data['flex_2'].rolling(3).median()
    data['flex_3'] = data['flex_3'].rolling(3).median()
    data['flex_4'] = data['flex_4'].rolling(3).median()
    data['flex_5'] = data['flex_5'].rolling(3).median()

    flx1 = data['flex_1'].to_numpy().reshape(-1, window_len)
    flx2 = data['flex_2'].to_numpy().reshape(-1, window_len)
    flx3 = data['flex_3'].to_numpy().reshape(-1, window_len)
    flx4 = data['flex_4'].to_numpy().reshape(-1, window_len)
    flx5 = data['flex_5'].to_numpy().reshape(-1, window_len)

    accx = data['ACCx'].to_numpy()
    accy = data['ACCy'].to_numpy()
    accz = data['ACCz'].to_numpy()

    accx = LowPassFilter.apply(accx).reshape(-1, window_len)
    accy = LowPassFilter.apply(accy).reshape(-1, window_len)
    accz = LowPassFilter.apply(accz).reshape(-1, window_len)

    channels = np.stack([
        flx1, flx2, flx3, flx4, flx5,
        accx, accy, accz
    ], axis=-1)

    return channels
