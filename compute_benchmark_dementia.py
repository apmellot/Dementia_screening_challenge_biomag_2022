import numpy as np
import pandas as pd

import pathlib

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
import h5io
import coffeine

DERIV_ROOT = pathlib.Path('/storage/store3/derivatives/biomag_hokuto_bids')
BIDS_ROOT = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto_bids'
)

BENCHMARKS = ['dummy', 'filterbank-riemann']

frequency_bands = {
    "low": (0.1, 1),
    "delta": (1, 4),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 15.0),
    "beta_low": (15.0, 26.0),
    "beta_mid": (26.0, 35.0),
    "beta_high": (35.0, 49)
}


def get_labels(subjects):
    labels = []
    for subject in subjects:
        subject = subject[4:]
        if subject.find('control')==0:
            labels.append('control')
        if subject.find('mci')==0:
            labels.append('mci')
        if subject.find('dementia')==0:
            labels.append('dementia')
        if subject.find('test')==0:
            labels.append('test')
    return labels


def load_data(benchmark):
    participants = pd.read_csv(BIDS_ROOT / "participants.tsv", sep="\t")
    subjects = participants.participant_id
    y = get_labels(subjects)
    if benchmark == 'dummy':
        X = np.zeros(shape=(len(y), 1))
        model = DummyClassifier(strategy='most_frequent')
    elif benchmark == 'filterbank-riemann':
        features = h5io.read_hdf5(
            DERIV_ROOT / f'features_fb_covs.h5')
        covs = [features[sub]['covs'] for sub in subjects]
        covs = np.array(covs)
        X = pd.DataFrame(
            {band: list(covs[:, ii]) for ii, band in
             enumerate(frequency_bands)})
        filter_bank_transformer = coffeine.make_filter_bank_transformer(
            names=list(frequency_bands),
            method='riemann',
            projection_params=dict(scale='auto', n_compo=65)  # n_compo value
        )
        model = make_pipeline(
            filter_bank_transformer, StandardScaler(),
            RidgeCV(alphas=np.logspace(-5, 10, 100)))
    return X, y, model

# if __name__ == "__main__":
