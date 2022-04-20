import numpy as np
import pandas as pd

import pathlib

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.metrics import make_scorer, accuracy_score

import h5io
import coffeine

DERIV_ROOT = pathlib.Path('/storage/store3/derivatives/biomag_hokuto_bids')
BIDS_ROOT = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto_bids'
)

BENCHMARKS = ['dummy', 'filterbank-riemann']
# BENCHMARKS = ['dummy']
N_JOBS = 5

frequency_bands = {
    "low": (0.1, 1),
    "delta": (1, 4),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 15.0),
    "beta_low": (15.0, 26.0),
    "beta_mid": (26.0, 35.0),
    "beta_high": (35.0, 49)
}


def get_subjects_labels(all_subjects):
    train_subjects = []
    # test_subjects = []
    train_labels = []
    for subject in all_subjects:
        # subject = subject[4:]
        if subject.find('control') == 4:
            train_labels.append('control')
            train_subjects.append(subject)
        elif subject.find('mci') == 4:
            train_labels.append('mci')
            train_subjects.append(subject)
        elif subject.find('dementia') == 4:
            train_labels.append('dementia')
            train_subjects.append(subject)
        # elif subject.find('test') == 0:
        #     test_subjects.append(subject)
    return train_subjects, train_labels


def load_data(benchmark):
    participants = pd.read_csv(BIDS_ROOT / "participants.tsv", sep="\t")
    all_subjects = participants.participant_id
    train_subjects, y = get_subjects_labels(all_subjects)
    if benchmark == 'dummy':
        X = np.zeros(shape=(len(y), 1))
        model = DummyClassifier(strategy='most_frequent')
    elif benchmark == 'filterbank-riemann':
        features = h5io.read_hdf5(
            DERIV_ROOT / 'features_fb_covs.h5')
        covs = [features[sub]['covs'] for sub in train_subjects]
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
            RandomForestClassifier(n_estimators=100))
    return X, y, model


def run_benchmark_cv(benchmark):
    X, y, model = load_data(benchmark=benchmark)
    cv = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=42)
    scoring = make_scorer(accuracy_score)

    print("Running cross validation ...")
    scores = cross_validate(
        model, X, y, cv=cv, scoring=scoring,
        n_jobs=N_JOBS)
    print("... done.")

    results = pd.DataFrame(
        {'accuracy': scores['test_score'],
         'fit_time': scores['fit_time'],
         'score_time': scores['score_time'],
         'benchmark': benchmark}
    )
    print(f'Accuracy of benchmark {benchmark} : \n', results['accuracy'])
    return results


if __name__ == "__main__":
    for benchmark in BENCHMARKS:
        results_df = run_benchmark_cv(benchmark)
        if results_df is not None:
            results_df.to_csv(
                f"./results/benchmark-{benchmark}.csv")
