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
import dameeg

DERIV_ROOT = pathlib.Path('/storage/store3/derivatives/biomag_hokuto_bids')
FEATURES_ROOT = DERIV_ROOT
BIDS_ROOT = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto_bids'
)
ROOT = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto'
)

BENCHMARKS = ['dummy', 'features-psd', 'filterbank-riemann', 'filterbank-riemann-da']
# BENCHMARKS = ['dummy', 'features_psd']
# BENCHMARKS = ['dummy']
# BENCHMARKS = ['dummy', 'filterbank-riemann', 'filterbank-riemann-da']
N_JOBS = 5
RANDOM_STATE = 42

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


def get_site(labels):
    subjects_A = []
    subjects_B = []
    for label in labels:
        site_info = pd.read_excel(ROOT / 'hokuto_profile.xlsx', sheet_name=label)
        for i in range(site_info.shape[0]):
            if site_info['Site'].iloc[i] == 'A':
                subjects_A.append('sub-' + site_info['ID'].iloc[i][7:])
            if site_info['Site'].iloc[i] == 'B':
                subjects_B.append('sub-' + site_info['ID'].iloc[i][7:])
    return subjects_A, subjects_B



def load_data(benchmark):
    participants = pd.read_csv(BIDS_ROOT / "participants.tsv", sep="\t")
    all_subjects = participants.participant_id
    train_subjects, y = get_subjects_labels(all_subjects)

    # Dummy model
    if benchmark == 'dummy':
        X = np.zeros(shape=(len(y), 1))
        model = DummyClassifier(strategy='most_frequent')

    # Riemann
    elif benchmark == 'filterbank-riemann':
        features = h5io.read_hdf5(DERIV_ROOT / 'features_fb_covs.h5')
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
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        )

    # Riemann + domain adaptation
    elif benchmark == 'filterbank-riemann-da':
        reg = 1e-30
        features = h5io.read_hdf5(DERIV_ROOT / 'features_fb_covs.h5')
        subjects_A, subjects_B = get_site(['control', 'dementia', 'mci'])
        _, y = get_subjects_labels(subjects_A + subjects_B)
        covs_A = [features[sub]['covs'] + reg*np.eye(features[sub]['covs'].shape[1])
                  for sub in subjects_A]
        covs_A = np.array(covs_A)
        covs_B = [features[sub]['covs'] + reg*np.eye(features[sub]['covs'].shape[1])
                  for sub in subjects_B]
        covs_B = np.array(covs_B)
        covs_A_aligned = np.zeros(covs_A.shape)
        covs_B_aligned = np.zeros(covs_B.shape)
        for i in range(len(frequency_bands.keys())):
            (covs_A_aligned[:, i],
             covs_B_aligned[:, i]) = dameeg.recenter_rescale.align_recenter_rescale(covs_A[:, i],
                                                                                    covs_B[:, i])
        X_A = pd.DataFrame(
            {band: list(covs_A_aligned[:, ii]) for ii, band in
                enumerate(frequency_bands)})
        X_B = pd.DataFrame(
            {band: list(covs_B_aligned[:, ii]) for ii, band in
                enumerate(frequency_bands)})
        X = pd.concat([X_A, X_B], axis=0)
        filter_bank_transformer = coffeine.make_filter_bank_transformer(
            names=list(frequency_bands),
            method='riemann',
            projection_params=dict(scale='auto', n_compo=65)  # n_compo value
        )
        model = make_pipeline(
            filter_bank_transformer, StandardScaler(),
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        )

    # Simple features from PSD
    elif benchmark == 'features-psd':
        features = h5io.read_hdf5(FEATURES_ROOT / 'features_features_psd.h5')
        X = np.concatenate(
            [features[sub][None, :] for sub in train_subjects],
            axis=0
        )
        model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        )
    return X, y, model


def run_benchmark_cv(benchmark):
    X, y, model = load_data(benchmark=benchmark)
    cv = StratifiedShuffleSplit(
        n_splits=20,
        test_size=0.2,
        random_state=RANDOM_STATE
    )
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
    # print(f'Accuracy of benchmark {benchmark} : \n', results['accuracy'])
    return results


if __name__ == "__main__":
    for benchmark in BENCHMARKS:
        results_df = run_benchmark_cv(benchmark)
        if results_df is not None:
            results_df.to_csv(
                f"./results/benchmark-{benchmark}.csv")
            mean_accuracy = results_df['accuracy'].mean()
            std_accuracy = results_df['accuracy'].std()
            print(f'Mean accuracy of {benchmark} model: {mean_accuracy} +- {std_accuracy}')
