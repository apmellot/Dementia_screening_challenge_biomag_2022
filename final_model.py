import pathlib
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import h5io
import coffeine

DERIV_ROOT = pathlib.Path('/storage/store3/derivatives/biomag_hokuto_bids')
FEATURES_ROOT = DERIV_ROOT
BIDS_ROOT = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto_bids'
)
ROOT = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto'
)
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
    test_subjects = []
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
        elif subject.find('test') == 4:
            test_subjects.append(subject)
    return train_subjects, test_subjects, train_labels


def get_subjects_age(age, labels):
    subjects = []
    for label in labels:
        site_info = pd.read_excel(ROOT / 'hokuto_profile.xlsx', sheet_name=label)
        for i in range(site_info.shape[0]):
            if site_info['Age'].iloc[i]>= age:
                subjects.append('sub-' + site_info['ID'].iloc[i][7:])
    print(len(subjects))
    return subjects


def run_model():
    all_subjects = get_subjects_age(50, ['control', 'dementia', 'mci', 'test_data'])
    train_subjects, test_subjects, y_train = get_subjects_labels(all_subjects)
    features_psd = h5io.read_hdf5(FEATURES_ROOT / 'features_features_psd.h5')
    X_psd_train = np.concatenate(
        [features_psd[sub][None, :] for sub in train_subjects],
        axis=0
    )
    X_psd_test = np.concatenate(
        [features_psd[sub][None, :] for sub in test_subjects],
        axis=0
    )
    reg = 1e-7
    rank = 120
    features_covs = h5io.read_hdf5(DERIV_ROOT / 'features_fb_covs.h5')
    covs_train = [features_covs[sub]['covs'] for sub in train_subjects]
    covs_train = np.array(covs_train)
    X_covs_train = pd.DataFrame(
        {band: list(covs_train[:, ii]) for ii, band in
            enumerate(frequency_bands)})
    covs_test = [features_covs[sub]['covs'] for sub in test_subjects]
    covs_test = np.array(covs_test)
    X_covs_test = pd.DataFrame(
        {band: list(covs_test[:, ii]) for ii, band in
            enumerate(frequency_bands)})
    
    filter_bank_transformer = coffeine.make_filter_bank_transformer(
        names=list(frequency_bands),
        method='riemann',
        projection_params=dict(scale='auto', n_compo=rank, reg=reg)
    )
    X_covs_train = filter_bank_transformer.fit_transform(X_covs_train)
    X_covs_test = filter_bank_transformer.transform(X_covs_test)
    X_train = np.concatenate((X_psd_train, X_covs_train), axis=1)
    X_test = np.concatenate((X_psd_test, X_covs_test), axis=1)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=1e4)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    test_subjects = ['hokuto_' + sub[4:] for sub in test_subjects]
    return test_subjects, y_pred, y_proba



if __name__ == "__main__":
    test_subjects, y_pred, y_proba = run_model()
    print(y_proba)
    results = pd.DataFrame(
        {
            'Test data ID': test_subjects,
            'Estimated diagnoses': y_pred,
            'Plausibility': (y_proba.max(axis=1)*100).astype(int)
        }
    )
    results.to_csv('./results/results_proba_covs_psd.csv')