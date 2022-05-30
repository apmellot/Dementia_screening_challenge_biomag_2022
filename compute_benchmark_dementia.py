import itertools
import numpy as np
import pandas as pd
import pathlib

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, make_scorer, confusion_matrix, accuracy_score

import h5io
import coffeine
from utils.spatial_filters import ProjCommonSpace

import matplotlib.pyplot as plt

DERIV_ROOT = pathlib.Path('/storage/store3/derivatives/biomag_hokuto_bids')
ROOT = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto'
)

BENCHMARKS = ['dummy', 'features-psd', 'filterbank-riemann', 'psd-filterbank-riemann']
# BENCHMARKS = ['features-psd']

N_JOBS = -10
RANDOM_STATE = 66

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
    train_labels = []
    for subject in all_subjects:
        if subject.find('control') == 4:
            train_labels.append('control')
            train_subjects.append(subject)
        elif subject.find('mci') == 4:
            train_labels.append('mci')
            train_subjects.append(subject)
        elif subject.find('dementia') == 4:
            train_labels.append('dementia')
            train_subjects.append(subject)
    return train_subjects, train_labels


def get_site(labels, subjects):
    subjects_A = []
    subjects_B = []
    for label in labels:
        site_info = pd.read_excel(ROOT / 'hokuto_profile.xlsx', sheet_name=label)
        for i in range(site_info.shape[0]):
            subject = 'sub-' + site_info['ID'].iloc[i][7:]
            if site_info['Site'].iloc[i] == 'A' and subject in subjects:
                subjects_A.append(subject)
            if site_info['Site'].iloc[i] == 'B' and subject in subjects:
                subjects_B.append(subject)
    return subjects_A, subjects_B


def get_subjects_age(age, labels):
    subjects = []
    for label in labels:
        site_info = pd.read_excel(ROOT / 'hokuto_profile.xlsx', sheet_name=label)
        for i in range(site_info.shape[0]):
            if site_info['Age'].iloc[i]>= age:
                subjects.append('sub-' + site_info['ID'].iloc[i][7:])
    print(len(subjects))
    return subjects


def load_data(benchmark):
    all_subjects = get_subjects_age(50, ['control', 'dementia', 'mci']) # we only keep subjects above 50 years
    train_subjects, y = get_subjects_labels(all_subjects)
    rank = 120

    # Dummy model
    if benchmark == 'dummy':
        X = np.zeros(shape=(len(y), 1))
        model = DummyClassifier(strategy='most_frequent')

    # Riemann fb covs
    elif benchmark == 'filterbank-riemann':
        reg = 1e-7
        features = h5io.read_hdf5(DERIV_ROOT / 'features_fb_covs.h5')
        covs = [features[sub]['covs'] for sub in train_subjects]
        covs = np.array(covs)
        X = pd.DataFrame(
            {band: list(covs[:, ii]) for ii, band in
             enumerate(frequency_bands)})
        filter_bank_transformer = coffeine.make_filter_bank_transformer(
            names=list(frequency_bands),
            method='riemann',
            projection_params=dict(scale='auto', n_compo=rank, reg=reg)
        )
        X = filter_bank_transformer.fit_transform(X)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=1e4, random_state=RANDOM_STATE)
        )
        # model = Pipeline(steps=[
        #     ('filterbank', filter_bank_transformer),
        #     ('scaler', StandardScaler()),
        #     ('log_reg', LogisticRegression())]
        # )

    # Simple features from PSD
    elif benchmark == 'features-psd':
        features = h5io.read_hdf5(DERIV_ROOT / 'features_psd_features.h5')
        X = np.concatenate(
            [features[sub][None, :] for sub in train_subjects],
            axis=0
        )
        # model = Pipeline(steps=[
        #     ('scaler', StandardScaler()),
        #     ('log_reg', LogisticRegression())]
        # )
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1, max_iter=1e4)
        )

    # PSD features + covariances
    elif benchmark == 'psd-filterbank-riemann':
        features_psd = h5io.read_hdf5(DERIV_ROOT / 'features_features_psd.h5')
        X_psd = np.concatenate(
            [features_psd[sub][None, :] for sub in train_subjects],
            axis=0
        )
        reg_fb = 1e-7
        rank_fb = 120
        features_fb = h5io.read_hdf5(DERIV_ROOT / 'features_fb_covs.h5')
        covs_fb = [features_fb[sub]['covs'] for sub in train_subjects]
        covs_fb = np.array(covs_fb)
        X_fb = pd.DataFrame(
            {band: list(covs_fb[:, ii]) for ii, band in
             enumerate(frequency_bands)})
        filter_bank_transformer_fb = coffeine.make_filter_bank_transformer(
            names=list(frequency_bands),
            method='riemann',
            projection_params=dict(scale='auto', n_compo=rank_fb, reg=reg_fb)
        )
        X_fb = filter_bank_transformer_fb.fit_transform(X_fb)
        X = np.concatenate((X_psd, X_fb), axis=1)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=1e4)
        )
        # model = Pipeline(steps=[
        #     ('scaler', StandardScaler()),
        #     ('log_reg', LogisticRegression())]
        # )
    return X, y, model


def run_benchmark_cv(benchmark):
    X, y, model = load_data(benchmark=benchmark)
    cv = StratifiedShuffleSplit(
        n_splits=20,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    # # Grid search for Riemann pipeline
    # param_grid = {
    #     'log_reg__C': np.logspace(-2, 2, num=5),
    #     'log_reg__random_state': [RANDOM_STATE],
    #     'log_reg__max_iter': [1e4]
    # }
    # grid_cv = GridSearchCV(
    #     model, param_grid,
    #     scoring=make_scorer(accuracy_score),
    #     # scoring=make_scorer(balanced_accuracy_score),
    #     cv=cv, n_jobs=N_JOBS
    # )
    # grid_cv.fit(X, y)
    # scores = grid_cv.cv_results_
    # results = {
    #     'best_params': grid_cv.best_params_,
    #     'mean_test_score': scores['mean_test_score'],
    #     'std_test_score': scores['std_test_score'],
    #     'fit_time': scores['mean_fit_time'],
    #     'score_time': scores['mean_score_time'],
    #     'benchmark': benchmark
    # }

    # # Confusion matrices
    # conf_mats = []
    # for train_index, test_index in cv.split(X, y):
    #     if benchmark == 'dummy' or benchmark =='features-psd':
    #         X_train = X[train_index]
    #         X_test = X[test_index]
    #     else:
    #         X_train = X.iloc[train_index]
    #         X_test = X.iloc[test_index]
    #     y_train = [y[i] for i in train_index]
    #     y_test = [y[i] for i in test_index]
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
    #     conf_mat = confusion_matrix(y_test, y_pred)
    #     conf_mats.append(conf_mat[None, :])
    # conf_mat_mean = np.concatenate(conf_mats, axis=0).sum(axis=0)

    scoring = make_scorer(balanced_accuracy_score)
    # scoring = make_scorer(accuracy_score)

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

    return results


if __name__ == "__main__":
    # fig, axes = plt.subplots(1, len(BENCHMARKS))
    for b, benchmark in enumerate(BENCHMARKS):
        results_df = run_benchmark_cv(benchmark)
        print(results_df)
        # results_df = results_df/results_df.sum(axis=1, keepdims=True)
        if results_df is not None:
            results_df.to_csv(
                f"./results/benchmark-{benchmark}.csv")
            mean_accuracy = results_df['accuracy'].mean()
            std_accuracy = results_df['accuracy'].std()
            print(f'Mean accuracy of {benchmark} model: \n{mean_accuracy} +- {std_accuracy}')
#         axes[b].matshow(results_df)
#         for (i, j), z in np.ndenumerate(results_df):
#             axes[b].text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
#                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        
# plt.show()