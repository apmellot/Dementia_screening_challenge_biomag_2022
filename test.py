# %%
import numpy as np
import pandas as pd

from pyriemann.embedding import LocallyLinearEmbedding
import matplotlib.pyplot as plt
import seaborn as sns

import pathlib
import h5io
import coffeine
import dameeg

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.metrics import make_scorer, accuracy_score

from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.base import invsqrtm, powm
from pyriemann.utils.distance import distance_riemann
from pyriemann.tangentspace import TangentSpace

# %%
DERIV_ROOT = pathlib.Path('/storage/store3/derivatives/biomag_hokuto_bids')
ROOT = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto'
)
N_JOBS = 5
RANDOM_STATE = 42

labels = ['control', 'mci', 'dementia']
subjects_A = []
subjects_B = []
for label in labels:
    site_info = pd.read_excel(ROOT / 'hokuto_profile.xlsx', sheet_name=label)
    for i in range(site_info.shape[0]):
        if site_info['Site'].iloc[i] == 'A':
            subjects_A.append('sub-' + site_info['ID'].iloc[i][7:])
        if site_info['Site'].iloc[i] == 'B':
            subjects_B.append('sub-' + site_info['ID'].iloc[i][7:])

frequency_bands = {
    "low": (0.1, 1),
    "delta": (1, 4),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 15.0),
    "beta_low": (15.0, 26.0),
    "beta_mid": (26.0, 35.0),
    "beta_high": (35.0, 49)
}


def get_labels(all_subjects):
    labels = []
    for subject in all_subjects:
        if subject.find('control') == 4:
            labels.append('control')
        elif subject.find('mci') == 4:
            labels.append('mci')
        elif subject.find('dementia') == 4:
            labels.append('dementia')
        elif subject.find('test') == 4:
            labels.append('test')
    return labels


features = h5io.read_hdf5(DERIV_ROOT / 'features_fb_covs.h5')

covs_A = [features[sub]['covs'] for sub in subjects_A]
covs_A = np.array(covs_A)
covs_B = [features[sub]['covs'] for sub in subjects_B]
covs_B = np.array(covs_B)
X_A = pd.DataFrame(
    {band: list(covs_A[:, ii]) for ii, band in
        enumerate(frequency_bands)})
X_B = pd.DataFrame(
    {band: list(covs_B[:, ii]) for ii, band in
        enumerate(frequency_bands)})

# %% Domain adaptation

rank = 65
X_At = pd.DataFrame()
X_Bt = pd.DataFrame()
for band in ['low']:
    spatial_filter = coffeine.ProjCommonSpace(n_compo=rank)
    spatial_filter.fit(X_A[band])
    X_A_filtered = spatial_filter.transform(X_A[band])
    X_A_filtered = np.array([X_A_filtered.iloc[i][0] for i in range(X_A_filtered.shape[0])])
    spatial_filter.fit(X_B[band])
    X_B_filtered = spatial_filter.transform(X_B[band])
    X_B_filtered = np.array([X_B_filtered.iloc[i][0] for i in range(X_B_filtered.shape[0])])
    # X_A_aligned, X_B_aligned = dameeg.align_procrustes(X_A_filtered, X_B_filtered)
    X_A_aligned, X_B_aligned = dameeg.recenter.align_recenter(X_A_filtered, X_B_filtered)
    X_A_aligned = pd.DataFrame({'cov': list(X_A_aligned)})
    X_B_aligned = pd.DataFrame({'cov': list(X_B_aligned)})
    covariance_transformer = coffeine.Riemann(metric='riemann')
    covariance_transformer.fit(X_A_aligned)
    X_A_vect = covariance_transformer.transform(X_A_aligned)
    covariance_transformer = coffeine.Riemann(metric='riemann')
    covariance_transformer.fit(X_B_aligned)
    X_B_vect = covariance_transformer.transform(X_B_aligned)
    if X_At.empty:
        X_At = X_A_vect
        X_Bt = X_B_vect
    else:
        X_At = pd.concat([X_At, X_A_vect], axis=1)
        X_Bt = pd.concat([X_Bt, X_B_vect], axis=1)

X = pd.concat([X_At, X_Bt], axis=0)
y = get_labels(subjects_A + subjects_B)
# %%


model = make_pipeline(StandardScaler(),
                      RandomForestClassifier(n_estimators=100,
                                             random_state=RANDOM_STATE))

cv = StratifiedShuffleSplit(n_splits=20, test_size=0.2,
                            random_state=RANDOM_STATE)
scoring = make_scorer(accuracy_score)

print("Running cross validation ...")
scores = cross_validate(
    model, X, y, cv=cv, scoring=scoring,
    n_jobs=N_JOBS)
print("... done.")

results = pd.DataFrame(
    {'accuracy': scores['test_score'],
        'fit_time': scores['fit_time'],
        'score_time': scores['score_time']}
)
# %%
X = np.concatenate((X_A_aligned, X_B_aligned))
# X = np.concatenate((X_A_filtered, X_B_filtered))
# X = np.concatenate((X_Af, X_Bf))

lapl = LocallyLinearEmbedding(n_components=2, metric='riemann')
embd = lapl.fit_transform(X)


df = pd.DataFrame({'subject': subjects_A + subjects_B,
                   'label': get_labels(subjects_A + subjects_B),
                   'x': embd[:, 0],
                   'y': embd[:, 1],
                   'site': ['A']*len(subjects_A) + ['B']*len(subjects_B)})

# %%
plt.figure()
sns.scatterplot(x='x', y='y', hue='label', style='site', data=df)
plt.title(f'Riemannian embedding of covariances in the {band} frequency band')
plt.show()

# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = pd.concat([X_At, X_Bt], axis=0)
X = pca.fit_transform(X)
y = get_labels(subjects_A + subjects_B)
df = pd.DataFrame({'subject': subjects_A + subjects_B,
                   'label': get_labels(subjects_A + subjects_B),
                   'x': X[:, 0],
                   'y': X[:, 1],
                   'site': ['A']*len(subjects_A) + ['B']*len(subjects_B)})
plt.figure()
sns.scatterplot(x='x', y='y', hue='label', style='site', data=df)
plt.title(f'Riemannian embedding of covariances in the {band} frequency band')
plt.show()
