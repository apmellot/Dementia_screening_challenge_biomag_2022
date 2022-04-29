import pathlib
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


DERIV_ROOT = pathlib.Path('/storage/store3/derivatives/biomag_hokuto_bids')
FEATURES_ROOT = DERIV_ROOT
BIDS_ROOT = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto_bids'
)
ROOT = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto'
)

import h5io

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
    # Simple features from PSD
    features = h5io.read_hdf5(FEATURES_ROOT / 'features_features_psd.h5')
    X_train = np.concatenate(
        [features[sub][None, :] for sub in train_subjects],
        axis=0
    )
    X_test = np.concatenate(
        [features[sub][None, :] for sub in test_subjects],
        axis=0
    )

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(3)
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return test_subjects, y_pred



if __name__ == "__main__":
    test_subjects, y_pred = run_model()
    results = pd.DataFrame(
        {
            'subjects': test_subjects,
            'predicted_labels': y_pred
        }
    )
    results.to_csv('./results/results.csv')