from joblib import Parallel, delayed
import pandas as pd

import pathlib

import mne_bids
import mne
import coffeine
import h5io

DERIV_ROOT = pathlib.Path('/storage/store3/derivatives/biomag_hokuto_bids')
BIDS_ROOT = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto_bids'
)
FEATURE_TYPE = ['fb_covs']
N_JOBS = 10

frequency_bands = {
    "low": (0.1, 1),
    "delta": (1, 4),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 15.0),
    "beta_low": (15.0, 26.0),
    "beta_mid": (26.0, 35.0),
    "beta_high": (35.0, 49)
}


def extract_fb_covs(epochs, frequency_bands):
    features, meta_info = coffeine.compute_features(
        epochs, features=('covs',), n_fft=1024, n_overlap=512,
        fs=epochs.info['sfreq'], fmax=49, frequency_bands=frequency_bands)
    features['meta_info'] = meta_info
    return features


def run_subject(subject, feature_type):
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        root=DERIV_ROOT,
        datatype="meg",
        task="rest",
        session="rest",
        processing="clean",
        suffix="epo",
        extension=".fif",
        check=False
    )
    epochs = mne.read_epochs(bids_path)
    if feature_type == 'fb_covs':
        out = extract_fb_covs(epochs, frequency_bands)

    return out


if __name__ == "__main__":

    participants = pd.read_csv(BIDS_ROOT / "participants.tsv", sep="\t")
    subjects = participants.participant_id
    features = []
    for feature_type in FEATURE_TYPE:
        features = Parallel(n_jobs=N_JOBS)(
            delayed(run_subject)(subject[4:], feature_type)
            for subject in subjects
        )
        out = {sub: ff for sub, ff in zip(subjects, features)
               if not isinstance(ff, str)}
        out_fname = DERIV_ROOT / f'features_{feature_type}.h5'
        h5io.write_hdf5(
            out_fname,
            out,
            overwrite=True
        )
