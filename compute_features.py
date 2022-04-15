import mne_bids
import mne

import numpy as np
import pandas as pd

import pathlib

DERIV_ROOT = pathlib.Path('/storage/store3/derivatives/biomag_hokuto_bids')
BIDS_ROOT = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto_bids'
)


def compute_dummy_features(epochs):
    return np.random.random(10)


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
    
    if feature_type == "random":
        out = compute_dummy_features(epochs)
        
    return out


if __name__ == "__main__":
    
    participants = pd.read_csv(BIDS_ROOT / "participants.tsv", sep="\t")
    features = []
    feature_type = "random"
    for subject in participants.participant_id:
        subject = subject[4:]
        features.append(run_subject(subject, feature_type)[None, :])
    features = np.concatenate(features, axis=0)
    np.save(DERIV_ROOT / f"features_{feature_type}.npy", features)
    