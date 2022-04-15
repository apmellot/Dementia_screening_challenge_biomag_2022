import numpy as np
import pandas as pd

import pathlib

DERIV_ROOT = pathlib.Path('/storage/store3/derivatives/biomag_hokuto_bids')
BIDS_ROOT = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto_bids'
)

BENCHMARKS = ['dummy']

if __name__ == "__main__":
    
    participants = pd.read_csv(BIDS_ROOT / "participants.tsv", sep="\t")
    labels = []
    feature_type = "random"
    for subject in participants.participant_id:
        subject = subject[4:]
        if subject.find('control')==0:
            labels.append('control')
        if subject.find('mci')==0:
            labels.append('mci')
        if subject.find('dementia')==0:
            labels.append('dementia')
        if subject.find('test')==0:
            labels.append('test')
    features = np.load(DERIV_ROOT / f"features_{feature_type}.npy")
    