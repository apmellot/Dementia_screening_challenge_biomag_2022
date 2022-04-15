import os
import mne
import mne_bids

base_path = '/storage/store/data/biomag_challenge/Biomag2022'

input_dir = '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto_fif'
bids_root = '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto_bids'


def convert_to_bids(raw_path, bids_root, subject):
    raw = mne.io.read_raw(raw_path)
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        datatype='meg',
        session='rest',
        task='rest',
        root=bids_root,
        extension='.fif',
        suffix='meg'
    )

    mne_bids.write_raw_bids(raw=raw, bids_path=bids_path, overwrite=True)
    

for subdir, dirs, files in os.walk(str(base_path)):
    for file in files:
        if file.endswith(".fif"):
            subject = file[7 : -4]
            raw_path = subdir + '/' + file
            convert_to_bids(raw_path, bids_root, subject)