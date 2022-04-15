import numpy as np
import mne
import os

from pathlib import Path

base_path = Path("/storage/store/data/biomag_challenge/Biomag2022")

converted_path = base_path / "biomag_hokuto_fieldtrip"
target_path = base_path / "biomag_hokuto_fif"

os.mkdir(target_path / "training")
os.mkdir(target_path / "training" / "control")
os.mkdir(target_path / "training" / "dementia")
os.mkdir(target_path / "training" / "mci")
os.mkdir(target_path / "test")


def create_raw(raw_data):
    raw_copy = raw_data.pick_types(meg=True)
    raw_copy.load_data()
    
    data = raw_copy.get_data()
    data *= 1e-15

    ch_types = raw_copy.get_channel_types()
    ch_names = raw_copy.ch_names
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1000)

    raw_meg = mne.io.RawArray(data, info)
    return raw_meg


for subdir, dirs, files in os.walk(str(converted_path)):
    for file in files:
        if file.endswith(".mat"):
            raw_data = mne.io.read_raw_fieldtrip(Path(subdir) / file, None)
            raw_meg = create_raw(raw_data)
            new_path = subdir.replace("biomag_hokuto_fieldtrip", "biomag_hokuto_fif")
            new_path = new_path + '/' + file
            raw_meg.save(new_path.replace(".mat", ".fif"))
