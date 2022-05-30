import mne
import os

from pathlib import Path

base_path = Path("/storage/store/data/biomag_challenge/Biomag2022")

path_fieldtrip = base_path / "biomag_hokuto_fieldtrip"
path_fif = base_path / "biomag_hokuto_fif"

os.mkdir(path_fif / "training")
os.mkdir(path_fif / "training" / "control")
os.mkdir(path_fif / "training" / "dementia")
os.mkdir(path_fif / "training" / "mci")
os.mkdir(path_fif / "test")


def create_raw(raw_fieldtrip):
    raw_copy = raw_fieldtrip.pick_types(meg=True)
    raw_copy.load_data()
    
    data = raw_copy.get_data()
    data *= 1e-15 # fT to T 

    ch_types = raw_copy.get_channel_types()
    ch_names = raw_copy.ch_names
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1000)

    raw_fif = mne.io.RawArray(data, info)
    return raw_fif


for subdir, dirs, files in os.walk(str(path_fieldtrip)):
    for file in files:
        if file.endswith(".mat"):
            raw_fieldtrip = mne.io.read_raw_fieldtrip(Path(subdir) / file, None)
            raw_fif = create_raw(raw_fieldtrip)
            new_path = subdir.replace("biomag_hokuto_fieldtrip", "biomag_hokuto_fif")
            new_path = new_path + '/' + file
            raw_fif.save(new_path.replace(".mat", ".fif"))
