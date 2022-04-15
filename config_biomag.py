import pathlib

bids_root = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto_bids')

deriv_root = pathlib.Path('/storage/store3/derivatives/biomag_hokuto_bids')


task = 'rest'
sessions = ['rest']  # keep empty for code flow
data_type = 'meg'
ch_types = ['meg']

l_freq = 0.1
h_freq = 49

reject = None

resample_sfreq = 200

epochs_tmin = 0.
epochs_tmax = 10.
rest_epochs_overlap = 0.
rest_epochs_duration = 10.
baseline = None

find_flat_channels_meg = False
find_noisy_channels_meg = False
use_maxwell_filter = False
run_source_estimation = False

random_state = 42

log_level = "info"

mne_log_level = "error"

# on_error = 'continue'
on_error = "debug"

N_JOBS = 20