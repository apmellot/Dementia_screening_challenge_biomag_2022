import os.path as op
import glob
import mne
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

data_path = '/home/amellot/Documents/these/biomag_competition/converted_files/training'
derivative_path = '/home/amellot/Documents/these/biomag_competition/derivatives'
labels = ['control', 'mci', 'dementia']
# labels = ['control']


def compute_mean_psd(raw_path, label):
    subject = raw_path.split('hokuto_')[1].split('.')[0]
    print(subject)
    raw = mne.io.read_raw_fieldtrip(raw_path, info=None)
    raw = raw.resample(sfreq=1000)
    psds, freqs = mne.time_frequency.psd_welch(raw, fmin=0.1, fmax=45, n_fft=1024, n_overlap=128)
    mean_psd = psds.mean(axis=0)
    output = pd.DataFrame(dict(subject=[subject]*freqs.shape[0],
                               label=[label]*freqs.shape[0],
                               psd=list(mean_psd),
                               freqs=list(freqs)))
    return output

results = pd.DataFrame()
for label in labels:
    fnames = glob.glob(
    op.join(data_path, label,
            "hokuto_*.mat"))
    output = Parallel(n_jobs=15)(
        delayed(compute_mean_psd)(raw_path, label)
        for raw_path in fnames
    )
    output = pd.concat(output)
    results = pd.concat([results, output])

results.to_csv(op.join(derivative_path, 'average_psd.csv'))