from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
import numpy as np
import pathlib

import mne_bids
import mne
import coffeine
import h5io

DERIV_ROOT = pathlib.Path('/storage/store3/derivatives/biomag_hokuto_bids')
FEATURES_ROOT = DERIV_ROOT
BIDS_ROOT = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto_bids'
)
# FEATURE_TYPE = ['fb_covs']
FEATURE_TYPE = ['features_psd']
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


def get_slope(x, y):
    lm = LinearRegression()
    lm.fit(x, y)
    return lm.coef_


def extract_simple_features(epochs):
    psd, freqs = mne.time_frequency.psd_welch(epochs, fmin=0.1, fmax=49,
                                              n_fft=1024, n_overlap=128)
    window = (freqs >= 0.1) & (freqs <= 120)
    freqs = freqs[window]
    psd = psd[..., window]
    psd = np.log10(psd)
    log_freq = np.log10(freqs[:, None])

    # Slope low frequencies
    low_freq = (freqs >= 0.1) & (freqs <= 1.5)
    y_low = psd[..., low_freq]
    x_low = np.log10(freqs[low_freq])[:, None]

    X_1f_low = np.array(
        [get_slope(x_low, yy.T)[:, 0]
         for yy in y_low]
    ).mean(0)

    # Slope high frequencies
    gamma_low = (freqs >= 35) & (freqs <= 49.8)
    y_gamma_low = psd[..., gamma_low]
    x_gamma_low = np.log10(freqs[gamma_low])[:, None]

    X_1f_gamma = np.array(
        [get_slope(x_gamma_low, yy.T)[:, 0]
         for yy in y_gamma_low]
    ).mean(0)

    # Alpha peak
    psd_fit = psd.mean(axis=(0, 1))
    poly_freqs = PolynomialFeatures(degree=15).fit_transform(log_freq)
    lm = LinearRegression()
    lm.fit(poly_freqs, psd_fit)
    resid = psd_fit - lm.predict(poly_freqs)
    filt = ((freqs >= 6) & (freqs < 15))
    idx = resid[filt].argmax(0)
    peak = freqs[filt][idx]
    # psd_fit = np.log10(psd).mean(axis=(0))
    # print(psd_fit.shape)
    # poly_freqs = PolynomialFeatures(degree=15).fit_transform(log_freq)
    # peaks = np.zeros(psd_fit.shape[1])
    # for i in range(psd_fit.shape[1]):
    #     lm = LinearRegression()
    #     lm.fit(poly_freqs, psd_fit[:, i])
    #     resid = psd_fit[:, i] - lm.predict(poly_freqs)
    #     filt = ((freqs >= 6) & (freqs < 15))
    #     idx = resid[filt].argmax(0)
    #     peaks[i] = freqs[filt][idx]

    # Median power
    psd_sum = psd_fit.cumsum()
    idx = np.abs(psd_sum[-1] / 2 - psd_sum).argmin()
    freq_median = freqs[idx]

    # # Spectral entropy
    # entropy = np.sum(- psd.mean(axis=(0, 1)) * psd_fit)

    # out = np.concatenate([X_1f_low, X_1f_gamma, peak, freq_median], axis=None)
    out = np.concatenate([X_1f_low, X_1f_gamma, peak], axis=None)
    print(out.shape)
    return out


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
    elif feature_type == 'features_psd':
        out = extract_simple_features(epochs)

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
        out_fname = FEATURES_ROOT / f'features_{feature_type}.h5'
        h5io.write_hdf5(
            out_fname,
            out,
            overwrite=True
        )
