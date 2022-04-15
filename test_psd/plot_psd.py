import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
import seaborn as sns

derivative_path = '/home/amellot/Documents/these/biomag_competition/derivatives'
results_all = pd.read_csv(op.join(derivative_path, 'average_psd.csv'), index_col=0)

psds_control = []
psds_dementia = []
psds_mci = []
labels = results_all['label'].unique()
for label in labels:
    results = results_all.query(f"label == '{label}'")
    subjects = results['subject'].unique()
    for subject in subjects:
        results_sub = results.query(f"subject == '{subject}'")
        if label == 'control':
            psds_control.append(list(results_sub['psd']))
        if label == 'mci':
            psds_mci.append(list(results_sub['psd']))
        if label == 'dementia':
            psds_dementia.append(list(results_sub['psd']))
psds_control = np.array(psds_control)
psds_dementia = np.array(psds_dementia)
psds_mci = np.array(psds_mci)

freqs = results_sub['freqs']
q_psd_control = np.quantile(psds_control, q=[0.25, 0.5, 0.75], axis=0)
q_psd_dementia = np.quantile(psds_dementia, q=[0.25, 0.5, 0.75], axis=0)
q_psd_mci = np.quantile(psds_mci, q=[0.25, 0.5, 0.75], axis=0)

plt.figure()
plt.plot(freqs, q_psd_control[1], 'r', label='control')
plt.fill_between(freqs,
                q_psd_control[0],
                q_psd_control[2],
                color='r',
                alpha=0.5)
plt.plot(freqs, q_psd_mci[1], 'b', label='mci')
plt.fill_between(freqs,
                q_psd_mci[0],
                q_psd_mci[2],
                color='b',
                alpha=0.5)
plt.plot(freqs, q_psd_dementia[1], 'g', label='dementia')
plt.fill_between(freqs,
                q_psd_dementia[0],
                q_psd_dementia[2],
                color='g',
                alpha=0.5)
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.xlabel("Frequency in Hz")
plt.ylabel("Power spectral density")
plt.title("PSD of all the subjects from the training set averaged on the sensors.")
plt.show()
