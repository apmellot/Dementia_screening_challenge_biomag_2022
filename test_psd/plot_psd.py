import numpy as np
import pandas as pd
import pathlib
import os.path as op
import matplotlib.pyplot as plt
import seaborn as sns

results_all = pd.read_csv('./average_psd.csv', index_col=0)
ROOT = pathlib.Path(
    '/storage/store/data/biomag_challenge/Biomag2022/biomag_hokuto'
)

def get_subjects_age(age, labels):
    subjects = []
    for label in labels:
        site_info = pd.read_excel(ROOT / 'hokuto_profile.xlsx', sheet_name=label)
        for i in range(site_info.shape[0]):
            if site_info['Age'].iloc[i]>= age:
                subjects.append(site_info['ID'].iloc[i][7:])
    print(len(subjects))
    return subjects


def get_site(labels, subjects):
    subjects_A = []
    subjects_B = []
    for label in labels:
        site_info = pd.read_excel(ROOT / 'hokuto_profile.xlsx', sheet_name=label)
        for i in range(site_info.shape[0]):
            subject = site_info['ID'].iloc[i][7:]
            print(subject)
            if site_info['Site'].iloc[i] == 'A' and subject in subjects:
                subjects_A.append(subject)
            if site_info['Site'].iloc[i] == 'B' and subject in subjects:
                subjects_B.append(subject)
    return subjects_A, subjects_B



# Par site
psds_A = []
psds_B = []
labels = results_all['label'].unique()
subjects_all = get_subjects_age(50, ['control', 'dementia', 'mci'])
subjects_A, subjects_B = get_site(labels, subjects_all)
for sub_A in subjects_A:
    results_sub = results_all.query(f"subject == '{sub_A}'")
    psds_A.append(list(results_sub['psd']))
for sub_B in subjects_B:
    results_sub = results_all.query(f"subject == '{sub_B}'")
    psds_B.append(list(results_sub['psd']))
psds_A = np.array(psds_A)
psds_B = np.array(psds_B)
q_psd_A = np.quantile(psds_A, q=[0.25, 0.5, 0.75], axis=0)
q_psd_B = np.quantile(psds_B, q=[0.25, 0.5, 0.75], axis=0)
freqs = results_sub['freqs']
plt.figure()
plt.plot(freqs, q_psd_A[1], 'r', label='site A')
plt.fill_between(freqs,
                 q_psd_A[0],
                 q_psd_A[2],
                 color='r',
                 alpha=0.5)
plt.plot(freqs, q_psd_B[1], 'b', label='site B')
plt.fill_between(freqs,
                 q_psd_B[0],
                 q_psd_B[2],
                 color='b',
                 alpha=0.5)

plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.xlabel("Frequency in Hz")
plt.ylabel("Power spectral density")
plt.title("PSD of all the subjects from the training set averaged on the sensors.")
plt.savefig('./average_psd_site.png')


# # Par classes
# psds_control = []
# psds_dementia = []
# psds_mci = []
# results_all = results_all[results_all['subject'].isin(subjects)]
# for label in labels:
#     results = results_all.query(f"label == '{label}'")
#     subjects = [sub for sub in results['subject'] if sub in subjects_all]
    # for subject in subjects:
    #     results_sub = results.query(f"subject == '{subject}'")
    #     if label == 'control':
    #         psds_control.append(list(results_sub['psd']))
    #     if label == 'mci':
    #         psds_mci.append(list(results_sub['psd']))
    #     if label == 'dementia':
    #         psds_dementia.append(list(results_sub['psd']))
# psds_control = np.array(psds_control)
# psds_dementia = np.array(psds_dementia)
# psds_mci = np.array(psds_mci)

# freqs = results_sub['freqs']
# q_psd_control = np.quantile(psds_control, q=[0.25, 0.5, 0.75], axis=0)
# q_psd_dementia = np.quantile(psds_dementia, q=[0.25, 0.5, 0.75], axis=0)
# q_psd_mci = np.quantile(psds_mci, q=[0.25, 0.5, 0.75], axis=0)

# plt.figure()
# plt.plot(freqs, q_psd_control[1], 'r', label='control')
# plt.fill_between(freqs,
#                 q_psd_control[0],
#                 q_psd_control[2],
#                 color='r',
#                 alpha=0.5)
# plt.plot(freqs, q_psd_mci[1], 'b', label='mci')
# plt.fill_between(freqs,
#                 q_psd_mci[0],
#                 q_psd_mci[2],
#                 color='b',
#                 alpha=0.5)
# plt.plot(freqs, q_psd_dementia[1], 'g', label='dementia')
# plt.fill_between(freqs,
#                 q_psd_dementia[0],
#                 q_psd_dementia[2],
#                 color='g',
#                 alpha=0.5)
# plt.xscale("log")
# plt.yscale("log")
# plt.legend()
# plt.xlabel("Frequency in Hz")
# plt.ylabel("Power spectral density")
# plt.title("PSD of all the subjects from the training set averaged on the sensors.")
# plt.show()
