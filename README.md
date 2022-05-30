# Challenge Biomag 2022: dementia screening

Code to reproduce results for the [Biomag challenge](https://biomag2020.org/awards/data-analysis-competitions/) on dementia screening, based on the Python library [MNE](https://mne.tools/stable/index.html) for MEG/EEG data processing.

## Convert the files

In order to use MNE, the files need to be converted to the Fieldtrip format. This requires the usage of the matlab library [SPM](https://www.fil.ion.ucl.ac.uk/spm/) to load the data with ```spm_eeg_load.m```, and the matlab library [Fieldtrip](https://www.fieldtriptoolbox.org/faq/how_to_select_the_correct_spm_toolbox/) to convert the data to Fieldtrip files with ```spm2fieldtrip```.

MNE provides tools to convert data written in the Fieldtrip format to .fif files, which contain raw data designed for MNE. To obtain such data, run the command:
~~~
python convert_fiedltrip_to_fif.py
~~~

## BIDS format and preprocessing

BIDS stands for Brain Imaging Data Structure and is a simple way to organize neuroimaging data. The aim of BIDS is to be a standard format to make the data accessible to everyone and avoid confusion. The [MNE-BIDS-pipeline](https://mne.tools/mne-bids-pipeline/index.html) is a processing pipeline for MEG and EEG data stored under the BIDS format. It includes several steps like preprocessing / epoching / evoked responses etc …

To convert the .fif files to BIDS, run the command:
~~~
python convert_to_bids.py
~~~

The preprocessing requires the usage of MNE-BIDS-pipeline, along with a configuration file ```config_biomag.py```.
We recommend downloading the MNE-BIDS-pipeline repository and placing it in the same folder this repository is downloaded, such that its relative position would be ../mne-bids-pipeline. 
See [installation instructions](https://mne.tools/mne-bids-pipeline/getting_started/install.html)

Run the following command to start the preprocessing:
~~~
python ../mne-bids-pipeline/run.py --config config_biomag.py --n_jobs 20 --steps=preprocessing
~~~
In our case, the data are resampled to 200Hz, bandpassed between 0.1Hz and 49Hz, and changed into epochs of 10 seconds.

## Feature engineering

To compute the features based on the PSD or the covariance, run the command:
~~~
python compute_features.py
~~~

## Training the model

To train the model, run the command:
~~~
python final_model.py
~~~



