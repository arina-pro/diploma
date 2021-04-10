"""Script which starts from timeseries extracted on ABIDE. Timeseries
   can be downloaded from "https://osf.io/hc4md/download" (1.7GB).

   phenotypes: if not downloaded, it should be downloaded from
   https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv
   Before, please read the data usage agreements and related material at
   http://preprocessed-connectomes-project.org/abide/index.html

   Prediction task is named as column "DX_GROUP".

   The timeseries are pre-extracted using several atlases
   AAL, Harvard Oxford, BASC, Power, MODL on ABIDE rs-fMRI datasets.

   After downloading, each folder should appear with name of the atlas and
   sub-folders, if necessary. For example, using BASC atlas, we have extracted
   timeseries signals with networks and regions. Regions implies while
   applying post-processing method to extract the biggest connected networks
   into separate regions. For MODL, we have extracted timeseries with
   dimensions 64 and 128 components.

   Dimensions of each atlas:
       AAL - 116
       BASC - 122
       Power - 264
       Harvard Oxford (cortical and sub-cortical) - 118
       MODL - 64 and 128

   The timeseries extraction process was done using Nilearn
   (http://nilearn.github.io/).

   Note: To run this script Nilearn is required to be installed.
"""
import warnings
import os
from os.path import join
import numpy as np
import pandas as pd

from downloader import fetch_abide


def _get_paths(phenotypic, atlas, timeseries_dir):
    """
    """
    timeseries = []
    IDs_subject = []
    diagnosis = []
    subject_ids = phenotypic['SUB_ID']
    for index, subject_id in enumerate(subject_ids):
        this_pheno = phenotypic[phenotypic['SUB_ID'] == subject_id]
        this_timeseries = join(timeseries_dir, atlas,
                               str(subject_id) + '_timeseries.txt')
        if os.path.exists(this_timeseries):
            timeseries.append(np.loadtxt(this_timeseries))
            IDs_subject.append(subject_id)
            diagnosis.append(this_pheno['DX_GROUP'].values[0])
    return timeseries, diagnosis, IDs_subject


# Path to data directory where timeseries are downloaded. If not
# provided this script will automatically download timeseries in the
# current directory.

timeseries_dir = None

# If provided, then the directory should contain folders of each atlas name
if timeseries_dir is not None:
    if not os.path.exists(timeseries_dir):
        warnings.warn('The timeseries data directory you provided, could '
                      'not be located. Downloading in current directory.',
                      stacklevel=2)
        timeseries_dir = fetch_abide(data_dir='./ABIDE')
else:
    # Checks if there is such folder in current directory. Otherwise,
    # downloads in current directory
    timeseries_dir = './ABIDE'
    if not os.path.exists(timeseries_dir):
        timeseries_dir = fetch_abide(data_dir='./ABIDE')

# Path to data directory where predictions results should be saved.
predictions_dir = None

if predictions_dir is not None:
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
else:
    predictions_dir = './ABIDE/connectivity' # save to folder
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

atlases = ['MODL/64'] #['AAL', 'HarvardOxford', 'BASC/networks', 'BASC/regions', 'Power', 'MODL/64', 'MODL/128']

dimensions = {'AAL': 116,
              'HarvardOxford': 118,
              'BASC/networks': 122,
              'BASC/regions': 122,
              'Power': 264,
              'MODL/64': 64,
              'MODL/128': 128}

# prepare dictionary for saving results
columns = ['atlas', 'measure', 'classifier', 'scores', 'iter_shuffle_split',
           'dataset', 'covariance_estimator', 'dimensionality']
results = dict()
for column_name in columns:
    results.setdefault(column_name, [])

pheno_dir = 'Phenotypic_V1_0b_preprocessed1.csv'
phenotypic = pd.read_csv(pheno_dir)

# Connectomes per measure
from connectome_matrices import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
measures = ['tangent'] #'correlation', 'partial correlation',


for atlas in atlases:
    print("Running script: with atlas: {0}".format(atlas))
    timeseries, diagnosis, _ = _get_paths(phenotypic, atlas, timeseries_dir)

    _, classes = np.unique(diagnosis, return_inverse=True)
    for measure in measures:
            print("[Connectivity measure] kind='{0}'".format(measure))
            connections = ConnectivityMeasure(
                cov_estimator=LedoitWolf(assume_centered=True),
                kind=measure)
            conn_coefs = connections.fit_transform(timeseries)

            results['coefs'].append(conn_coefs)
            results['atlas'].append(atlas)
            results['measure'].append(measure)
            results['dataset'].append('ABIDE')
            results['dimensionality'].append(dimensions[atlas])
            results['covariance_estimator'].append('LedoitWolf')
    res = pd.DataFrame(results)
    # save classification scores per atlas
    this_atlas_dir = join(predictions_dir, atlas)
    if not os.path.exists(this_atlas_dir):
        os.makedirs(this_atlas_dir)
    res.to_csv(join(this_atlas_dir, 'abide_conn_coefs.csv'))
all_results = pd.DataFrame(results)
all_results.to_csv('all_conn_coefs.csv')
