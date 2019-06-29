from scipy.io import loadmat, savemat
import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn import cross_validation
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD, adam, nadam, Adagrad
from keras.regularizers import l1,l2
from keras.callbacks import EarlyStopping, CSVLogger
from deep_pnas_py_src.models import build_nn_resnet

import os
import os.path
import sys
import argparse
import time
import csv

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def hcp_ten_pair_wise_reprod_check(model):
    # These directories contain fitted SH Coefficients of HCP subjects saved as .mat files
    # If this function is intended to be used then pass in paths as additional arguments or
    # modify the hardcoded paths.
    orig_dir = '/HCP_ten_subjects_orig_retest/Original/'
    retest_dir = '/HCP_ten_subjects_orig_retest/Retest/'

    subject_dir = ['103818', '105923', '111312', '114823', '115320', '122317', '125525', '130518', '139839', '143325',
                   '144226', '146129']

    for each in subject_dir:

        print(each)
        orig_file = each + "_reshaped_orig_sh_dwmri.mat"
        orig_mat_file = os.path.join(orig_dir, each, str('T1w'), str('Diffusion'), orig_file)

        retest_file = each + "_reshaped_retest_sh_dwmri.mat"
        retest_mat_file = os.path.join(retest_dir, each, str('T1w'), str('Diffusion'), retest_file)

        orig_mat = loadmat(orig_mat_file)
        retest_mat = loadmat(retest_mat_file)

        orig_np = np.array(orig_mat['sh_vol_re'])
        retest_np = np.array(retest_mat['sh_vol_re'])

        f1 = each + "dnn_orig_preds.mat"
        f2 = each + "dnn_retest_pred.mat"

        out_file_1 = os.path.join(orig_dir, f1)
        out_file_2 = os.path.join(retest_dir, f2)

        out_path_1 = os.path.dirname(out_file_1)
        if not os.path.exists(out_path_1):
            os.makedirs(out_path_1)

        out_path_2 = os.path.dirname(out_file_2)
        if not os.path.exists(out_path_2):
            os.makedirs(out_path_2)

        # Pred 3TA TS04
        pred = model.predict(orig_np)
        savemat(out_file_1, mdict={'predicted': pred})
        print('Original Saved')

        # Pred 3TB TS04
        predi = model.predict(retest_np)
        savemat(out_file_2, mdict={'predicted': predi})
        print('Retest Saved')

    print('All Done ..')
    return None

model_weights_path = r'model_weights.h5'
model_weights_path = os.path.normpath(model_weights_path)

model_D = build_nn_resnet()
print('Model Built Successfully ..')
model_D.load_weights(model_weights_path)
print('Model Weights loaded succesfully ..')

print ("Savine HCP Subject Predictions")
hcp_ten_pair_wise_reprod_check(model_D)