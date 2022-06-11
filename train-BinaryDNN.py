# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           train-DNN.py
#  Author: Joshuha Thomas-Wilsker
#  Institute of High Energy Physics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Code to train deep neural network
# for HH->WWyy analysis.
# @Last Modified by:   Ram Krishna Sharma
# @Last Modified time: 2021-08-03
import os
import sys
# Next two files are to get rid of warning while traning on IHEP GPU from matplotlib
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from numpy.testing import assert_allclose
import pickle
from array import array
import time
import pandas
import pandas as pd
import optparse, json, argparse, math
from os import environ
import ROOT

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger

import shap
from root_numpy import root2array, tree2array
from plotting.plotter import plotter
# import pydotplus as pydot
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# exit()

# tf.debugging.set_log_device_placement(True)


seed = 7
np.random.seed(7)
rng = np.random.RandomState(31337)
CURRENT_DATETIME = datetime.now()

def GenerateGitPatchAndLog(logFileName,GitPatchName):
    #CMSSWDirPath = os.environ['CMSSW_BASE']
    #CMSSWRel = CMSSWDirPath.split("/")[-1]

    os.system('git diff > '+GitPatchName)

    outScript = open(logFileName,"w");
    #outScript.write('\nCMSSW Version used: '+CMSSWRel+'\n')
    #outScript.write('\nCurrent directory path: '+CMSSWDirPath+'\n')
    outScript.close()

    os.system('echo -e "\n\n============\n== Latest commit summary \n\n" >> '+logFileName )
    os.system("git log -1 --pretty=tformat:' Commit: %h %n Date: %ad %n Relative time: %ar %n Commit Message: %s' >> "+logFileName )
    os.system('echo -e "\n\n============\n" >> '+logFileName )
    os.system('git log -1 --format="SHA: %H" >> '+logFileName )

def load_data_from_EOS(self, directory, mask='', prepend='root://eosuser.cern.ch'):
    eos_dir = '/eos/user/%s ' % (directory)
    eos_cmd = 'eos ' + prepend + ' ls ' + eos_dir
    print(eos_cmd)
    #out = commands.getoutput(eos_cmd)
    return

def load_data(inputPath,variables,criteria):
    """
    Load data from .root file into a pandas dataframe and return it.

    :param      inputPath:  Path of input root files
    :type       inputPath:  String
    :param      variables:  List of all input variables that need to read from input root files
    :type       variables:  list
    :param      criteria:   Selection cuts
    :type       criteria:   String
    """
    my_cols_list=variables
    print ("Variable list: ",my_cols_list)
    print ("Variable list[-5]: ",my_cols_list[:-5])
    print ("Variable list[-6]: ",my_cols_list[:-6])
    data = pd.DataFrame(columns=my_cols_list)
    keys=['HH','bckg']
    for key in keys :
        print('key: ', key)
        if 'HH' in key:
            sampleNames=key
            subdir_name = 'Signal'
            fileNames = [
            'output_sig125'
            ]
            target=1
        else:
            sampleNames = key
            subdir_name = 'Backgrounds'
            # contain two trees pp&pf+ff
            fileNames = [
            'New_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt',
            'New_pp'
            ]
            target=0

        for filen in fileNames:
            if 'output_sig125' in filen:
                treename=['tagsDumper/trees/ggh_125_13TeV_UntaggedTag']
                process_ID = "ggh"
            # consider pp & pf+ff to be different bkgs
            elif 'New_DataDriven_QCD_SFs_sEoEWgt_2DpTWgt' in filen:
                treename=['DataDriven_QCD',
                ]
                process_ID = 'QCD_pfff'
            elif 'New_pp' in filen:
                treename=['pp',
                ]
                process_ID = 'QCD_pp'


            fileName = os.path.join(subdir_name,filen)
            filename_fullpath = inputPath+"/"+fileName+".root"
            print("Input file: ", filename_fullpath)
            tfile = ROOT.TFile(filename_fullpath)
            if 'HH' in key:
                for tname in treename:
                    ch_0 = tfile.Get(tname)
                    if ch_0 is not None :
                        criteria_tmp = criteria
                        #if process_ID == "HH": criteria_tmp = criteria + " && (event%2!=0)"
                        # Create dataframe for ttree
                        chunk_arr = tree2array(tree=ch_0, branches=my_cols_list[0:11], selection=criteria_tmp) #attention: get all the variable including weight and norm_sfs
                        #chunk_arr = tree2array(tree=ch_0, branches=my_cols_list[:-5], selection=criteria, start=0, stop=500)
                        # This dataframe will be a chunk of the final total dataframe used in training
                        chunk_df = pd.DataFrame(chunk_arr, columns=my_cols_list[0:10]) # attention use first 10 as the training variable in the input_variable json file ,the last two are about the weight
                        # Add values for the process defined columns.
                        # (i.e. the values that do not change for a given process).
                        chunk_df['key']=key
                        chunk_df['target']=target
                        chunk_df['weight']=chunk_arr['weight']*(chunk_arr['vtxprob']*1./chunk_arr['sigmarv']+(1-chunk_arr['vtxprob'])*1./chunk_arr['sigmawv'])
                        # chunk_df['weight_NLO_SM']=chunk_df['weight_NLO_SM']
                        chunk_df['weight_NLO_SM']=1.0
                        chunk_df['process_ID']=process_ID
                        chunk_df['classweight']=1.0
                        chunk_df['unweighted'] = 1.0
                        # Append this chunk to the 'total' dataframe
                        data = data.append(chunk_df, ignore_index=True)
                    else:
                        print("TTree == None")
                    ch_0.Delete()
            else:
                for tname in treename:
                    ch_0 = tfile.Get(tname)
                    if ch_0 is not None :
                        criteria_tmp = criteria
                        #if process_ID == "HH": criteria_tmp = criteria + " && (event%2!=0)"
                        # Create dataframe for ttree
                        chunk_arr = tree2array(tree=ch_0, branches=my_cols_list[::], selection=criteria_tmp)
                        #chunk_arr = tree2array(tree=ch_0, branches=my_cols_list[:-5], selection=criteria, start=0, stop=500)
                        # This dataframe will be a chunk of the final total dataframe used in training
                        chunk_df = pd.DataFrame(chunk_arr, columns=my_cols_list[0:10])
                        # Add values for the process defined columns.
                        # (i.e. the values that do not change for a given process).
                        chunk_df['key']=key
                        chunk_df['target']=target
                        chunk_df['weight']=chunk_arr["weight"] * chunk_arr['Norm_SFs']
                        chunk_df['weight_NLO_SM']=1.0
                        chunk_df['process_ID']=process_ID
                        chunk_df['classweight']=1.0
                        chunk_df['unweighted'] = 1.0
                        # Append this chunk to the 'total' dataframe
                        data = data.append(chunk_df, ignore_index=True)
                    else:
                        print("TTree == None")
                    ch_0.Delete()
            tfile.Close()
        if len(data) == 0 : continue

    return data

def load_trained_model(model_path):
    print('<load_trained_model> weights_path: ', model_path)
    model = load_model(model_path, compile=False)
    return model

def custom_LearningRate_schedular(epoch,lr):
    if epoch < 10:
        return 0.01
    else:
        # return 0.1 * tf.math.exp(0.1 * (10 - epoch))
        return 0.01 * tf.math.exp(0.05 * (10 - epoch))

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def ANN_model(
                   num_variables,
                   optimizer='Nadam',
                   activation='relu',
                   loss='binary_crossentropy',
                   dropout_rate=0.2,
                   init_mode='glorot_normal',
                   learn_rate=0.001,
                   metrics=METRICS
                   ):
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = Sequential()
    # model.add(Dense(num_variables,input_dim=num_variables,kernel_initializer=init_mode,activation=activation))
    model.add(Dense(num_variables,kernel_initializer=init_mode,activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    if optimizer=='Adam':
        model.compile(loss=loss,optimizer=Adam(lr=learn_rate),metrics=metrics)
    if optimizer=='Nadam':
        model.compile(loss=loss,optimizer=Nadam(lr=learn_rate),metrics=metrics)
    if optimizer=='Adamax':
        model.compile(loss=loss,optimizer=Adamax(lr=learn_rate),metrics=metrics)
    if optimizer=='Adadelta':
        model.compile(loss=loss,optimizer=Adadelta(lr=learn_rate),metrics=metrics)
    if optimizer=='Adagrad':
        model.compile(loss=loss,optimizer=Adagrad(lr=learn_rate),metrics=metrics)
    return model

def baseline_model(
                   num_variables,
                   optimizer='Nadam',
                   activation='relu',
                   loss='binary_crossentropy',
                   dropout_rate=0.2,
                   init_mode='glorot_normal',
                   learn_rate=0.001,
                   metrics=METRICS
                   ):
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = Sequential()
    model.add(Dense(70,input_dim=num_variables,kernel_initializer=init_mode,activation=activation))
    # model.add(Dropout(dropout_rate))
    model.add(Dense(35,activation=activation))
    # model.add(Dropout(dropout_rate))
    model.add(Dense(10,activation=activation))
    model.add(Dense(4,activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    if optimizer=='Adam':
        model.compile(loss=loss,optimizer=Adam(lr=learn_rate),metrics=metrics)
    if optimizer=='Nadam':
        model.compile(loss=loss,optimizer=Nadam(lr=learn_rate),metrics=metrics)
    if optimizer=='Adamax':
        model.compile(loss=loss,optimizer=Adamax(lr=learn_rate),metrics=metrics)
    if optimizer=='Adadelta':
        model.compile(loss=loss,optimizer=Adadelta(lr=learn_rate),metrics=metrics)
    if optimizer=='Adagrad':
        model.compile(loss=loss,optimizer=Adagrad(lr=learn_rate),metrics=metrics)
    return model

def baseline_model2(
                   num_variables,
                   optimizer='Nadam',
                   activation='relu',
                   loss='binary_crossentropy',
                   dropout_rate=0.2,
                   init_mode='glorot_normal',
                   learn_rate=0.001,
                   metrics=METRICS
                   ):
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = Sequential()
    model.add(Dense(10,input_dim=num_variables,kernel_initializer=init_mode,activation=activation))
    # model.add(Dropout(dropout_rate))
    model.add(Dense(10,activation=activation))
    model.add(Dense(4,activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    if optimizer=='Adam':
        model.compile(loss=loss,optimizer=Adam(lr=learn_rate),metrics=metrics)
    if optimizer=='Nadam':
        model.compile(loss=loss,optimizer=Nadam(lr=learn_rate),metrics=metrics)
    if optimizer=='Adamax':
        model.compile(loss=loss,optimizer=Adamax(lr=learn_rate),metrics=metrics)
    if optimizer=='Adadelta':
        model.compile(loss=loss,optimizer=Adadelta(lr=learn_rate),metrics=metrics)
    if optimizer=='Adagrad':
        model.compile(loss=loss,optimizer=Adagrad(lr=learn_rate),metrics=metrics)
    return model

def baseline_modelScan(
                   num_variables,
                   optimizer='Nadam',
                   activation='relu',
                   loss='binary_crossentropy',
                   dropout_rate=0.2,
                   init_mode='glorot_normal',
                   learn_rate=0.001,
                   metrics=METRICS,
                   nHiddenLayer = 1,
                   dropoutLayer = 0
                   ):
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = Sequential()
    # neuronsFirstHiddenLayer = (((num_variables+1)*2)/3)
    neuronsHiddenLayer = []
    neuronsInputLayer = num_variables
    for x in range(0,10):
        neuronsHiddenLayer.append((((neuronsInputLayer+1)*2)/3))
        neuronsInputLayer = neuronsHiddenLayer[x]
    if (nHiddenLayer>=1):
        model.add(Dense(neuronsHiddenLayer[0],input_dim=num_variables,kernel_initializer=init_mode,activation=activation))
    if (dropoutLayer):
        model.add(Dropout(dropout_rate))
    if (nHiddenLayer>=2):
        model.add(Dense(neuronsHiddenLayer[1],activation=activation))
    if (dropoutLayer):
        model.add(Dropout(dropout_rate))
    if (nHiddenLayer>=3):
        model.add(Dense(neuronsHiddenLayer[2],activation=activation))
    if (nHiddenLayer>=4):
        model.add(Dense(neuronsHiddenLayer[3],activation=activation))
    if (nHiddenLayer>=5):
        model.add(Dense(neuronsHiddenLayer[4],activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    if optimizer=='Adam':
        model.compile(loss=loss,optimizer=Adam(lr=learn_rate),metrics=metrics)
    if optimizer=='Nadam':
        model.compile(loss=loss,optimizer=Nadam(lr=learn_rate),metrics=metrics)
    if optimizer=='Adamax':
        model.compile(loss=loss,optimizer=Adamax(lr=learn_rate),metrics=metrics)
    if optimizer=='Adadelta':
        model.compile(loss=loss,optimizer=Adadelta(lr=learn_rate),metrics=metrics)
    if optimizer=='Adagrad':
        model.compile(loss=loss,optimizer=Adagrad(lr=learn_rate),metrics=metrics)
    return model

def gscv_model(
                num_variables=35,
                optimizer="Nadam",
                activation='relu',
                init_mode='glorot_normal',
                learn_rate=0.01,
                neurons=10,
                metrics=METRICS
                ):
    model = Sequential()
    model.add(Dense(neurons,input_dim=num_variables,kernel_initializer=init_mode,activation=activation))
    model.add(Dense(10,activation=activation))
    model.add(Dense(4,activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    if optimizer=='Adam':
        model.compile(loss='binary_crossentropy',optimizer=Adam(lr=learn_rate),metrics=metrics)
    if optimizer=='Nadam':
        model.compile(loss='binary_crossentropy',optimizer=Nadam(lr=learn_rate),metrics=metrics)
    if optimizer=='Adamax':
        model.compile(loss='binary_crossentropy',optimizer=Adamax(lr=learn_rate),metrics=metrics)
    if optimizer=='Adadelta':
        model.compile(loss='binary_crossentropy',optimizer=Adadelta(lr=learn_rate),metrics=metrics)
    if optimizer=='Adagrad':
        model.compile(loss='binary_crossentropy',optimizer=Adagrad(lr=learn_rate),metrics=metrics)
    return model

def SL_MultiClassModel(
               num_variables,
               optimizer='Nadam',
               activation='relu',
               loss='binary_crossentropy',
               dropout_rate=0.2,
               init_mode='glorot_normal',
               learn_rate=0.001,
               metrics=METRICS
               ):
    model = Sequential()
    model.add(Dense(64, input_dim=num_variables,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(32,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(16,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(8,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, activation="sigmoid"))
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=metrics)
    return model

def new_model(
               num_variables,
               optimizer='Nadam',
               activation='relu',
               loss='binary_crossentropy',
               dropout_rate=0.2,
               init_mode='glorot_normal',
               learn_rate=0.001,
               metrics=METRICS
               ):
    model = Sequential()
    model.add(Dense(10, input_dim=num_variables,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dense(16,kernel_regularizer=regularizers.l2(0.01)))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, activation="sigmoid"))
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=metrics)
    return model

def new_model2(
               num_variables,
               optimizer='Nadam',
               activation='relu',
               loss='binary_crossentropy',
               dropout_rate=0.2,
               init_mode='glorot_normal',
               learn_rate=0.001,
               metrics=METRICS
               ):
    model = Sequential()
    model.add(Dense(20, input_dim=num_variables,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dense(16,kernel_regularizer=regularizers.l2(0.01)))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Dense(14))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(7))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, activation="sigmoid"))
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=metrics)
    return model

def new_model3(
               num_variables,
               optimizer='Nadam',
               activation='relu',
               loss='binary_crossentropy',
               dropout_rate=0.2,
               init_mode='glorot_normal',
               learn_rate=0.001,
               metrics=METRICS
               ):
    model = Sequential()
    model.add(Dense(20, input_dim=num_variables,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dense(16,kernel_regularizer=regularizers.l2(0.01)))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, activation="sigmoid"))
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=metrics)
    return model

def new_model5(
               num_variables,
               optimizer='Nadam',
               activation='relu',
               loss='binary_crossentropy',
               dropout_rate=0.2,
               init_mode='glorot_normal',
               learn_rate=0.001,
               metrics=METRICS
               ):
    model = Sequential()
    model.add(Dense(256, input_dim=num_variables,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dense(1, activation="sigmoid"))
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
    return model

def new_model6(
               num_variables,
               optimizer='Nadam',
               activation='relu',
               loss='binary_crossentropy',
               dropout_rate=0.2,
               init_mode='glorot_normal',
               learn_rate=0.001,
               metrics=METRICS
               ):
    model = Sequential()
    model.add(Dense(256, input_dim=num_variables,kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64))
    model.add(Activation(activation))
    model.add(Dense(1, activation="sigmoid"))
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
    return model

def new_model7(
               num_variables,
               optimizer='Nadam',
               activation='relu',
               loss='binary_crossentropy',
               dropout_rate=0.2,
               init_mode='glorot_normal',
               learn_rate=0.001,
               metrics=METRICS
               ):
    model = Sequential()
    model.add(Dense(256, input_dim=num_variables,kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation(activation))
    model.add(Dense(128))
    model.add(Activation(activation))
    model.add(Dense(128))
    model.add(Activation(activation))
    model.add(Dense(64))
    model.add(Activation(activation))
    model.add(Dense(1, activation="sigmoid"))
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
    return model

def check_dir(dir):
    if not os.path.exists(dir):
        print('mkdir: ', dir)
        os.makedirs(dir)
        os.system("cp dnn_parameter.json "+dir)

def main():
    print('Using Keras version: ', keras.__version__)

    usage = 'usage: %prog [options]'
    parent_parser = argparse.ArgumentParser(usage)
    parent_parser.add_argument('-l', '--load_dataset', dest='load_dataset', help='Option to load dataset from root file (0=False, 1=True)', default=False, type=bool)
    parent_parser.add_argument('-t', '--train_model', dest='train_model', help='Option to train model or simply make diagnostic plots (0=False, 1=True)', default=True, type=bool)
    parent_parser.add_argument('-s', '--suff', dest='suffix', help='Option to choose suffix for training', default='TEST', type=str)
    parent_parser.add_argument('-i', '--inputs_file_path', dest='inputs_file_path', help='Path to directory containing directories \'Bkgs\' and \'Signal\' which contain background and signal ntuples respectively.', default='', type=str)
    parent_parser.add_argument('-w', '--weights', dest='weights', help='weights to use', default='BalanceYields', type=str,choices=['BalanceYields','BalanceNonWeighted'])
    parent_parser.add_argument('-cw', '--classweight', dest='classweight', help='classweight to use', default=False, type=bool)
    parent_parser.add_argument('-sw', '--sampleweight', dest='sampleweight', help='sampleweight to use', default=False, type=bool)
    parent_parser.add_argument('-j', '--json', dest='json', help='input variable json file', default='input_variables.json', type=str)

    parent_parser.add_argument("-ModelToUse", "--ModelToUse", type=str, default="FH_ANv5", help = "Name of optimizer to train with")
    parent_parser.add_argument('-dlr', '--dynamic_lr', dest='dynamic_lr', help='vary learn rate with epoch', default=False, type=bool)
    parent_parser.add_argument("-e", "--epochs", type=int, default=200, help = "Number of epochs to train")
    parent_parser.add_argument("-b", "--batch_size", type=int, default=100, help = "Number of batch_size to train")
    parent_parser.add_argument("-o", "--optimizer", type=str, default="Nadam", help = "Name of optimizer to train with")
    parent_parser.add_argument("-a", "--activation", type=str, default="relu", help = "activation to be used. default is the relu")
    parent_parser.add_argument("-d", "--dropout_rate", type=float, default=0.2, help = "dropout rate to be used. Default value is 0.2")
    parent_parser.add_argument('-lr', '--lr', dest='learnRate', help='Learn rate', default=0.1, type=float)

    parent_parser.add_argument('-p', '--para', dest='hyp_param_scan', help='Option to run hyper-parameter scan', default=False, type=bool)
    parent_parser.add_argument('-g', '--GridSearch', dest='GridSearch', help='Option to train model or simply make diagnostic plots (0=False, 1=True)', default=False, type=bool)
    parent_parser.add_argument('-r', '--RandomSearch', dest='RandomSearch', help='Option to train model or simply make diagnostic plots (0=False, 1=True)', default=True, type=bool)

    parent_parser.add_argument("-nHiddenLayer", "--nHiddenLayer", type=int, default=1, help = "Number of Hidden layers")
    parent_parser.add_argument("-dropoutLayer", "--dropoutLayer", type=int, default=0, help = "If you want to include dropoutLayer with the first two hidden layers")



    args = parent_parser.parse_args()

    print('#---------------------------------------')
    print('#    Print all input arguments         #')
    print('#---------------------------------------')
    print('load_dataset     = %s'%args.load_dataset)
    print('train_model      = %s'%args.train_model)
    print('suffix           = %s'%args.suffix)
    print('inputs_file_path = %s'%args.inputs_file_path)
    print('weights          = %s'%args.weights)
    print('classweight      = %s'%args.classweight)
    print('sampleweight     = %s'%args.sampleweight)
    print('Input Var json   = %s'%args.json)
    print('')
    print('dynamic LearnRate= %s'%args.dynamic_lr)
    print('Learn rate       = %s'%args.learnRate)
    print('epochs           = %s'%args.epochs)
    print('batch_size       = %s'%args.batch_size)
    print('optimizer        = %s'%args.optimizer)
    print('activation       = %s'%args.activation)
    print('dropout_rate     = %s'%args.dropout_rate)
    print('')
    print('hyp_param_scan   = %s'%args.hyp_param_scan)
    print('GridSearch       = %s'%args.GridSearch)
    print('RandomSearch     = %s'%args.RandomSearch)
    print('')
    print('nHiddenLayer     = %s'%args.nHiddenLayer)
    print('dropoutLayer     = %s'%args.dropoutLayer)
    print('#---------------------------------------')


    do_model_fit = args.train_model
    suffix = args.suffix

    # Create instance of the input files directory
    # inputs_file_path = 'HHWWgg_DataSignalMCnTuples/2017/'
    # SL Lxplus = '/eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/'
    # FH Lxplus = '/eos/user/r/rasharma/post_doc_ihep/double-higgs/ntuples/January_2021_Production/DNN_MoreVar_v2/'
    # FH IHEP = '/hpcfs/bes/mlgpu/sharma/ML_GPU/Samples/DNN_MoreVar_v2/'
    inputs_file_path = args.inputs_file_path

    hyp_param_scan=args.hyp_param_scan
    # Set model hyper-parameters
    weights         = args.weights
    optimizer       = args.optimizer
    activation      = args.activation
    dropout_rate    = args.dropout_rate
    validation_split= 0.1
    GridSearch      = args.GridSearch
    RandomSearch    = args.RandomSearch

    # Create instance of output directory where all results are saved.
    output_directory = 'HHWWyyDNN_binary_%s_%s/' % (suffix,weights)
    check_dir(output_directory)

    # hyper-parameter scan results
    if weights == 'BalanceNonWeighted':
        learn_rate  = args.learnRate
        epochs      = args.epochs
        batch_size  = args.batch_size
        optimizer   = args.optimizer
    if weights == 'BalanceYields':
        learn_rate  = args.learnRate
        epochs      = args.epochs
        batch_size  = args.batch_size
        optimizer   = args.optimizer

    print('#---------------------------------------')
    print("Input DNN parameters:")
    print("\tepochs: ",epochs)
    print("\tbatch_size: ",batch_size)
    print("\tlearn_rate: ",learn_rate)
    print("\toptimizer: ",optimizer)
    print('#---------------------------------------')

    """
    Before we start save git patch. This will be helpful in debug the code later or taking care of the differences between many traning directory.
    """
    LogdirName= "gitLog_"+(str(CURRENT_DATETIME.year)[-2:]
              +str(format(CURRENT_DATETIME.month,'02d'))
              +str(format(CURRENT_DATETIME.day,'02d'))
              +"_"
              +str(format(CURRENT_DATETIME.hour,'02d'))
              +str(format(CURRENT_DATETIME.minute,'02d'))
              +str(format(CURRENT_DATETIME.second,'02d'))
              )
    GenerateGitPatchAndLog(LogdirName+".log",LogdirName+".patch")
    os.system('mv '+LogdirName+".log "+LogdirName+".patch "+output_directory)

    hyperparam_file = os.path.join(output_directory,'additional_model_hyper_params.txt')
    additional_hyperparams = open(hyperparam_file,'w')
    additional_hyperparams.write("optimizer: "+optimizer+"\n")
    additional_hyperparams.write("learn_rate: "+str(learn_rate)+"\n")
    additional_hyperparams.write("epochs: "+str(epochs)+"\n")
    additional_hyperparams.write("validation_split: "+str(validation_split)+"\n")
    additional_hyperparams.write("weights: "+weights+"\n")
    # Create plots subdirectory
    plots_dir = os.path.join(output_directory,'plots/')
    input_var_jsonFile = open(args.json,'r')
    # selection_criteria = '( (Leading_Photon_pt/CMS_hgg_mass) > 1/3. && (Subleading_Photon_pt/CMS_hgg_mass) > 1/4. && Leading_Photon_MVA>-0.7 && Subleading_Photon_MVA>-0.7 && SumTwoMaxBjets<0.6186)'
    # selection_criteria = '( (Leading_Photon_pt/CMS_hgg_mass) > 1/3. && (Subleading_Photon_pt/CMS_hgg_mass) > 1/4. && Leading_Photon_MVA>-0.7 && Subleading_Photon_MVA>-0.7)'
    # selection_criteria = '( (leadptom) > 1/3. && (subleadptom) > 1/4. && leadmva>-0.9 && subleadmva>-0.9)'
    selection_criteria = ''
    # selection_criteria = '( (Leading_Photon_pt/CMS_hgg_mass) > 1/3. && (Subleading_Photon_pt/CMS_hgg_mass) > 1/4. && Leading_Photon_MVA>-0.7 && Subleading_Photon_MVA>-0.7 && New_pTBasedSel_WW_mass < 200)'

    # Load Variables from .json
    variable_list = json.load(input_var_jsonFile,encoding="utf-8").items()

    # Earlier this framework is keeping .csv file in the output directory of
    # a particular run. But this takes lots of time in reading the same data
    # again and again. So, now I am sending the .csv file into the directory named
    # with the same name as the json file.
    #
    # NOTE (IMP): If the list of variable changes then change the name of
    # json file. Else it will read the old list of input variables. If you
    # want to keep the same name of json file then remove the directory that
    # corresponds to this name (it it exits).
    #
    # TO-DO: Make the code intelligent so that it will first check if the
    # input variables is exactly same as the one that already exits. If yes,
    # then delete the old entry and create a new one.
    CSV_file_Dir_Name = (args.json).replace(".json","")
    if not os.path.isdir(CSV_file_Dir_Name): os.mkdir(CSV_file_Dir_Name)
    os.system('cp '+args.json +' '+CSV_file_Dir_Name+"/") # also copy the json file to the new created directory

    # Create list of headers for dataset .csv
    column_headers = []
    for key,var in variable_list:
        column_headers.append(key)
    column_headers.append('weight')
    column_headers.append('weight_NLO_SM')
    column_headers.append('unweighted')
    column_headers.append('target')
    column_headers.append('key')
    column_headers.append('classweight')
    column_headers.append('process_ID')

    # Load ttree into .csv including all variables listed in column_headers
    print('<train-DNN> Input file path: ', inputs_file_path)
    # outputdataframe_name = '%s/output_dataframe.csv' %(output_directory)
    outputdataframe_name = '%s/output_dataframe.csv' %(CSV_file_Dir_Name)
    if os.path.isfile(outputdataframe_name) and (args.load_dataset == 0):
        """Load dataset or not

        If one changes the input training variables then we have to reload dataset.
        Don't use the previous .csv file if you update the list of input variables.
        """
        data = pandas.read_csv(outputdataframe_name)
        print('<train-DNN> Loading data .csv from: %s . . . . ' % (outputdataframe_name))
    else:
        print('<train-DNN> Creating new data .csv @: %s . . . . ' % (inputs_file_path))
        data = load_data(inputs_file_path,column_headers,selection_criteria)
        # Change sentinal value to speed up training.
        # data = data.mask(data<-25., -9.)
        # data[data<-25] = -9.0
        # data = data.replace(to_replace=-99.,value=-9.0)
        num = data._get_numeric_data()
        num[num < -25.] = -9.0
        data.to_csv(outputdataframe_name, index=False)
        data = pandas.read_csv(outputdataframe_name)

    print('#---------------------------------------')
    print('#    Print pandas dataframe            #')
    print('#---------------------------------------')
    print(data.head())
    print('#---------------------------------------')
    print('#---------------------------------------')
    print('#    describe pandas dataframe         #')
    print('#---------------------------------------')
    print(data.describe())
    print('#---------------------------------------')
    n = len(data)
    nHH = len(data.iloc[data.target.values == 1])
    nbckg = len(data.iloc[data.target.values == 0])
    print("Total (train+validation) length of HH = %i, bckg = %i" % (nHH, nbckg))
    bkg, sig = np.bincount(data['target'])
    total = bkg + sig
    print('Raw events:\n    Total: {}\n    Signal: {} ({:.2f}% of total)\n    Background: {} ({:.2f}% of total)\n'.format(
    total, sig, 100 * sig / total, bkg, 100 * bkg / total))
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / bkg)*(total)/2.0
    weight_for_1 = (1 / sig)*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class background: {:.2f}'.format(weight_for_0))
    print('Weight for class signal: {:.2f}'.format(weight_for_1))
    print('#---------------------------------------')
    print('<main> data columns: ', (data.columns.values.tolist()))
    print('#---------------------------------------')

    # Make instance of plotter tool
    Plotter = plotter()
    # Create statistically independant training/testing data
    traindataset, valdataset = train_test_split(data, test_size=0.1)
    valdataset.to_csv((output_directory+'valid_dataset.csv'), index=False)

    print('<train-DNN> Training dataset shape: ', traindataset.shape)
    print('<train-DNN> Validation dataset shape: ', valdataset.shape)

    # Event weights
    weights_for_HH = traindataset.loc[traindataset['process_ID']=='ggh', 'weight']
    # weights_for_HH_NLO = traindataset.loc[traindataset['process_ID']=='HH', 'weight_NLO_SM']
    # weights_for_Hgg = traindataset.loc[traindataset['process_ID']=='Hgg', 'weight']
    # weights_for_DiPhoton = traindataset.loc[traindataset['process_ID']=='DiPhoton', 'weight']
    weights_for_QCD_pfff = traindataset.loc[traindataset['process_ID']=='QCD_pfff', 'weight']
    weights_for_QCD_pp = traindataset.loc[traindataset['process_ID']=='QCD_pp', 'weight']
    # weights_for_TTGsJets = traindataset.loc[traindataset['process_ID']=='TTGsJets', 'weight']
    # weights_for_bbgg = traindataset.loc[traindataset['process_ID']=='bbgg', 'weight']

    HHsum_weighted= sum(weights_for_HH)
    # Hggsum_weighted= sum(weights_for_Hgg)
    # DiPhotonsum_weighted= sum(weights_for_DiPhoton)
    QCD_pfff_sum_weighted= sum(weights_for_QCD_pfff)
    QCD_pp_sum_weighted= sum(weights_for_QCD_pp)
    # TTGsJetssum_weighted= sum(weights_for_TTGsJets)
    # bbggsum_weighted= sum(weights_for_bbgg)
    # bckgsum_weighted = Hggsum_weighted + DiPhotonsum_weighted +  QCDsum_weighted + TTGsJetssum_weighted + bbggsum_weighted
    bckgsum_weighted = QCD_pfff_sum_weighted + QCD_pp_sum_weighted

    nevents_for_HH = traindataset.loc[traindataset['process_ID']=='ggh', 'unweighted']
    # nevents_for_Hgg = traindataset.loc[traindataset['process_ID']=='Hgg', 'unweighted']
    # nevents_for_DiPhoton = traindataset.loc[traindataset['process_ID']=='DiPhoton', 'unweighted']
    nevents_for_QCD_pfff = traindataset.loc[traindataset['process_ID']=='QCD_pfff', 'unweighted']
    nevents_for_QCD_pp = traindataset.loc[traindataset['process_ID']=='QCD_pp', 'unweighted']
    # nevents_for_TTGsJets = traindataset.loc[traindataset['process_ID']=='TTGsJets', 'unweighted']
    # nevents_for_bbgg = traindataset.loc[traindataset['process_ID']=='bbgg', 'unweighted']

    HHsum_unweighted= sum(nevents_for_HH)
    # Hggsum_unweighted= sum(nevents_for_Hgg)
    # DiPhotonsum_unweighted= sum(nevents_for_DiPhoton)
    QCD_pfff_sum_unweighted= sum(nevents_for_QCD_pfff)
    QCD_pp_sum_unweighted= sum(nevents_for_QCD_pp)
    # TTGsJetssum_unweighted= sum(nevents_for_TTGsJets)
    # bbggsum_unweighted= sum(nevents_for_bbgg)
    bckgsum_unweighted = QCD_pfff_sum_unweighted + QCD_pp_sum_unweighted

    # HHsum_weighted = 2*HHsum_weighted
    # HHsum_unweighted = 2*HHsum_unweighted

    if weights=='BalanceYields':
        print('#---------------------------------------')
        print('#    BalanceYields: Print weight       #')
        print('#---------------------------------------')
        print('{0:22} = {1:11}'.format('HHsum_weighted' , HHsum_weighted))
        # print('{0:22} = {1:11}'.format('Hggsum_weighted' , Hggsum_weighted))
        # print('{0:22} = {1:11}'.format('DiPhotonsum_weighted', DiPhotonsum_weighted))
        print('{0:22} = {1:11}'.format('QCD_pfff_sum_weighted', QCD_pfff_sum_weighted))
        print('{0:22} = {1:11}'.format('QCD_pp_sum_weighted', QCD_pp_sum_weighted))
        # print('{0:22} = {1:11}'.format('TTGsJetssum_weighted', TTGsJetssum_weighted))
        # print('{0:22} = {1:11}'.format('bbggsum_weighted ',bbggsum_weighted))
        print('{0:22} = {1:11}'.format('bckgsum_weighted', bckgsum_weighted))
        print('New classweight: (HHsum_unweighted/HHsum_weighted) = ',(HHsum_unweighted/HHsum_weighted) )
        print('New classweight: (HHsum_unweighted/bckgsum_unweighted) = ',(HHsum_unweighted/bckgsum_unweighted))
        print('#---------------------------------------')
        traindataset.loc[traindataset['process_ID']=='ggh', ['classweight']] = (bckgsum_unweighted / HHsum_unweighted) * (bckgsum_weighted / HHsum_weighted)
        # traindataset.loc[traindataset['process_ID']=='Hgg', ['classweight']] = (HHsum_unweighted/bckgsum_weighted)
        # traindataset.loc[traindataset['process_ID']=='DiPhoton', ['classweight']] = (HHsum_unweighted/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='QCD_pfff', ['classweight']] = 1
        traindataset.loc[traindataset['process_ID']=='QCD_pp', ['classweight']] = 1
        # traindataset.loc[traindataset['process_ID']=='TTGsJets', ['classweight']] = (HHsum_unweighted/bckgsum_weighted)

    if weights=='BalanceNonWeighted':
        print('#---------------------------------------')
        print('#    BalanceNonWeighted: Print weight  #')
        print('#---------------------------------------')
        print('HHsum_unweighted= ' , HHsum_unweighted)
        # print('Hggsum_unweighted= ' , Hggsum_unweighted)
        # print('DiPhotonsum_unweighted= ', DiPhotonsum_unweighted)
        print('QCD_pfff_sum_unweighted= ', QCD_pfff_sum_unweighted)
        print('QCD_pp_sum_unweighted= ', QCD_pp_sum_unweighted)
        # print('TTGsJetssum_unweighted= ', TTGsJetssum_unweighted)
        # print('bbggsum_unweighted = ', bbggsum_unweighted)
        print('bckgsum_unweighted= ', bckgsum_unweighted)
        print('#---------------------------------------')

        traindataset.loc[traindataset['process_ID']=='ggh', ['classweight']] = (bckgsum_unweighted / HHsum_unweighted)*(bck)
        # traindataset.loc[traindataset['process_ID']=='Hgg', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        # traindataset.loc[traindataset['process_ID']=='DiPhoton', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='QCD_pfff', ['classweight']] = 1
        traindataset.loc[traindataset['process_ID']=='QCD_pp', ['classweight']] = 1
        # traindataset.loc[traindataset['process_ID']=='TTGsJets', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        # traindataset.loc[traindataset['process_ID']=='bbgg', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)

    # exit()

    # Remove column headers that aren't input variables
    # Remove column headers that aren't input variables
    nonTrainingVariables = ['weight','Norm_SFs', 'weight_NLO_SM', 'kinWeight', 'unweighted', 'target', 'key', 'classweight', 'process_ID']
    # training_columns = column_headers[:-6]
    training_columns = [h for h in column_headers if h not in nonTrainingVariables]
    print('#---------------------------------------')
    print('<train-DNN> Training features: ', training_columns)
    print('<train-DNN> len(Training features): ', len(training_columns))
    print('#---------------------------------------')

    column_order_txt = '%s/column_order.txt' %(output_directory)
    column_order_file = open(column_order_txt, "wb")
    for tc_i in training_columns:
        line = tc_i+"\n"
        pickle.dump(str(line), column_order_file)

    num_variables = len(training_columns)

    # Extract training and testing data
    X_train = traindataset[training_columns].values
    X_test = valdataset[training_columns].values

    # Extract labels data
    Y_train = traindataset['target'].values
    Y_test = valdataset['target'].values

    # Create dataframe containing input features only (for correlation matrix)
    train_df = data.iloc[:traindataset.shape[0]]

    # Event weights if wanted
    # train_weights = traindataset['weight'].values
    # test_weights = valdataset['weight'].values

    # Weights applied during training.
    if weights=='BalanceYields':
        trainingweights = traindataset.loc[:,'classweight']*traindataset.loc[:,'weight']*traindataset.loc[:,'weight_NLO_SM']
        test_weights = valdataset.loc[:,'classweight']*valdataset.loc[:,'weight']*valdataset.loc[:,'weight_NLO_SM']

    if weights=='BalanceNonWeighted':
        trainingweights = traindataset.loc[:,'classweight']*traindataset.loc[:,'weight_NLO_SM']
    trainingweights = np.array(trainingweights)
    # Event weights for overfitting plots
    train_weights = trainingweights
    test_weights = np.array(test_weights)

    ## Input Variable Correlation plot
    correlation_plot_file_name = 'correlation_plot'
    Plotter.correlation_matrix(train_df)
    Plotter.save_plots(dir=plots_dir, filename=correlation_plot_file_name+'.png')
    Plotter.save_plots(dir=plots_dir, filename=correlation_plot_file_name+'.pdf')

    # print(Plotter.corrFilter(train_df, .15))

    # exit()

    # Fit label encoder to Y_train
    newencoder = LabelEncoder()
    newencoder.fit(Y_train)
    # Transform to encoded array
    encoded_Y = newencoder.transform(Y_train)
    encoded_Y_test = newencoder.transform(Y_test)

    if do_model_fit == 1:
        print('<train-BinaryDNN> Training new model . . . . ')
        histories = []
        labels = []

        if hyp_param_scan == 1:
            print('Begin at local time: ', time.localtime())
            hyp_param_scan_name = output_directory+'/hyp_param_scan_results.txt'
            hyp_param_scan_results = open(hyp_param_scan_name,'a')
            time_str = str(time.localtime())+'\n'
            hyp_param_scan_results.write(time_str+'\n')
            hyp_param_scan_results.write(weights+'\n')
            learn_rate = [0.00001, 0.0001, 0.001, 0.01]
            # epochs = [50]
            epochs = [100, 150, 200, 300]
            # batch_size = [60]
            batch_size = [60, 100, 200, 250]
            num_variables_arr =[num_variables]
            # optimizer = ['Adagrad']
            optimizer = ['Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
            init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
            activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
            neurons = [1, 5, 10, 15, 20, 25, 30, 50, 70]
            if GridSearch == 1:
                print("===============================================")
                print("==       GridSearchCV                        ==")
                print("===============================================")
                # param_grid = dict(learn_rate=learn_rate,epochs=epochs,batch_size=batch_size,optimizer=optimizer,num_variables=num_variables_arr,init_mode=init_mode)
                param_grid = dict(init_mode=init_mode)
                model = KerasClassifier(build_fn=gscv_model,verbose=1)
                grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
                grid_result = grid.fit(X_train,Y_train,shuffle=True,sample_weight=trainingweights)
                print("Best score: %f , best params: %s" % (grid_result.best_score_,grid_result.best_params_))
                dnn_parameter['Optimized_FH'][0]['epochs'] = grid_result.best_params_['epochs']
                dnn_parameter['Optimized_FH'][0]['batch_size'] = grid_result.best_params_['batch_size']
                dnn_parameter['Optimized_FH'][0]['learn_rate'] = grid_result.best_params_['learn_rate']
                dnn_parameter['Optimized_FH'][0]['optimizer'] = grid_result.best_params_['optimizer']
                f_dnn_parameter = open(output_directory+"/dnn_parameter.json", "w")   # open json file
                json.dump(dnn_parameter, f_dnn_parameter, indent=4, sort_keys=False)   # update json file
                f_dnn_parameter.close() # close json file
                hyp_param_scan_results.write("Best score: %f , best params: %s\n" %(grid_result.best_score_,grid_result.best_params_))
                means = grid_result.cv_results_['mean_test_score']
                stds = grid_result.cv_results_['std_test_score']
                params = grid_result.cv_results_['params']
                for mean, stdev, param in zip(means, stds, params):
                    print("Mean (stdev) test score: %f (%f) with parameters: %r" % (mean,stdev,param))
                    hyp_param_scan_results.write("Mean (stdev) test score: %f (%f) with parameters: %r\n" % (mean,stdev,param))
            elif RandomSearch == 1:
                print("===============================================")
                print("==       RandomizedSearchCV                  ==")
                print("===============================================")
                # param_grid = dict(learn_rate=learn_rate,epochs=epochs,batch_size=batch_size,optimizer=optimizer,num_variables=num_variables_arr)
                # param_grid = dict(init_mode=init_mode)
                # param_grid = dict(activation=activation)
                param_grid = dict(neurons=neurons)
                # model = KerasClassifier(build_fn=gscv_model,verbose=1)
                model = KerasClassifier(build_fn=gscv_model, epochs=300, batch_size=60, verbose=1)
                # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
                grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=-1, verbose=1)
                grid_result = grid.fit(X_train,Y_train,shuffle=True,sample_weight=trainingweights)
                print("Best score: %f , best params: %s" % (grid_result.best_score_,grid_result.best_params_))
                dnn_parameter['Optimized_FH'][0]['epochs'] = grid_result.best_params_['epochs']
                dnn_parameter['Optimized_FH'][0]['batch_size'] = grid_result.best_params_['batch_size']
                dnn_parameter['Optimized_FH'][0]['learn_rate'] = grid_result.best_params_['learn_rate']
                dnn_parameter['Optimized_FH'][0]['optimizer'] = grid_result.best_params_['optimizer']
                f_dnn_parameter = open(output_directory+"/dnn_parameter.json", "w")   # open json file
                json.dump(dnn_parameter, f_dnn_parameter, indent=4, sort_keys=False)   # update json file
                f_dnn_parameter.close() # close json file
                hyp_param_scan_results.write("Best score: %f , best params: %s\n" %(grid_result.best_score_,grid_result.best_params_))
                means = grid_result.cv_results_['mean_test_score']
                stds = grid_result.cv_results_['std_test_score']
                params = grid_result.cv_results_['params']
                for mean, stdev, param in zip(means, stds, params):
                    print("Mean (stdev) test score: %f (%f) with parameters: %r" % (mean,stdev,param))
                    hyp_param_scan_results.write("Mean (stdev) test score: %f (%f) with parameters: %r\n" % (mean,stdev,param))
            exit()
        else:
            learn_rate  = args.learnRate
            epochs  = args.epochs
            batch_size= args.batch_size
            optimizer= args.optimizer

            print("DNN parameters: Before traning the model:")
            print("\tepochs: ",epochs)
            print("\tbatch_size: ",batch_size)
            print("\tlearn_rate: ",learn_rate)
            print("\toptimizer: ",optimizer)

            # Define model for analysis
            early_stopping_monitor = EarlyStopping(patience=100, monitor='val_loss', min_delta=0.01, verbose=0) # callbacks
            # Learning rate schedular
            LearnRateScheduler = LearningRateScheduler(custom_LearningRate_schedular,verbose=1) # callbacks
            if (args.dynamic_lr):
                LearnRateScheduler = LearningRateScheduler(custom_LearningRate_schedular,verbose=1) # callbacks
            csv_logger = CSVLogger('%s/training.log'%(output_directory), separator=',', append=True) # callbacks

            if args.ModelToUse == "SL":
                model = SL_MultiClassModel(num_variables, optimizer=optimizer, activation=activation, dropout_rate=dropout_rate, learn_rate=learn_rate)
            elif args.ModelToUse == "FH_ANv5":
                model = new_model5(num_variables, optimizer=optimizer, activation=activation, dropout_rate=dropout_rate, learn_rate=learn_rate)
            elif args.ModelToUse == "FH_ANv5_NoBN":
                model = new_model6(num_variables, optimizer=optimizer, activation=activation, dropout_rate=dropout_rate, learn_rate=learn_rate)
            elif args.ModelToUse == "FH_ANv5_NoBN_NoDO":
                model = new_model6(num_variables, optimizer=optimizer, activation=activation, dropout_rate=dropout_rate, learn_rate=learn_rate)
            elif args.ModelToUse == "SimpleV1":
                model = ANN_model(num_variables, optimizer=optimizer, activation=activation, dropout_rate=dropout_rate, learn_rate=learn_rate)
            elif args.ModelToUse == "SimpleV2":
                model = baseline_model(num_variables, optimizer=optimizer, activation=activation, dropout_rate=dropout_rate, learn_rate=learn_rate)
            else:
                model = new_model6(num_variables, optimizer=optimizer, activation=activation, dropout_rate=dropout_rate, learn_rate=learn_rate)
            print("Model used: {}".format(args.ModelToUse))
            print("Model Summary:")
            print("="*51)
            model.summary()
            print("="*51)

            # Tensorboard
            # logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

            # Fit the model
            # Batch size = examples before updating weights (larger = faster training)
            # Epoch = One pass over data (useful for periodic logging and evaluation)
            #class_weights = np.array(class_weight.compute_class_weight('balanced',np.unique(Y_train),Y_train))
            if (args.dynamic_lr):
                print('#---------------------------------------')
                print('#    dynamic learn rate True           #')
                print('#    Command:\n\thistory = model.fit(X_train,Y_train,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=True,sample_weight=trainingweights,callbacks=[early_stopping_monitor,LearnRateScheduler,csv_logger])')
                print('#---------------------------------------')
                # Sample Weight
                # history = model.fit(X_train,Y_train,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=True,sample_weight=trainingweights,callbacks=[early_stopping_monitor,LearnRateScheduler,csv_logger])
                # Class Weight
                history = model.fit(X_train,Y_train,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=True,class_weight=class_weight,callbacks=[early_stopping_monitor,csv_logger,LearnRateScheduler])
            elif args.classweight:
                print('#---------------------------------------')
                print('#    classweight: True                 #')
                print('#    Command:\n\thistory = model.fit(X_train,Y_train,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=True,callbacks=[early_stopping_monitor,csv_logger],class_weight=class_weight)')
                print('#---------------------------------------')
                history = model.fit(X_train,Y_train,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=True,class_weight=class_weight,callbacks=[early_stopping_monitor,csv_logger])
            elif args.sampleweight:
                print('#---------------------------------------')
                print('#    sampleweight True                 #')
                print('#    Command:\n\thistory = model.fit(X_train,Y_train,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True,sample_weight=trainingweights,callbacks=[early_stopping_monitor,csv_logger])')
                print('#---------------------------------------')
                history = model.fit(X_train,Y_train,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=True,sample_weight=trainingweights,callbacks=[early_stopping_monitor,csv_logger])
            else:
                print('#---------------------------------------------------------------------------')
                print('#    without dynamic_learn rate, no sampleweight, no classweight           #')
                print('#    Command:\n\thistory = model.fit(X_train,Y_train,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=True,callbacks=[early_stopping_monitor,csv_logger])')
                print('#---------------------------------------------------------------------------')
                history = model.fit(X_train,Y_train,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=True,callbacks=[early_stopping_monitor,csv_logger])
            histories.append(history)
            labels.append(optimizer)

            # Make plot of loss function evolution
            Plotter.plot_training_progress_acc(histories, labels)
            acc_progress_filename = 'DNN_acc_wrt_epoch'
            Plotter.save_plots(dir=plots_dir, filename=acc_progress_filename+'.png')
            Plotter.save_plots(dir=plots_dir, filename=acc_progress_filename+'.pdf')

            Plotter.history_plot(history, label='loss')
            Plotter.save_plots(dir=plots_dir, filename='history_loss.png')
            Plotter.save_plots(dir=plots_dir, filename='history_loss.pdf')

            Plotter.plot_metrics(history)
            all_metrics = 'all_metrics'
            Plotter.save_plots(dir=plots_dir, filename=all_metrics+'.png')
            Plotter.save_plots(dir=plots_dir, filename=all_metrics+'.pdf')
    else:
        model_name = os.path.join(output_directory,'model.h5')
        model = load_trained_model(model_name)

    # Node probabilities for training sample events
    result_probs = model.predict(np.array(X_train))
    # result_classes = model.predict_classes(np.array(X_train))
    # model.predict_classes is going to be deprecated.. so one should use np.argmax
    # result_classes = np.argmax(model.predict(np.array(X_train)), axis=-1)
    result_classes = np.argmax(result_probs, axis=1)

    # Node probabilities for testing sample events
    result_probs_test = model.predict(np.array(X_test))
    # result_classes_test = model.predict_classes(np.array(X_test))
    # result_classes_test = np.argmax(model.predict(np.array(X_test)), axis=-1)
    result_classes_test = np.argmax(result_probs_test, axis=1)

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)
    y_test = np.argmax(Y_test, axis=-1)
    # cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print("=================")
    # print(cm)
    print("=================")

    # plt.figure(figsize=(5,5))
    # sns.heatmap(cm, annot=True, fmt="d")
    # # plt.title('Confusion matrix @{:.2f}'.format(p))
    # plt.title('Confusion matrix')
    # plt.ylabel('Actual label')
    # plt.xlabel('Predicted label')
    # plt.tight_layout()
    # plt.savefig(plots_dir + "/confusion_matrix.png")

    # print("=================")
    # print("f1_score")
    # print("f1 score (binary)     : {}".format(f1_score(y_test, y_pred,average='binary')))
    # print("f1 score (micro)     : {}".format(f1_score(y_test, y_pred,average='micro')))
    # print("f1 score (macro)     : {}".format(f1_score(y_test, y_pred,average='macro')))
    # print("f1 score (weighted)  : {}".format(f1_score(y_test, y_pred,average='weighted')))
    # print("=================")

    # print("f1_score")
    # print("f1 score (binary, labels=0,1)     : {}".format(f1_score(y_test, y_pred,labels=[0,1],average='binary')))
    # print("f1 score (micro, labels=0,1)     : {}".format(f1_score(y_test, y_pred,labels=[0,1],average='micro')))
    # print("f1 score (macro, labels=0,1)     : {}".format(f1_score(y_test, y_pred,labels=[0,1],average='macro')))
    # print("f1 score (weighted, labels=0,1)  : {}".format(f1_score(y_test, y_pred,labels=[0,1],average='weighted')))
    # print("=================")


    # print("f1_score")
    # print("f1 score (binary, labels=0)     : {}".format(f1_score(y_test, y_pred,labels=[0],average='binary')))
    # print("f1 score (micro, labels=0)     : {}".format(f1_score(y_test, y_pred,labels=[0],average='micro')))
    # print("f1 score (macro, labels=0)     : {}".format(f1_score(y_test, y_pred,labels=[0],average='macro')))
    # print("f1 score (weighted, labels=0)  : {}".format(f1_score(y_test, y_pred,labels=[0],average='weighted')))
    # print("=================")

    # print("f1_score")
    # print("f1 score (binary, labels=1)     : {}".format(f1_score(y_test, y_pred,labels=[1],average='binary')))
    # print("f1 score (micro, labels=1)     : {}".format(f1_score(y_test, y_pred,labels=[1],average='micro')))
    # print("f1 score (macro, labels=1)     : {}".format(f1_score(y_test, y_pred,labels=[1],average='macro')))
    # print("f1 score (weighted, labels=1)  : {}".format(f1_score(y_test, y_pred,labels=[1],average='weighted')))

    print("=================")

    # Store model in file
    model_output_name = os.path.join(output_directory,'model.h5')
    model.save(model_output_name)
    weights_output_name = os.path.join(output_directory,'model_weights.h5')
    model.save_weig`hts(weights_output_name)
    model_json = model.to_json()
    model_json_name = os.path.join(output_directory,'model_serialised.json')

    ##-- Convert model to pb
    CONVERT_COMMAND = "python convert_hdf5_2_pb.py --input %s/model.h5 --output %s/model.pb"%(output_directory, output_directory)
    print("Converting model.h5 to model.pb...")
    print(CONVERT_COMMAND)
    os.system(CONVERT_COMMAND)

    with open(model_json_name,'w') as json_file:
        json_file.write(model_json)
    model.summary()
    model_schematic_name = os.path.join(output_directory,'model_schematic.png')
    plot_model(model, to_file=model_schematic_name, show_shapes=True,
                      show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96) # rankdir='LR' for horizontal plot

    # exit()
    print('================')
    print('Training event labels: ', len(Y_train))
    print('Training event probs', len(result_probs))
    print('Training event weights: ', len(train_weights))
    print('Testing events: ', len(Y_test))
    print('Testing event probs', len(result_probs_test))
    print('Testing event weights: ', len(test_weights))
    print('================')

    # Initialise output directory.
    Plotter.plots_directory = plots_dir
    Plotter.output_directory = output_directory

    # Make overfitting plots of output nodes
    Plotter.binary_overfitting(model, Y_train, Y_test, result_probs, result_probs_test, plots_dir, train_weights, test_weights)

    Plotter.ROC(model, X_test, Y_test, X_train, Y_train)
    Plotter.save_plots(dir=plots_dir, filename='ROC.png')
    Plotter.save_plots(dir=plots_dir, filename='ROC.pdf')


    print("="*51)
    print("\tSHAP computation: DeepExplainer")
    print("="*51)
    e = shap.DeepExplainer(model, X_train[:400, ])
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    shap_values = e.shap_values(X_test[:400, ])
    Plotter.plot_dot(title="DeepExplainer_sigmoid_y0", x=X_test[:400, ], shap_values=shap_values, column_headers=column_headers)
    Plotter.plot_dot_bar(title="DeepExplainer_Bar_sigmoid_y0", x=X_test[:400,], shap_values=shap_values, column_headers=column_headers)
    Plotter.plot_dot_bar_all(title="DeepExplainer_Bar_sigmoid_y0_all", x=X_test[:400,], shap_values=shap_values, column_headers=column_headers)

    print("\tSHAP computation: GradientExplainer")
    e = shap.GradientExplainer(model, X_train[:100, ])
    shap_values = e.shap_values(X_test[:100, ])
    Plotter.plot_dot(title="GradientExplainer_sigmoid_y0", x=X_test[:100, ], shap_values=shap_values, column_headers=column_headers)
    Plotter.plot_dot_bar(title="GradientExplainer_Bar_sigmoid_y0", x=X_test[:400,], shap_values=shap_values, column_headers=column_headers)
    Plotter.plot_dot_bar_all(title="GradientExplainer_Bar_sigmoid_y0_all", x=X_test[:400,], shap_values=shap_values, column_headers=column_headers)
    #e = shap.KernelExplainer(model.predict, X_train[:100, ])
    #shap_values = e.shap_values(X_test[:100, ])
    #Plotter.plot_dot(title="KernelExplainer_sigmoid_y0", x=X_test[:100, ],shap_values=shap_values, column_headers=column_headers)
    #Plotter.plot_dot_bar(title="KernelExplainer_Bar_sigmoid_y0", x=X_test[:100,], shap_values=shap_values, column_headers=column_headers)
    #Plotter.plot_dot_bar_all(title="KernelExplainer_bar_All_Var_sigmoid_y0", x=X_test[:100,], shap_values=shap_values, column_headers=column_headers)

    # Create confusion matrices for training and testing performance
    # Plotter.conf_matrix(original_encoded_train_Y,result_classes_train,train_weights,'index')
    # Plotter.save_plots(dir=plots_dir, filename='yields_norm_confusion_matrix_TRAIN.png')
    # Plotter.conf_matrix(original_encoded_test_Y,result_classes_test,test_weights,'index')
    # Plotter.save_plots(dir=plots_dir, filename='yields_norm_confusion_matrix_TEST.png')

    # Plotter.conf_matrix(original_encoded_train_Y,result_classes_train,train_weights,'columns')
    # Plotter.save_plots(dir=plots_dir, filename='yields_norm_columns_confusion_matrix_TRAIN.png')
    # Plotter.conf_matrix(original_encoded_test_Y,result_classes_test,test_weights,'columns')
    # Plotter.save_plots(dir=plots_dir, filename='yields_norm_columns_confusion_matrix_TEST.png')

    # Plotter.conf_matrix(original_encoded_train_Y,result_classes_train,train_weights,'')
    # Plotter.save_plots(dir=plots_dir, filename='yields_matrix_TRAIN.png')
    # Plotter.conf_matrix(original_encoded_test_Y,result_classes_test,test_weights,'')
    # Plotter.save_plots(dir=plots_dir, filename='yields_matrix_TEST.png')

main()
