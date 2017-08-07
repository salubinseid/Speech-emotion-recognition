
import sys
import time
from time import gmtime, strftime

import os
import glob
import numpy
import cPickle
import aifc
import math
import numpy as np 
from numpy import NaN, Inf, arange, isscalar, array
from scipy.fftpack import rfft
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from scipy.signal import fftconvolve
from matplotlib.mlab import find
import matplotlib.pyplot as plt
from scipy import linalg as la
from scipy.signal import lfilter, hamming

import librosa
import librosa.display
import tensorflow as tf
from matplotlib.pyplot import specgram 

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    """
    stft - short time fourier transform
    Returns a complex-valued matrix D such that 
        np.abs(D[f, t]) is the magnitude of frequency bin f at frame t
        np.angle(D[f, t]) is the phase of frequency bin f at frame t
 

    """
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    """
    chorma_stft - 
    Compute a chromagram from a waveform or power spectrogram
    """
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    #return mfccs,chroma,mel,contrast,tonnetz
    return mfccs,chroma,mel

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
    features, labels = np.empty((0,180)), np.empty(0)
    #features, labels = np.empty((0,193)), np.empty(0)
    
    for label, sub_dir in enumerate(sub_dirs):
    	
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            
            try:
              mfccs, chroma, mel = extract_feature(fn) # function calling 
              #mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn) # function calling 
            except Exception as e:
              print "Error encountered while parsing file: ", fn
              continue
            ext_features = np.hstack([mfccs,chroma,mel])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn.split('/')[2].split('-')[0])

            print "processing emotion cate: ", fn.split('/')[2].split('-')[0], "at ", strftime("%Y-%m-%d %H:%M:%S", gmtime())

    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    
    """
    labels - holds array of input categories eg [0 2 0 0 0 1 1 1 1 1] 
    n_labels  - holds the number of input data eg 10 
    n_unique_labels - holds the number of categories  eg 3 
    one_hot_encode - 
    """

    n_labels = len(labels) 
    n_unique_labels = len(np.unique(labels))
    
    one_hot_encode = np.zeros((n_labels,n_unique_labels))  # create an array of zeros 10 by 3 
    #print one_hot_encode

    one_hot_encode[np.arange(n_labels), labels] = 1
    #print one_hot_encode
    
    return one_hot_encode

parent_dir = 'Sound-Data' # Training directory 
tr_sub_dirs = ["angry","exc","fea","fru","hap","neu","sad","sur"]
#tr_sub_dirs = ["fold1","fold2", "fold3", "fold4"]
# angry, excited, Fear, frustration,happy, neutral, sad, surprised

ts_sub_dirs = ["fold-ts"]   # testing Directory 
#ts_sub_dirs = ["fold-ts2"]   # testing Directory 

tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs) # Training Features 
tr_labels = one_hot_encode(tr_labels)
#print tr_labels

ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs) # Testing Features 
ts_labels = one_hot_encode(ts_labels)

#print tr_features # number of training features 
#print ts_labels   # number of labels like angry 

np.savetxt("tr_features.txt", tr_features)  # save array value to the text file
np.savetxt("tr_labels.txt", tr_labels)
