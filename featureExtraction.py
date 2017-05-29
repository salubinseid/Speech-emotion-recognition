
import sys
import time
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
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
    features, labels = np.empty((0,193)), np.empty(0)
    
    for label, sub_dir in enumerate(sub_dirs):
    	
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
              mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn) # function calling 
            except Exception as e:
              print "Error encountered while parsing file: ", fn
              continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn.split('/')[2].split('-')[1])
    	
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)

    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
   
    
    return one_hot_encode

parent_dir = 'Sound-Data'
tr_sub_dirs = ["fold1","fold2","fold3","fold4","fold5","fold6","fold7","fold8","fold9","fold10"]
# Fold1 - angry , Fold2 - happy, Fold3 - sad, Fold4 - neutral, Fold5 - frustrated, Fold6 - excited, 
# Fold7 - fearful, Fold8 - surprised, Fold9 - disgusted, Fold10 - other

ts_sub_dirs = ["fold-ts"]

tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)
ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)
tr_labels = one_hot_encode(tr_labels)
ts_labels = one_hot_encode(ts_labels)

#print tr_features


np.savetxt("tr_features.txt", tr_features)  # save array value to the text file
np.savetxt("tr_labels.txt", tr_labels)
