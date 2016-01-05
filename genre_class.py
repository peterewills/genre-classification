from __future__ import division
from scipy.io.wavfile import read
import numpy as np
from features import mfcc
import os
import csv
import random
from sklearn.naive_bayes import GaussianNB

class Track(object):
    """
    A musical track.

    These are the fundamental objects of our classification algorithm.
    """
    
    def __init__(self,path,genre=None,sample=11025):
        """
        Initialize a track.
        """
        self.path = path
        self.genre = genre
        self.mfcc,self.sample = getmfcc(self.path)
        self.n_mfcc = self.mfcc / np.sqrt(energy(self.mfcc))

def train(train_path):
    """ Train on all .wav files in path given by user. """

    filelist = os.listdir(train_path)

    for path in filelist:
        if '.wav' not in path:
            filelist.remove(path)

    # initialize dictionary to contain genres
    genre_list = []

    with open(train_path+'genre_key.csv','rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            genre_list += [row]

    genre_set = set([item[1] for item in genre_list])
    genre_dict = dict(genre_list)

    train_collection = [Track(train_path+item,genre = genre_dict[item]) for item in filelist]
    
    return train_collection,genre_set
    
    
def collect(path):

    filelist = os.listdir(path)

    for item in filelist:
        if '.wav' not in item:
            filelist.remove(item)

    pathlist = [path+item for item in filelist]

    collection = [Track(trackpath) for trackpath in pathlist]

    return collection

def classify(query_path,train_collection,genre_set=None):
    """ """
    # Randomized dimension reduction, down to 1000 dimensions
    m = 1000
    
    t_group = train_collection
    q_group = collect(query_path)
    
    # Build GNB classifier, doing randomized dimension reduction
    t_mfccs = [track.n_mfcc.flatten() for track in t_group]
    q_mfccs = [track.n_mfcc.flatten() for track in q_group]
    
    n = len(t_mfccs[0])
    
    jl_ind = random.sample(range(n),m)
    t_mfccs = [mfcc[jl_ind] for mfcc in t_mfccs]
    q_mfccs = [mfcc[jl_ind] for mfcc in q_mfccs]
        
    t_genres = [gmap(track.genre,genre_set) for track in t_group]

    clf = GaussianNB()
    clf = clf.fit(t_mfccs,t_genres)
    
    for track in q_group:
        mfcc = track.n_mfcc.flatten()
        guess = int(clf.predict(mfcc[jl_ind]))
        track.genre = gmap(guess,genre_set)

    return q_group
    
def gmap(genre,gset):
    if isinstance(genre,int):
        glist = list(gset)
        return glist[genre]
    else:
        return list(gset).index(genre)
    
def energy(mfcc):
    """The L2 (Frobenius) norm of mfcc"""
    
    energy = sum(mfcc.flatten()**2)
    return energy

    
def getmiddle(sampling_rate,waveform):
    """ Function grabs the middle 2/3rds of a waveform sampled at a given rate"""
    
    a = sampling_rate
    b = waveform
    
    # trim to two minutes:
    one_min = a * 60
    middle = len(b)/2
    
    if len(b) * 2/3. < 2*one_min:
        # Take a slice of 2/3rds length out of the middle of the track
        midsection = b[middle-len(b)/3 : middle + len(b)/3]
        # loop that biz until it's 2 minutes long
        while len(midsection) < 2 * one_min:
            midsection = np.hstack((midsection,midsection))
        # trim it so it's exactly 2 minutes long
        midsection = midsection[:2 * one_min]
    else:
        # just take the middle 2 minutes if its long enough
        midsection = b[middle-one_min : middle+one_min]
        
    return midsection


def getmfcc(trackpath):
    """Builds mfcc of track given the path of the .wav file"""
    
    (a,b) = read(trackpath)
    
    midsection = getmiddle(a,b)
    
    # now just take the mfcc of midsection with appropriate options:
    win_len = .2
    win_step = .1
    num_bins = 30
    
    mfcc_feat = mfcc(midsection,a,win_len,win_step,num_bins)

    sample = a
    
    return mfcc_feat,sample
