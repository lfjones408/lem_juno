import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with h5py.File('data/libAtmos.h5', 'r') as file:
    def print_structure(name, obj):
        print(name)

    print("File structure:")
    file.visititems(print_structure)

    for key in file.keys():
        featurePath = f'{key}/PMT/FeatureVector'
        featureVector = file[featurePath][:]

        # muTrackPath = f'{key}/muTrack'
        # muTrack = file[muTrackPath][:]

        # print(f'Feature vector shape: {featureVector.shape}')
        # print(f'Muon track shape: {muTrack.shape}')