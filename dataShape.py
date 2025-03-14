import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with h5py.File('data/libAtm_big.h5', 'r') as file:
    # def print_structure(name, obj):
    #     print(name)

    # print("File structure:")
    # file.visititems(print_structure)

    for event_keys in file.keys():
        featureVector = file.get(f'{event_keys}/featureVector')
        pmtmax = 17612
        if len(featureVector) != pmtmax:
            shortpmt = len(featureVector)
            print(f"Short featureVector at {event_keys}: {shortpmt}")