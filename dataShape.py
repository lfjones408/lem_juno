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
        event = file[key]

        for key2 in event.keys():
            if key2 == 'Energy' or key2 == 'Zenith' or key2 == 'NuType':
                continue
            print(event[key2])

            pmt_event = event[key2]

            Trigger_data = pmt_event['PMT_Features'][:]
            print(Trigger_data[:, 0])
   

        print(event['Energy'][()])
        print(event['Zenith'][()])
        print(event['NuType'][()])

       