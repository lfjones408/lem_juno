import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

def normalise_distance(distance, maxDistance):
    return distance / maxDistance

distances = []

# Define event data
with h5py.File('data/test.h5', 'r') as f:
    testEvent = f['/Event_0']

    testEvent_energy = testEvent['energy'][()]
    testEvent_position = (testEvent['vertex'][()])/1e3
    testEvent_nuType = testEvent['nuType'][()]

    testEvent_Phi = testEvent['PMT/Phi'][:]
    testEvent_Theta = testEvent['PMT/Theta'][:]
    testEvent_maxCharge = testEvent['PMT/maxCharge'][:]
    testEvent_sumCharge = testEvent['PMT/sumCharge'][:]
    testEvent_maxTime = testEvent['PMT/maxTime'][:]
    testEvent_FHT = testEvent['PMT/FHT'][:]

    # Define weights for each feature
    weights = {
        'position': 1.0,
        'phi': 1.0,
        'theta': 1.0,
        'maxCharge': 1.0,
        'sumCharge': 1.0,
        'maxTime': 1.0,
        'FHT': 1.0
    }

    # Load in Library event data
    for event_keys in f.keys():
        if event_keys == '/Event_0':
            continue
        print(f"Comparing -> {event_keys}")

        # Load the data for the current event
        Event_energy = f[f'{event_keys}/energy'][()]
        Event_position = (f[f'{event_keys}/vertex'][()])/1e3
        Event_nuType = f[f'{event_keys}/nuType'][()]

        Event_Phi = f[f'{event_keys}/PMT/Phi'][:]
        Event_Theta = f[f'{event_keys}/PMT/Theta'][:]
        Event_maxCharge = f[f'{event_keys}/PMT/maxCharge'][:]
        Event_sumCharge = f[f'{event_keys}/PMT/sumCharge'][:]
        Event_maxTime = f[f'{event_keys}/PMT/maxTime'][:]
        Event_FHT = f[f'{event_keys}/PMT/FHT'][:]

        # Calculate weighted Euclidean distances
        position_distance = weights['position'] * np.linalg.norm(testEvent_position - Event_position)
        phi_distance = weights['phi'] * np.linalg.norm(testEvent_Phi - Event_Phi)
        theta_distance = weights['theta'] * np.linalg.norm(testEvent_Theta - Event_Theta)
        maxCharge_distance = weights['maxCharge'] * np.linalg.norm(testEvent_maxCharge - Event_maxCharge)
        sumCharge_distance = weights['sumCharge'] * np.linalg.norm(testEvent_sumCharge - Event_sumCharge)
        maxTime_distance = weights['maxTime'] * np.linalg.norm(testEvent_maxTime - Event_maxTime)
        FHT_distance = weights['FHT'] * np.linalg.norm(testEvent_FHT - Event_FHT)

        # Combine distances into a single metric
        total_distance = (position_distance + phi_distance + theta_distance +
                          maxCharge_distance + sumCharge_distance +
                          maxTime_distance + FHT_distance)
        
        distances.append(total_distance)
        
# Normalise the distances
maxDistance = max(distances)

for i, distance in enumerate(distances):
    distances[i] = normalise_distance(distance, maxDistance)
    print(f"Normalised distance for event {i}: {distances[i]}")
