import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import KDTree

def normalise_distance(distance, maxDistance):
    return distance / maxDistance

def PDG2Name(pdg):
    if pdg == 12:
        return 'nuE'
    elif pdg == -12:
        return 'Anti-nuE'
    elif pdg == 14:
        return 'nuMu'
    elif pdg == -14:
        return 'Anti-nuMu'
    else:
        return 'Unknown'

def spherical2cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cartesian2spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def rotate(r, theta, phi, alpha, beta, gamma):
    Rx = np.array( [[1,             0,              0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha),  np.cos(alpha)]])
    
    Ry = np.array( [[ np.cos(beta), 0, np.sin(beta)],
                    [            0, 1,            0],
                    [-np.sin(beta), 0, np.cos(beta)]])
    
    Rz = np.array( [[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma),  np.cos(gamma), 0],
                    [            0,              0, 1]])
    
    R = np.dot(Rz, np.dot(Ry, Rx))

    x, y, z = spherical2cartesian(r, theta, phi)
    x, y, z = np.dot(R, np.array([x, y, z]))
    r, theta, phi = cartesian2spherical(x, y, z)

    return r, theta, phi

def comparissonPlot(event1, event2):
    phi1 = event1['PMT/Phi'][:]
    theta1 = event1['PMT/Theta'][:]
    theta1 = np.pi/2 - theta1

    phi2 = event2['PMT/Phi'][:]
    theta2 = event2['PMT/Theta'][:]
    theta2 = np.pi/2 - theta2

    features = ['PMT/maxCharge', 'PMT/sumCharge', 'PMT/maxTime', 'PMT/FHT']
    
    fig, axs = plt.subplots(4, 2, figsize=(24, 18), subplot_kw=dict(projection="mollweide"))
    axs = axs.flatten()

    # Set titles for each column
    title1 = 'Input Event: E={:.2f} MeV, Position=({:.2f}, {:.2f}, {:.2f})m, NuType={}'.format(event1['energy'][()], event1['vertex'][()][0]/1e3, event1['vertex'][()][1]/1e3, event1['vertex'][()][2]/1e3, PDG2Name(event1['nuType'][()]))
    title2 = 'Match Event: E={:.2f} MeV, Position=({:.2f}, {:.2f}, {:.2f})m, NuType={}'.format(event2['energy'][()], event2['vertex'][()][0]/1e3, event2['vertex'][()][1]/1e3, event2['vertex'][()][2]/1e3, PDG2Name(event2['nuType'][()]))
    fig.text(0.27, 0.95, title1, ha='center', fontsize=16)
    fig.text(0.73, 0.95, title2, ha='center', fontsize=16)
    
    for i, feature in enumerate(features):
        # Filter out phi and theta values where the feature value is 0
        mask1 = np.array(event1[feature]) != 0
        mask2 = np.array(event2[feature]) != 0

        filtered_phi1 = phi1[mask1]
        filtered_theta1 = theta1[mask1]
        filtered_feature1 = event1[feature][mask1]

        filtered_phi2 = phi2[mask2]
        filtered_theta2 = theta2[mask2]
        filtered_feature2 = event2[feature][mask2]

        # Plot the data
        locals()[f'LHS_{feature}'] = axs[2*i].scatter(filtered_phi1, filtered_theta1, c=filtered_feature1, cmap='plasma', marker='o', s=2, alpha=0.8)
        axs[2*i].set_title(f'{feature}_event1')
        axs[2*i].set_xlabel('Phi')
        axs[2*i].set_ylabel('Theta')
        axs[2*i].grid(True)

        locals()[f'cbar_LHS_{feature}'] = plt.colorbar(locals()[f'LHS_{feature}'], ax=axs[2*i])

        locals()[f'RHS_{feature}'] = axs[2*i+1].scatter(filtered_phi2, filtered_theta2, c=filtered_feature2, cmap='plasma', marker='o', s=2, alpha=0.8)
        axs[2*i+1].set_title(f'{feature}_event2')
        axs[2*i+1].set_xlabel('Phi')
        axs[2*i+1].set_ylabel('Theta')
        axs[2*i+1].grid(True)

        locals()[f'cbar_RHS_{feature}'] = plt.colorbar(locals()[f'RHS_{feature}'], ax=axs[2*i+1])
        
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    name = event1.name.split('/')[-1]

    plot_dir = f'plots/{name}'
    os.makedirs(plot_dir, exist_ok=True)

    plt.savefig(f'{plot_dir}/bestMatch.pdf')

distances = []

# Define event data
filename = 'data/lib.h5'

# Rotation Test
    

with h5py.File(filename, 'r') as data:
    testEvent = data['Event_9']

    testEvent_energy = testEvent['energy'][()]
    testEvent_position = (testEvent['vertex'][()])/1e3
    testEvent_nuType = testEvent['nuType'][()]

    testEvent_PMT = testEvent['PMT/FeatureVector'][:]
    testEvent_Phi = testEvent[testEvent_PMT][:,0]
    testEvent_Theta = testEvent[testEvent_PMT][:,1]
    testEvent_FHT = testEvent[testEvent_PMT][:,2]
    testEvent_maxTime = testEvent[testEvent_PMT][:,3]
    testEvent_maxCharge = testEvent[testEvent_PMT][:,4]
    testEvent_sumCharge = testEvent[testEvent_PMT][:,5]

    # testEvent_Phi = testEvent['PMT/Phi'][:]
    # testEvent_Theta = testEvent['PMT/Theta'][:]
    # testEvent_maxCharge = testEvent['PMT/maxCharge'][:]
    # testEvent_sumCharge = testEvent['PMT/sumCharge'][:]
    # testEvent_maxTime = testEvent['PMT/maxTime'][:]
    # testEvent_FHT = testEvent['PMT/FHT'][:]

    # Define weights for each feature
    weights = {
        'maxCharge': 1.0,
        'sumCharge': 1.0,
        'maxTime': 1.0,
        'FHT': 1.0
    }

    # Load in Library event data
    for event_keys in data.keys():
        if event_keys == '/Event_0':
            continue
        print(f"Comparing -> {event_keys}")

        # Load the data for the current event
        Event_energy = data[f'{event_keys}/energy'][()]
        Event_position = (data[f'{event_keys}/vertex'][()])/1e3
        Event_nuType = data[f'{event_keys}/nuType'][()]

        Event_PMT = data[f'{event_keys}/PMT/FeatureVector'][:]
        Event_Phi = Event_PMT[:,0]
        Event_Theta = Event_PMT[:,1]
        Event_FHT = Event_PMT[:,2]
        Event_maxTime = Event_PMT[:,3]
        Event_maxCharge = Event_PMT[:,4]
        Event_sumCharge = Event_PMT[:,5]

        # Event_Phi = data[f'{event_keys}/PMT/Phi'][:]
        # Event_Theta = data[f'{event_keys}/PMT/Theta'][:]
        # Event_maxCharge = data[f'{event_keys}/PMT/maxCharge'][:]
        # Event_sumCharge = data[f'{event_keys}/PMT/sumCharge'][:]
        # Event_maxTime = data[f'{event_keys}/PMT/maxTime'][:]
        # Event_FHT = data[f'{event_keys}/PMT/FHT'][:]

        # Calculate weighted Euclidean distances
        maxCharge_distance = weights['maxCharge'] * np.linalg.norm(testEvent_maxCharge - Event_maxCharge)
        sumCharge_distance = weights['sumCharge'] * np.linalg.norm(testEvent_sumCharge - Event_sumCharge)
        maxTime_distance = weights['maxTime'] * np.linalg.norm(testEvent_maxTime - Event_maxTime)
        FHT_distance = weights['FHT'] * np.linalg.norm(testEvent_FHT - Event_FHT)

        # Combine distances into a single metric
        total_distance = (maxCharge_distance + sumCharge_distance +
                        maxTime_distance + FHT_distance)
        
        distances.append((event_keys, total_distance))

    
    # Normalise the distances
    maxDistance = max([d[1] for d in distances])

    for i, (event_key, distance) in enumerate(distances):
        distances[i] = (event_key, normalise_distance(distance, maxDistance))
        print(f"Normalised distance for event {event_key}: {distances[i][1]}")

    # Find the event with the smallest non-zero distance
    non_zero_distances = [(event_key, d) for event_key, d in distances if d > 0]
    min_event, min_distance = min(non_zero_distances, key=lambda x: x[1])

    print(f"The smallest non-zero distance is: {min_distance} for {min_event}")

    # Make a comparison plot
    comparissonPlot(testEvent, data[min_event])

