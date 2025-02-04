import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
with h5py.File('data/test.h5', 'r') as f:
    def print_group_contents(group, indent=0):
        count = 0
        for key in group.keys():
            if count >= 10:
                break
            item = group[key]
            if isinstance(item, h5py.Group):
                print(' ' * indent + f"Group: {key}")
                print_group_contents(item, indent + 2)
            elif isinstance(item, h5py.Dataset):
                print(' ' * indent + f"Dataset: {key}, Shape: {item.shape}")
            count += 1

    # print_group_contents(f)

    # Iterate over each event in the HDF5 file
    for event_key in f.keys():
        print(f"Processing {event_key}")

        # Create a directory for the current event
        event_dir = f'plots/{event_key}'
        os.makedirs(event_dir, exist_ok=True)

        # Load the data for the current event
        Event_energy = f[f'{event_key}/energy'][()]
        Event_position = (f[f'{event_key}/vertex'][()])/1e3 # Convert to meters
        Event_nuType = f[f'{event_key}/nuType'][()]
        Event_Edep = f[f'{event_key}/Edep'][:]
        Event_EdepPos = f[f'{event_key}/EdepPos'][:]/1e3 # Convert to meters

        Event_FHT = f[f'{event_key}/PMT/FHT'][:]
        Event_Phi = f[f'{event_key}/PMT/Phi'][:]
        Event_Theta = f[f'{event_key}/PMT/Theta'][:]
        Event_maxCharge = f[f'{event_key}/PMT/maxCharge'][:]
        Event_sumCharge = f[f'{event_key}/PMT/sumCharge'][:]
        Event_maxTime = f[f'{event_key}/PMT/maxTime'][:]

        # # Print event level information
        # print(f"Energy: {Event_energy}")
        # print(f"Position: {Event_position}")
        # print(f"NuType: {Event_nuType}")

        # # Print Edep information
        # print(f"Edep Shape    : {Event_Edep.shape}")
        # print(f"Edep          : {Event_Edep}")
        # print(f"EdepPos Shape : {Event_EdepPos.shape}")
        # print(f"EdepPos       : {Event_EdepPos}")

        # Spherical to Cartesian coordinates
        r = 20.05 # m
        x = r * np.sin(Event_Theta) * np.cos(Event_Phi)
        y = r * np.sin(Event_Theta) * np.sin(Event_Phi)
        z = r * np.cos(Event_Theta)

        # Create 3D plots for each feature
        features = {
            'FHT': Event_FHT,
            'maxCharge': Event_maxCharge,
            'sumCharge': Event_sumCharge,
            'maxTime': Event_maxTime
        }

        for feature_name, feature_data in features.items():
            # Replace invalid values with a default size
            feature_data = np.nan_to_num(feature_data, nan=1.0, posinf=1.0, neginf=1.0)

            # Ensure all values are positive and non-zero
            feature_data = np.where(feature_data <= 0, 1.0, feature_data)

            # Scale marker sizes based on feature values
            marker_sizes = (feature_data / np.max(feature_data)) * 10

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Use feature_data for color gradient and marker_sizes for size
            scatter = ax.scatter(x, y, z, c=feature_data, cmap='plasma', marker='o', s=marker_sizes, alpha=0.8)

            # Add Edep positions

            edepX, edepY, edepZ = Event_EdepPos[:, 0], Event_EdepPos[:, 1], Event_EdepPos[:, 2]
            ax.scatter(edepX, edepY, edepZ, c='red', marker='x', s=50, label='Edep')

            ax.set_title(f'3D Plot of {feature_name} {event_key} PMT Data')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Set axis limits to zoom out
            ax.set_xlim([-r, r])
            ax.set_ylim([-r, r])
            ax.set_zlim([-r, r])

            # Add color bar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label(feature_name)

            # Save the plot in the event directory
            plt.savefig(f'{event_dir}/{feature_name}_3d.pdf')
            plt.close()