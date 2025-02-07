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

            fig1 = plt.figure(figsize=(12, 8))
            ax1 = fig1.add_subplot(111, projection='3d')

            # Use feature_data for color gradient and marker_sizes for size
            sphericalPlot = ax1.scatter(x, y, z, c=feature_data, cmap='plasma', marker='o', s=marker_sizes, alpha=0.8)

            # Add Edep positions
            edepX, edepY, edepZ = Event_EdepPos[:, 0], Event_EdepPos[:, 1], Event_EdepPos[:, 2]
            ax1.scatter(edepX, edepY, edepZ, c='red', marker='x', s=50, label='Edep')

            ax1.set_title(f'3D Plot of {feature_name} {event_key} PMT Data')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')

            # Add color bar
            cbarSpherecial = plt.colorbar(sphericalPlot, ax=ax1, pad=0.1)
            cbarSpherecial.set_label(feature_name)

            # Save the plot in the event directory
            plt.savefig(f'{event_dir}/{feature_name}_3d.pdf')
            plt.close(fig1)

            # Create Mollweide projection plot
            fig2 = plt.figure(figsize=(12, 8))
            ax2 = fig2.add_subplot(111, projection='mollweide')

            # Convert spherical coordinates to radians for Mollweide projection
            phi_rad = Event_Phi
            theta_rad = np.radians(90 - np.degrees(Event_Theta))  # Convert to colatitude

            mollweidePlot = ax2.scatter(phi_rad, theta_rad, c=feature_data, cmap='plasma', marker='o', s=marker_sizes, alpha=0.8)

            ax2.set_title(f'Mollweide Projection of {feature_name} {event_key} PMT Data')
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
            ax2.grid(True)

            # Add color bar
            cbarMollweide = plt.colorbar(mollweidePlot, ax=ax2, pad=0.1)
            cbarMollweide.set_label(feature_name)

            # Save the plot in the event directory
            plt.savefig(f'{event_dir}/{feature_name}_mollweide.pdf')
            plt.close(fig2)