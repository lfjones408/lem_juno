import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def markerScale(data, type):
    if type == 'charge':
        marker = np.where(data <= 0, 0, (data / np.max(data)) * 10)
    elif type == 'time':
        # Replace zero values with a very small number to avoid divide by zero
        safe_data = np.where(data == 0, np.finfo(float).eps, data)
        marker = np.where(data <= 0, 0, (1 / safe_data) * 10)
    return marker

def print_first_values(event_key, f):
    print(f"-------- {event_key} ----------")
    Event_energy = f[f'{event_key}/energy'][()]
    Event_position = (f[f'{event_key}/vertex'][()])/1e3 # Convert to meters
    Event_nuType = f[f'{event_key}/nuType'][()]

    Event_FHT = f[f'{event_key}/PMT/FHT'][:]
    Event_Phi = f[f'{event_key}/PMT/Phi'][:]
    Event_Theta = f[f'{event_key}/PMT/Theta'][:]
    Event_maxCharge = f[f'{event_key}/PMT/maxCharge'][:]
    Event_sumCharge = f[f'{event_key}/PMT/sumCharge'][:]
    Event_maxTime = f[f'{event_key}/PMT/maxTime'][:]

    print(f"Energy   : {Event_energy}")
    print(f"NuType   : {Event_nuType}")
    print(f"Vertex   : {Event_position}")
    print(f"Phi      : {Event_Phi[1000]}")
    print(f"Theta    : {Event_Theta[1000]}")
    print(f"FHT      : {Event_FHT[1000]}")
    print(f"MaxTime  : {Event_maxTime[1000]}")
    print(f"MaxCharge: {Event_maxCharge[1000]}")
    print(f"SumCharge: {Event_sumCharge[1000]}")

def line_sphere_intersection(x0, y0, z0, x1, y1, z1, r):
    # Calculate the direction vector components
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    # Coefficients of the quadratic equation at^2 + bt + c = 0
    a = dx**2 + dy**2 + dz**2
    b = 2 * (dx * x0 + dy * y0 + dz * z0)
    c = x0**2 + y0**2 + z0**2 - r**2

    # Calculate the discriminant
    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        # No intersection
        return None, None
    elif discriminant == 0:
        # One intersection (tangent to the sphere)
        t = -b / (2*a)
        intersection = np.array([x0 + t*dx, y0 + t*dy, z0 + t*dz])
        return intersection, None
    else:
        # Two intersections
        t1 = (-b + np.sqrt(discriminant)) / (2*a)
        t2 = (-b - np.sqrt(discriminant)) / (2*a)
        intersection1 = np.array([x0 + t1*dx, y0 + t1*dy, z0 + t1*dz]).flatten()
        intersection2 = np.array([x0 + t2*dx, y0 + t2*dy, z0 + t2*dz]).flatten()
        return intersection1, intersection2

# Load the data
filename = 'data/libAtmos.h5'

event_count = 0

with h5py.File(filename, 'r') as f:
    # Iterate over each event in the HDF5 file
    for event_key in f.keys():
        print(f"Processing {event_key}")

        # Create a directory for the current event
        event_dir = f'plots/productions/J24_1_2/rec-0/{event_key}'
        os.makedirs(event_dir, exist_ok=True)

        # Load the data for the current event
        # Event_energy = f[f'{event_key}/energy'][()]
        # Event_position = (f[f'{event_key}/vertex'][()])/1e3 # Convert to meters
        # Event_nuType = f[f'{event_key}/nuType'][()]
        # Event_Edep = f[f'{event_key}/Edep'][:]
        # Event_EdepPos = f[f'{event_key}/EdepPos'][:]/1e3 # Convert to meters
        # Event_Tracks = f[f'{event_key}/muTrack'][:]
        # Event_Track_Initial = Event_Tracks[:, :3]
        # Event_Track_Final = Event_Tracks[:, 3:6]
        # Event_Track_Initial_Mom = Event_Tracks[:, 6:9]

        # print(f"Initial Track: {Event_Track_Initial}")
        # print(f"Final Track: {Event_Track_Final}")

        Event_PMT = f[f'{event_key}/PMT/FeatureVector'][:]
        Event_Phi = Event_PMT[:, 0]
        Event_Theta = Event_PMT[:, 1]
        Event_FHT = Event_PMT[:, 2]
        Event_maxTime = Event_PMT[:, 3]
        Event_maxCharge = Event_PMT[:, 4]
        Event_sumCharge = Event_PMT[:, 5]

    
        # Event_FHT = f[f'{event_key}/PMT/FHT'][:]
        # Event_Phi = f[f'{event_key}/PMT/Phi'][:]
        # Event_Theta = f[f'{event_key}/PMT/Theta'][:]
        # Event_maxCharge = f[f'{event_key}/PMT/maxCharge'][:]
        # Event_sumCharge = f[f'{event_key}/PMT/sumCharge'][:]
        # Event_maxTime = f[f'{event_key}/PMT/maxTime'][:]

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

        # Create 3D Plots for each feature
        fig1 = plt.figure(figsize=(24, 16))

        proj = '3d'
        # have normal scale for charge like
        markMaxCharge = markerScale(Event_maxCharge, 'charge')
        markSumCharge = markerScale(Event_sumCharge, 'charge')

        # add inverse for time like features
        markMaxTime = markerScale(Event_maxTime, 'time')
        markFHT     = markerScale(Event_FHT, 'time')

        ax1 = fig1.add_subplot(221, projection=proj)
        ax2 = fig1.add_subplot(222, projection=proj)
        ax3 = fig1.add_subplot(223, projection=proj)
        ax4 = fig1.add_subplot(224, projection=proj)

        # for i in range(Event_Track_Initial.shape[0]):
        #     ax1.plot([Event_Track_Initial[i, 0], Event_Track_Final[i, 0]], 
        #              [Event_Track_Initial[i, 1], Event_Track_Final[i, 1]], 
        #              [Event_Track_Initial[i, 2], Event_Track_Final[i, 2]], 'r')
        #     ax2.plot([Event_Track_Initial[i, 0], Event_Track_Final[i, 0]], 
        #              [Event_Track_Initial[i, 1], Event_Track_Final[i, 1]], 
        #              [Event_Track_Initial[i, 2], Event_Track_Final[i, 2]], 'r')
        #     ax3.plot([Event_Track_Initial[i, 0], Event_Track_Final[i, 0]], 
        #              [Event_Track_Initial[i, 1], Event_Track_Final[i, 1]], 
        #              [Event_Track_Initial[i, 2], Event_Track_Final[i, 2]], 'r')
        #     ax4.plot([Event_Track_Initial[i, 0], Event_Track_Final[i, 0]], 
        #              [Event_Track_Initial[i, 1], Event_Track_Final[i, 1]], 
        #              [Event_Track_Initial[i, 2], Event_Track_Final[i, 2]], 'r')

        fhtPlot       = ax1.scatter(x, y, z, c=Event_FHT      , cmap='plasma', marker='o', s=1            , alpha=0.8)
        maxTimePlot   = ax2.scatter(x, y, z, c=Event_maxTime  , cmap='plasma', marker='o', s=1            , alpha=0.8)
        maxChargePlot = ax3.scatter(x, y, z, c=Event_maxCharge, cmap='plasma', marker='o', s=1, alpha=0.8)
        sumChargePlot = ax4.scatter(x, y, z, c=Event_sumCharge, cmap='plasma', marker='o', s=1, alpha=0.8)

        limits = (-25, 25)

        ax1.set_xlim(limits)
        ax1.set_ylim(limits)
        ax1.set_zlim(limits)

        ax2.set_xlim(limits)
        ax2.set_ylim(limits)
        ax2.set_zlim(limits)

        ax3.set_xlim(limits)
        ax3.set_ylim(limits)
        ax3.set_zlim(limits)

        ax4.set_xlim(limits)
        ax4.set_ylim(limits)
        ax4.set_zlim(limits)


        ax1.set_title(f'3D Plot of FHT {event_key} PMT Data')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax2.set_title(f'3D Plot of maxTime {event_key} PMT Data')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        ax3.set_title(f'3D Plot of maxCharge {event_key} PMT Data')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')

        ax4.set_title(f'3D Plot of sumCharge {event_key} PMT Data')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')

        cbarFHT = plt.colorbar(fhtPlot, ax=ax1, pad=0.1)
        cbarFHT.set_label('FHT')

        cbarMaxTime = plt.colorbar(maxTimePlot, ax=ax2, pad=0.1)
        cbarMaxTime.set_label('maxTime')

        cbarMaxCharge = plt.colorbar(maxChargePlot, ax=ax3, pad=0.1)
        cbarMaxCharge.set_label('maxCharge')

        cbarSumCharge = plt.colorbar(sumChargePlot, ax=ax4, pad=0.1)
        cbarSumCharge.set_label('sumCharge')
        
        plt.savefig(f'{event_dir}/features_3d.pdf')
        plt.close(fig1)

        # Create Mollweide projection plot
        fig2 = plt.figure(figsize=(24, 16))

        ax1 = fig2.add_subplot(221, projection='mollweide')
        ax2 = fig2.add_subplot(222, projection='mollweide')
        ax3 = fig2.add_subplot(223, projection='mollweide')
        ax4 = fig2.add_subplot(224, projection='mollweide')

        phi = Event_Phi
        theta = ((np.pi/2) - Event_Theta)

        fhtPlot       = ax1.scatter(phi, theta, c=Event_FHT      , cmap='plasma', marker='o', s=5            , alpha=0.8)
        maxTimePlot   = ax2.scatter(phi, theta, c=Event_maxTime  , cmap='plasma', marker='o', s=5            , alpha=0.8)
        maxChargePlot = ax3.scatter(phi, theta, c=Event_maxCharge, cmap='plasma', marker='o', s=5, alpha=0.8)
        sumChargePlot = ax4.scatter(phi, theta, c=Event_sumCharge, cmap='plasma', marker='o', s=5, alpha=0.8)

        # for i in range(Event_Track_Initial.shape[0]):
        #     intersection_in, intersection_out = line_sphere_intersection(
        #         Event_Track_Initial[i, 0], Event_Track_Initial[i, 1], Event_Track_Initial[i, 2],
        #         Event_Track_Final[i, 0], Event_Track_Final[i, 1], Event_Track_Final[i, 2], r
        #     )

            # if intersection_in is not None:
            #     _, in_theta, in_phi = cartesian2spherical(intersection_in[0], intersection_in[1], intersection_in[2])
            #     in_theta = (np.pi / 2) - in_theta
            #     ax1.plot(in_phi, in_theta, 'rx')
            #     ax2.plot(in_phi, in_theta, 'rx')
            #     ax3.plot(in_phi, in_theta, 'rx')
            #     ax4.plot(in_phi, in_theta, 'rx')

            # if intersection_out is not None:
            #     _, out_theta, out_phi = cartesian2spherical(intersection_out[0], intersection_out[1], intersection_out[2])
            #     out_theta = (np.pi / 2) - out_theta
            #     ax1.plot(out_phi, out_theta, 'bx')
            #     ax2.plot(out_phi, out_theta, 'bx')
            #     ax3.plot(out_phi, out_theta, 'bx')
            #     ax4.plot(out_phi, out_theta, 'bx')

        ax1.set_title(f'Mollweide Projection of FHT {event_key} PMT Data')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.grid(True)

        ax2.set_title(f'Mollweide Projection of maxTime {event_key} PMT Data')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.grid(True)

        ax3.set_title(f'Mollweide Projection of maxCharge {event_key} PMT Data')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.grid(True)

        ax4.set_title(f'Mollweide Projection of sumCharge {event_key} PMT Data')
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        ax4.grid(True)

        cbarFHT = plt.colorbar(fhtPlot, ax=ax1, pad=0.1)
        cbarFHT.set_label('FHT')

        cbarMaxTime = plt.colorbar(maxTimePlot, ax=ax2, pad=0.1)
        cbarMaxTime.set_label('maxTime')

        cbarMaxCharge = plt.colorbar(maxChargePlot, ax=ax3, pad=0.1)
        cbarMaxCharge.set_label('maxCharge')

        cbarSumCharge = plt.colorbar(sumChargePlot, ax=ax4, pad=0.1)
        cbarSumCharge.set_label('sumCharge')

        plt.savefig(f'{event_dir}/features_mollweide.pdf')  
        plt.close(fig2)

        # 2D plots
        fig3 = plt.figure(figsize=(24, 16))

        ax1 = fig3.add_subplot(221)
        ax2 = fig3.add_subplot(222)
        ax3 = fig3.add_subplot(223)
        ax4 = fig3.add_subplot(224)

        cosTheta = np.cos(Event_Theta)

        fhtPlot       = ax1.scatter(Event_Phi, cosTheta, c=Event_FHT      , cmap='plasma', marker='o', s=5            , alpha=0.8)
        maxTimePlot   = ax2.scatter(Event_Phi, cosTheta, c=Event_maxTime  , cmap='plasma', marker='o', s=5            , alpha=0.8)
        maxChargePlot = ax3.scatter(Event_Phi, cosTheta, c=Event_maxCharge, cmap='plasma', marker='o', s=5, alpha=0.8)
        sumChargePlot = ax4.scatter(Event_Phi, cosTheta, c=Event_sumCharge, cmap='plasma', marker='o', s=5, alpha=0.8)

        # for i in range(Event_Track_Initial.shape[0]):
        #     intersection_in, intersection_out = line_sphere_intersection(
        #         Event_Track_Initial[i, 0], Event_Track_Initial[i, 1], Event_Track_Initial[i, 2],
        #         Event_Track_Final[i, 0], Event_Track_Final[i, 1], Event_Track_Final[i, 2], r
        #     )

        #     if intersection_in is not None:
        #         _, in_theta , in_phi  = cartesian2spherical(intersection_in[0] , intersection_in[1] , intersection_in[2])
        #         in_costheta  = np.cos(in_theta)
        #         ax1.plot(in_phi, in_costheta, 'rx')
        #         ax2.plot(in_phi, in_costheta, 'rx')
        #         ax3.plot(in_phi, in_costheta, 'rx')
        #         ax4.plot(in_phi, in_costheta, 'rx')
 
        #     if intersection_out is not None:
        #         _, out_theta, out_phi = cartesian2spherical(intersection_out[0], intersection_out[1], intersection_out[2])
        #         out_costheta = np.cos(out_theta)
        #         ax1.plot(out_phi, out_costheta, 'bx')
        #         ax2.plot(out_phi, out_costheta, 'bx')
        #         ax3.plot(out_phi, out_costheta, 'bx')
        #         ax4.plot(out_phi, out_costheta, 'bx')

        ax1.set_title(f'2D Plot of FHT {event_key} PMT Data')
        ax1.set_xlabel('Phi')
        ax1.set_ylabel('cos(Theta)')
        ax1.grid(True)

        ax2.set_title(f'2D Plot of maxTime {event_key} PMT Data')
        ax2.set_xlabel('Phi')
        ax2.set_ylabel('cos(Theta)')
        ax2.grid(True)

        ax3.set_title(f'2D Plot of maxCharge {event_key} PMT Data')
        ax3.set_xlabel('Phi')
        ax3.set_ylabel('cos(Theta)')
        ax3.grid(True)

        ax4.set_title(f'2D Plot of sumCharge {event_key} PMT Data')
        ax4.set_xlabel('Phi')
        ax4.set_ylabel('cos(Theta)')
        ax4.grid(True)

        limitx = (-3, 3)
        limity = (-1, 1)

        ax1.set_xlim(limitx)
        ax1.set_ylim(limity)

        ax2.set_xlim(limitx)
        ax2.set_ylim(limity)

        ax3.set_xlim(limitx)
        ax3.set_ylim(limity)

        ax4.set_xlim(limitx)
        ax4.set_ylim(limity)

        cbarFHT = plt.colorbar(fhtPlot, ax=ax1, pad=0.1)
        cbarFHT.set_label('FHT')

        cbarMaxTime = plt.colorbar(maxTimePlot, ax=ax2, pad=0.1)
        cbarMaxTime.set_label('maxTime')
        
        cbarMaxCharge = plt.colorbar(maxChargePlot, ax=ax3, pad=0.1)
        cbarMaxCharge.set_label('maxCharge')

        cbarSumCharge = plt.colorbar(sumChargePlot, ax=ax4, pad=0.1)
        cbarSumCharge.set_label('sumCharge')

        plt.savefig(f'{event_dir}/features_2d.pdf')
        plt.close(fig3)

        event_count += 1

        if event_count >= 10:
            break

        # marker_sizes = (Event_maxCharge / np.max(Event_maxCharge)) * 10

        # fig1 = plt.figure(figsize=(24, 16))
        # ax1 = fig1.add_subplot(221, projection='mollweide')
        # ax2 = fig1.add_subplot(222, projection='mollweide')
        # ax3 = fig1.add_subplot(223, projection='3d')
        # ax4 = fig1.add_subplot(224, projection='3d')

        # alpha = np.pi/2
        # beta = 0
        # gamma = 0 

        # phi = Event_Phi
        # theta = ((np.pi/2) - Event_Theta)

        # R, theta_rot, phi_rot = rotate(r, Event_Theta, Event_Phi, alpha, 0, 0)
        # x_rot, y_rot, z_rot = spherical2cartesian(R, theta_rot, phi_rot)
        # theta_rot = ((np.pi/2) - theta_rot)

        # ax1.scatter(phi, theta, c=Event_maxCharge, cmap='plasma', marker='o', s=marker_sizes, alpha=0.8)
        # ax2.scatter(phi_rot, theta_rot, c=Event_maxCharge, cmap='plasma', marker='o', s=marker_sizes, alpha=0.8)
        # ax3.scatter(x, y, z, c=Event_maxCharge, cmap='plasma', marker='o', s=marker_sizes, alpha=0.8)
        # ax4.scatter(x_rot, y_rot, z_rot, c=Event_maxCharge, cmap='plasma', marker='o', s=marker_sizes, alpha=0.8)

        # ax1.set_title(f'Mollweide Projection of maxCharge {event_key} PMT Data')
        # ax1.set_xlabel('Longitude')
        # ax1.set_ylabel('Latitude')
        # ax1.grid(True)

        # ax2.set_title(f'Mollweide Projection of maxCharge {event_key} PMT Data')
        # ax2.set_xlabel('Longitude')
        # ax2.set_ylabel('Latitude')
        # ax2.grid(True)

        # ax3.set_title(f'3D Plot of maxCharge {event_key} PMT Data')
        # ax3.set_xlabel('X')
        # ax3.set_ylabel('Y')
        # ax3.set_zlabel('Z')

        # ax4.set_title(f'3D Plot of maxCharge {event_key} PMT Data')
        # ax4.set_xlabel('X')
        # ax4.set_ylabel('Y')
        # ax4.set_zlabel('Z')
        
        # plt.savefig(f'plots/rotationTest/FHT_mollweide_{event_key}.pdf')
        # plt.close(fig1)

        # for feature_name, feature_data in features.items():
        #     # Replace invalid values with a default size
        #     feature_data = np.nan_to_num(feature_data, nan=1.0, posinf=1.0, neginf=1.0)

        #     # Ensure all values are positive and non-zero
        #     feature_data = np.where(feature_data <= 0, 1.0, feature_data)

        #     # Scale marker sizes based on feature values
        #     marker_sizes = (feature_data / np.max(feature_data)) * 10

        #     fig1 = plt.figure(figsize=(12, 8))
        #     ax1 = fig1.add_subplot(111, projection='3d')

        #     # Use feature_data for color gradient and marker_sizes for size
        #     sphericalPlot = ax1.scatter(x, y, z, c=feature_data, cmap='plasma', marker='o', s=marker_sizes, alpha=0.8)

        #     # Add Edep positions
        #     edepX, edepY, edepZ = Event_EdepPos[:, 0], Event_EdepPos[:, 1], Event_EdepPos[:, 2]
        #     ax1.scatter(edepX, edepY, edepZ, c='red', marker='x', s=50, label='Edep')

        #     ax1.set_title(f'3D Plot of {feature_name} {event_key} PMT Data')
        #     ax1.set_xlabel('X')
        #     ax1.set_ylabel('Y')
        #     ax1.set_zlabel('Z')

        #     # Add color bar
        #     cbarSpherecial = plt.colorbar(sphericalPlot, ax=ax1, pad=0.1)
        #     cbarSpherecial.set_label(feature_name)

        #     # Save the plot in the event directory
        #     plt.savefig(f'{event_dir}/{feature_name}_3d.pdf')
        #     plt.close(fig1)

        #     # Create Mollweide projection plot
        #     fig2 = plt.figure(figsize=(12, 8))
        #     ax2 = fig2.add_subplot(111, projection='mollweide')

        #     # Convert spherical coordinates to radians for Mollweide projection
        #     phi_rad = Event_Phi
        #     theta_rad = np.radians(90 - np.degrees(Event_Theta))  # Convert to colatitude

        #     mollweidePlot = ax2.scatter(phi_rad, theta_rad, c=feature_data, cmap='plasma', marker='o', s=marker_sizes, alpha=0.8)

        #     ax2.set_title(f'Mollweide Projection of {feature_name} {event_key} PMT Data')
        #     ax2.set_xlabel('Longitude')
        #     ax2.set_ylabel('Latitude')
        #     ax2.grid(True)

        #     # Add color bar
        #     cbarMollweide = plt.colorbar(mollweidePlot, ax=ax2, pad=0.1)
        #     cbarMollweide.set_label(feature_name)

        #     # Save the plot in the event directory
        #     plt.savefig(f'{event_dir}/{feature_name}_mollweide.pdf')
        #     plt.close(fig2)