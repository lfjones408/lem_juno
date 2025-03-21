import os
import numpy as np
import h5py
import ROOT
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
from scipy.integrate import quad
from scipy.special import gammaln

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

def comparissonPlot(event1, event2, name):
    pmt1 = event1.get('featureVector')
    phi1 = pmt1[:,0]
    theta1 = pmt1[:,1]
    theta1 = np.pi/2 - theta1

    pmt2 = event2.get('featureVector')
    phi2 = pmt2[:,0]
    theta2 = pmt2[:,1]
    theta2 = np.pi/2 - theta2

    featureMap = [2, 3, 4, 5]
    features = ['FHT', 'maxTime', 'maxCharge', 'sumCharge']
    
    fig, axs = plt.subplots(4, 2, figsize=(24, 18), subplot_kw=dict(projection="mollweide"))
    axs = axs.flatten()

    # Set titles for each column
    title1 = 'Input Event: E={:.2f} MeV, Zenith={:.2f}, NuType={}'.format(event1['energy'][()], event1['zenith'][()], PDG2Name(event1['nuType'][()]))
    title2 = 'Match Event: E={:.2f} MeV, Zenith={:.2f}, NuType={}'.format(event2['energy'][()], event2['zenith'][()], PDG2Name(event2['nuType'][()]))
    fig.text(0.27, 0.95, title1, ha='center', fontsize=16)
    fig.text(0.73, 0.95, title2, ha='center', fontsize=16)

    for i, feature in enumerate(features):
        # Filter out phi and theta values where the feature value is 0
        map = featureMap[i]
        mask1 = np.array(pmt1[:, map]) != 0
        mask2 = np.array(pmt2[:, map]) != 0

        filtered_phi1 = phi1[mask1]
        filtered_theta1 = theta1[mask1]
        filtered_feature1 = pmt1[:, map][mask1]

        filtered_phi2 = phi2[mask2]
        filtered_theta2 = theta2[mask2]
        filtered_feature2 = pmt2[:, map][mask2]

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

    plot_dir = f'plots/lem_comparison'
    print(f"Saving plots to {plot_dir}")
    os.makedirs(plot_dir, exist_ok=True)

    plt.savefig(f'{plot_dir}/{name}.pdf')

def similarityPlot(similarity, energies, zenith):
    energies = np.array(energies, dtype='float64')
    zenith = np.array(zenith, dtype='float64')
    similarity = np.array(similarity, dtype='float64')

    graphEnergy = ROOT.TGraph(len(energies), np.array(energies), np.array(similarity))
    graphEnergy.GetXaxis().SetTitle("E_{Event} - E_{Test}")
    graphEnergy.GetYaxis().SetTitle("Similarity")
    graphEnergy.SetMarkerStyle(20)
    graphEnergy.SetLineWidth(2)
    graphEnergy.SetTitle("Energy vs Similarity")

    canvasEnergy = ROOT.TCanvas("canvasEnergy", "canvasEnergy", 800, 600)
    graphEnergy.Draw("AP")
    canvasEnergy.SaveAs("plots/lem_comparison/similarityEnergy.pdf")

    graphZenith = ROOT.TGraph(len(zenith), np.array(zenith), np.array(similarity))
    graphZenith.GetXaxis().SetTitle("Zenith_{Event} - Zenith_{Test}")
    graphZenith.GetYaxis().SetTitle("Similarity")
    graphZenith.SetMarkerStyle(20)
    graphZenith.SetLineWidth(2)
    graphZenith.SetTitle("Zenith vs Similarity")

    canvasZenith = ROOT.TCanvas("canvasZenith", "canvasZenith", 800, 600)
    graphZenith.Draw("AP")
    canvasZenith.SaveAs("plots/lem_comparison/similarityZenith.pdf")

def log_poisson_pmf(k, lambd):
    poiss = k*np.log(lambd) - lambd - gammaln(k + 1)
    return poiss

def log_gauss_pmf(x, mean, sigma):
    gauss = - (1/2)*np.log(2*np.pi*sigma**2) - ((x - mean)**2/(2*(sigma**2)))
    return gauss

def log_likelihood_noisy(n, mean, sigma):
    def integrand(lambd):
        log_poiss = log_poisson_pmf(n, lambd)
        log_gauss = log_gauss_pmf(lambd, mean, sigma)
        return np.exp(log_poiss + log_gauss)

    lower = max(0, mean - 5 * sigma)
    upper = mean + 5 * sigma

    integral, _ = quad(integrand, lower, upper)

    integral, _ = quad(integrand, lower, upper)
    log_integral = np.log(integral) if integral > 0 else -1000

    return log_integral

distances = []
sumLikelihood = []
dif_energies = []
dif_zeniths = []

# Define event data
filename = 'data/libAtm_big.h5'

# Rotation Test
counter = 0

with h5py.File(filename, 'r') as data:
    testEvent = data['Event_9']

    testEvent_energy = testEvent['energy'][()]
    testEvent_zenith = testEvent['zenith'][()]
    testEvent_nuType = testEvent['nuType'][()]

    testEvent_PMT = testEvent.get('featureVector')[:]
    testEvent_Phi = testEvent_PMT[:,0]
    testEvent_Theta = testEvent_PMT[:,1]
    testEvent_FHT = testEvent_PMT[:,2]
    testEvent_maxTime = testEvent_PMT[:,3]
    testEvent_maxCharge = testEvent_PMT[:,4]
    testEvent_sumCharge = testEvent_PMT[:,5]

    # Define weights for each feature
    weights = {
        'maxCharge': 1.0,
        'sumCharge': 0,
        'maxTime': 0,
        'FHT':  1.0
    }

    # Load in Library event data
    for event_keys in data.keys():
        if counter > 10:
            continue

        print(f"Comparing -> {event_keys}")

        counter = counter + 1 

        # Load the data for the current event
        Event_energy = data[f'{event_keys}/energy'][()]
        Event_zenith = data[f'{event_keys}/zenith'][()]
        Event_nuType = data[f'{event_keys}/nuType'][()]

        Event_PMT = data.get(f'{event_keys}/featureVector')[:]

        if len(Event_PMT) != 17612:
            continue

        Event_Phi = Event_PMT[:,0]
        Event_Theta = Event_PMT[:,1]
        Event_FHT = Event_PMT[:,2]
        Event_maxTime = Event_PMT[:,3]
        Event_maxCharge = Event_PMT[:,4]
        Event_sumCharge = Event_PMT[:,5]

        # New metric
        likelihood = []
        for test, MCdata in zip(testEvent_sumCharge, Event_sumCharge):
            sig = MCdata*0.1
            metricIter = log_likelihood_noisy(test, MCdata, sig)
            if not np.isfinite(metricIter):
                print(test, MCdata)
                print(metricIter)
            likelihood.append(metricIter)

        sumLikelihood.append((event_keys, np.sum(likelihood)))
        dif_energies.append(Event_energy - testEvent_energy)
        dif_zeniths.append(Event_zenith - testEvent_zenith)


    # Convert the distances to a numpy array
    sumLikelihood = np.array(sumLikelihood, dtype='object')

    print(sumLikelihood)

    best_match, max_likelihood = max(sumLikelihood, key = lambda x: x[1])

    print(f"Best match is {best_match}, with similarity: {max_likelihood:.2e}")

    # # Normalise the distances
    # maxDistance = max([d[1] for d in distances])

    # for i, (event_key, distance) in enumerate(distances):
    #     distances[i] = (event_key, normalise_distance(distance, maxDistance))
    #     # print(f"Normalised distance for event {event_key}: {distances[i][1]}")

    # # Find the event with the smallest non-zero distance
    # non_zero_distances = [(event_key, d) for event_key, d in distances if d > 0]
    # min_event, min_distance = min(non_zero_distances, key=lambda x: x[1])

    # similarity = np.array(distances)[:,1]

    # print(f"The smallest non-zero distance is: {min_distance} for {min_event}")

    # non_zero_distances =  np.array(non_zero_distances)
    
    # smallest_nonzero = np.partition(non_zero_distances[:,1], 10)[:10]

    # most_similar = non_zero_distances[np.isin(non_zero_distances[:,1], smallest_nonzero)]

    # energy_matches = []
    # zenith_matches = []

    # print("Most similar events: ")
    # for evt in most_similar:
    #     print(f"{evt[0]}: {evt[1]}")
    #     evtIter = data[f'{evt[0]}']
    #     # comparissonPlot(testEvent, evtIter, f"comparison {evt[0]}")
        
    #     energyIter = evtIter['energy'][()]
    #     zenithIter = evtIter['zenith'][()]
    #     energy_matches.append(energyIter)
    #     zenith_matches.append(zenithIter)

    # print(f"Energy : {np.mean(energy_matches)} +- {np.std(energy_matches)/np.sqrt(len(energy_matches))}")
    # print(f"Zenith : {np.mean(zenith_matches)} +- {np.std(zenith_matches)/np.sqrt(len(zenith_matches))}")

    # # Make a comparison plot
    comparissonPlot(testEvent, data[best_match], "bestMatch")
    similarityPlot(sumLikelihood[:,1], dif_energies, dif_zeniths)

    # TO DO: -> Change to poison statistics
    #           -> test
    #           -> optimise sigma
    #        -> Add parser option