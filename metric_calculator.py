import numpy as np
import math
import data_generator
import pickle
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


def compute_density_surface(x, y, sigma, dim=100):
    """
    Compute the density surface from an input point set with the given parameters

    Input:  
    x, the x coordinates of the points
    y, the y coordinates of the points
    sigma, the bandwidth of each kernel
    """

    # Initialize 2D array to hold the height values
    z = np.zeros((dim, dim), dtype=float)

    # Loop over all points and add their kernels to the surface
    for p in range(0, len(x)):
        for i in range(-sigma, sigma):
            for j in range(-sigma, sigma):
                d = math.sqrt(i*i + j*j)
                tx = (x[p] + i)
                ty = (y[p] + j)
                
                # If the point is out of bounds, skip it
                if (tx < 0 or tx >= dim or ty < 0 or ty >= dim):
                    continue

                # Add the kernel to the surface                
                # Cone kernel
                z[tx][ty] += max((1/len(x)) * (1 - (d/sigma)), 0)

    # Return the density surface
    return z

def compute_metric(x, y, surface):
    """
    Compute the metric value of an input point set given the density surface

    Input:
    x, the x coordinates of the points
    y, the y coordinates of the points
    surface, the density surface
    """

    sumval = 0
    c = 0

    for i in range(0, len(x)):
        if x[i] >= 0 and x[i] <= 100 and y[i] >= 0 and y[i] <= 100:
            sumval += (surface[x[i]][y[i]])
            c += 1

    return sumval - 1

def compute_clustering_profile(x, y, scales):
    """
    Compute the clustering profile of a point set for a given set of scales

    Input: 
    x, the x coordinates of the pointset
    y, the y coordinates of the pointset
    scales, a list of scales to include in the profile

    Output:
    The clustering profile of the point set, a vector of clustering values at different scales
    """

    profile = []
    
    for sigma in scales:
        temp_surface = compute_density_surface(x, y, sigma, dim=110)
        profile.append(compute_metric(x, y, temp_surface) / (sigma))

    return profile

def create_full_profile(name):
    """
    Create the full clustering profile of a point set

    Input:
    name, the name of the file that contains the pointset
    """
    
    x, y = data_generator.read_data("data/" + name)
    
    profile = compute_clustering_profile(x, y, range(1, 120))

    with open("profiles/fullprofile_" + name, "wb") as fp:
        pickle.dump(profile, fp)

def read_full_profile(name):
    """
    Read the full clustering profile of a point set

    Input:
    name, the name of the file that contains the profile
    """
    with open(name, "rb") as fp:
        profile = pickle.load(fp)
    return profile

def find_scales(profile):
    """
    Find the relevant spatial scales in an input clustering profile

    Input:
    profile, the clustering profile to search
    """
    
    # First order spatial scales
    peaks = find_peaks(profile + [0], prominence=0)
    scales = [(peaks[0][i]+1, peaks[1]['prominences'][i]) for i in range(len(peaks[0]))]

    # Second order spatial scales above 0
    derivative = np.gradient(profile)
    der_max = find_peaks(derivative+[0], prominence=0)
    near_scales_pos = [(der_max[0][i]+1, -der_max[1]['prominences'][i]) for i in range(len(der_max[0])) if derivative[der_max[0][i]] < 0]

    # Second order spatial scales below 0
    inv_derivative = [-d for d in derivative]
    der_min = find_peaks(inv_derivative+[0], prominence=0)
    near_scales_neg = [(der_min[0][i]+1, -der_min[1]['prominences'][i]) for i in range(len(der_min[0])) if derivative[der_min[0][i]] > 0]

    return near_scales_pos + near_scales_neg + scales

def visualize(x, y, ax):
    """
    Visualize an input pointset

    Input:
    x, the x coordinates of the point set
    y, the y coordinates of the point set
    ax, the axes object to plot the points on
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(x, y, s=1, c="black")
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_aspect('equal')
    
def visualize_fullprofile(fullp, scales, ax):
    """
    Visualize an input clustering profile

    Input:
    fullp, the profile to plot
    scales, the dimension of the profile
    ax, the axes object to plot the profile on
    """
    ax.plot(scales, fullp, c='black', lw=1)
    ax.set_xscale("log")

def create_plots(pointset_name, ax_points, ax_profile, ax_der=None):
    """
    Compute relevant spatial scales and visualize the point set, the clustering profile, and optionally the derivative of the clustering profile

    Input:
    pointset_name, the name of the files containing the data and clustering profile\
    ax_point, the axes object to visualize the point set on 
    ax_profile, the axes object to visualize the clustering profile on
    ax_der, the axes object to visualize the derivative of the profile on
    """

    x, y = data_generator.read_data("data/" + pointset_name)
    fullp = read_full_profile("profiles/fullprofile_" + pointset_name)

    derp = np.gradient(fullp)
    scales = find_scales(fullp)
    
    # Print the relevant spatial scales: first order spatial scales have positive prominence, second order - negative prominence
    print([scale for scale in scales])
    
    visualize(x, y, ax_points)
    visualize_fullprofile(fullp, range(1,120), ax_profile)

    # Add the relevant spatial scales to the visualizations
    ax_profile.plot([scale[0] for scale in scales if scale[1] > 0], [fullp[scale[0]-1] for scale in scales if scale[1] > 0], '.', markerfacecolor='red', markeredgecolor='red')
    ax_profile.plot([scale[0] for scale in scales if scale[1] <= 0], [fullp[scale[0]-1] for scale in scales if scale[1] <= 0], '.', markerfacecolor='blue', markeredgecolor='blue')

    if ax_der:
        ax_der.axhline(y = 0, color = 'grey', linestyle = '--', lw=1)
        visualize_fullprofile(derp, range(1,120), ax_der)
        ax_der.plot([scale[0] for scale in scales if scale[1] > 0], [derp[scale[0]-1] for scale in scales if scale[1] > 0], '.', markerfacecolor='red', markeredgecolor='red')
        ax_der.plot([scale[0] for scale in scales if scale[1] <= 0], [derp[scale[0]-1] for scale in scales if scale[1] <= 0], '.', markerfacecolor='blue', markeredgecolor='blue')

def main():
    fig1, ax1 = plt.subplots(2, 2, sharey = 'col', sharex='col')
    fig1.set_size_inches(4, 4)
    fig2, ax2 = plt.subplots(2, 2, sharey = True, sharex=True)
    fig2.set_size_inches(5, 3)
    fig3, ax3 = plt.subplots(2, 2, sharey = True, sharex=True)
    fig3.set_size_inches(5, 3)
    
    create_plots("uniform", ax1[0][0], ax2[0][0], ax3[0][0])
    ax2[0][0].set_title("(a)", fontsize=7, loc='left', pad=3)
    ax2[0][0].set_ylabel("$M_P$", fontsize=7)
    ax2[0][0].tick_params(labelsize=7)

    create_plots("small", ax1[0][1], ax2[0][1], ax3[0][1])
    ax2[0][1].set_title("(b)", fontsize=7, loc='left', pad=3)
    ax2[0][1].tick_params(labelsize=7)

    create_plots("med", ax1[1][0], ax2[1][0], ax3[1][0])
    ax2[1][0].set_title("(c)", fontsize=7, loc='left', pad=3)
    ax2[1][0].set_xlabel("$\sigma$", fontsize=7)
    ax2[1][0].set_ylabel("$M_P$", fontsize=7)
    ax2[1][0].tick_params(labelsize=7)

    create_plots("multi", ax1[1][1], ax2[1][1], ax3[1][1])
    ax2[1][1].set_title("(d)", fontsize=7, loc='left', pad=3)
    ax2[1][1].set_xlabel("$\sigma$", fontsize=7)
    ax2[1][1].tick_params(labelsize=7)
        
    plt.show()

if __name__ == "__main__":
    main()