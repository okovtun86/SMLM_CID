# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 18:00:28 2024

@author: Oleg Kovtun
"""

#%%
### install the required libraries
###pip install numpy matplotlib scipy scikit-learn seaborn pandas

#%%
### Python implementation of coordinate-based single-molecule colocalization index in 2D [ https://doi.org/10.1038/s41598-022-08746-4 ]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


#%%

def CI_index(x1, y1, x2, y2, LP1, LP2):
    # LOCAL-DENSITY
    CIX = {}
    
    for channel in range(1, 3):
        if channel == 1:
            x = x1
            y = y1
            LP = LP1
        else:
            x = x2
            y = y2
            LP = LP2
        
        # Find NND distance
        tree = KDTree(np.column_stack((x, y)))
        D, IDX = tree.query(np.column_stack((x, y)), k=2)
        MNND = np.mean(D[:, 1])
        
        # Save NND and MNND in output dictionary
        CIX[channel] = {}
        CIX[channel]['NND_each_loc'] = D[:, 1]
        CIX[channel]['MNND'] = MNND
        
        # Determine effective resolution as search radius
        effective_resolution = np.sqrt((MNND ** 2) + (LP ** 2))
        CIX[channel]['effective_resolution'] = effective_resolution
        
        # Find all localizations in search radius
        IDR = tree.query_ball_point(np.column_stack((x, y)), effective_resolution)
        
        # Determine local density for each localization
        local_density = np.array([len(neighbors) for neighbors in IDR])
        mean_local_density = np.mean(local_density)
        
        # Save local density in output dictionary
        CIX[channel]['LD_each_loc'] = local_density
        CIX[channel]['MLD'] = mean_local_density
    
    # CO-LOCALIZATION INDEX
    for active_channel in range(1, 3):
        if active_channel == 1:
            localizations_assayX = x1
            localizations_assayY = y1
            localizations_otherX = x2
            localizations_otherY = y2
            other_channel = 2
        else:
            localizations_assayX = x2
            localizations_assayY = y2
            localizations_otherX = x1
            localizations_otherY = y1
            other_channel = 1
        
        mean_local_density_other = CIX[other_channel]['MLD']
        effective_resolution = CIX[other_channel]['effective_resolution']
        
        # Find all neighbors in the opposing channel within the search radius
        tree_other = KDTree(np.column_stack((localizations_otherX, localizations_otherY)))
        IDQ = tree_other.query_ball_point(np.column_stack((localizations_assayX, localizations_assayY)), effective_resolution)
        
        # Determine local density of all localizations in opposing channel
        opposing_local_density = np.array([len(neighbors) for neighbors in IDQ])
        
        # Normalize to mean local density to obtain co-localization index
        CI_index_all_localization = opposing_local_density / (mean_local_density_other - 1)
        Mean_CI_ROI = np.mean(CI_index_all_localization)
        
        # Save in output dictionary
        CIX[active_channel]['opposing_LD_each_loc'] = opposing_local_density
        CIX[active_channel]['CI_each_loc'] = CI_index_all_localization
        CIX[active_channel]['Mean_CI'] = Mean_CI_ROI
    
    return CIX

# Fill in name of Test data set file
filename_csv = 'Microtubules 0 degrees rotation.csv'

# choose to plot maps and save data (0 if no, 1 if yes)
plot_maps = 1
save_results = 1

#%%

# load data
Data = pd.read_csv(filename_csv)

# extract channels
channel_1 = Data['Channel'] == 1
channel_2 = Data['Channel'] == 2

x1 = Data['X (nm)'][channel_1].to_numpy()
y1 = Data['Y (nm)'][channel_1].to_numpy()
x2 = Data['X (nm)'][channel_2].to_numpy()
y2 = Data['Y (nm)'][channel_2].to_numpy()
LP1 = Data['Precision (nm)'][channel_1].mean()
LP2 = Data['Precision (nm)'][channel_2].mean()

# Co-localization
CIX = CI_index(x1, y1, x2, y2, LP1, LP2)

# Plot maps
if plot_maps == 1:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    plt.style.use('dark_background')
    # Plot overlay channels
    
    fig_merge, ax_merge = plt.subplots(dpi=300)
    ax_merge.scatter(x1, y1, s=3, c='cyan', alpha=0.3, label='Channel 1')
    ax_merge.scatter(x2, y2, s=3, c='magenta', alpha=0.3, label='Channel 2')
    ax_merge.legend()
    ax_merge.set_aspect('equal')
    ax_merge.axis('off')
    
    # Plot co-localization maps
    fig_coloc1, ax_coloc1 = plt.subplots(dpi=300)
    sm = ScalarMappable(cmap='CMRmap')
    sm.set_array([])
    ax_coloc1.scatter(
        x1, y1, s=3, c=CIX[1]['CI_each_loc'], cmap='CMRmap', vmin=0, vmax=2)
    fig_coloc1.colorbar(sm, label='Colocalization Index')
    ax_coloc1.set_aspect('equal')
    ax_coloc1.axis('off')
    fig_coloc1.suptitle('Colocalization map channel 1')
    
    fig_coloc2, ax_coloc2 = plt.subplots(dpi=300)
    sm = ScalarMappable(cmap='CMRmap')
    sm.set_array([])
    ax_coloc2.scatter(
        x2, y2, s=3, c=CIX[2]['CI_each_loc'], cmap='CMRmap', vmin=0, vmax=2)
    fig_coloc2.colorbar(sm, label='Colocalization Index')
    ax_coloc2.set_aspect('equal')
    ax_coloc2.axis('off')
    fig_coloc2.suptitle('Colocalization map channel 2')

    # Save plots (if desired)
    if save_results == 1:
        fig_merge.savefig('Merge.png',dpi=300)
        fig_coloc1.savefig('Coloc map channel 1.png',dpi=300)
        fig_coloc2.savefig('Coloc map channel 2.png',dpi=300)
        


print('Done')

#%%