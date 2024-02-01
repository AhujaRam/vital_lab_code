import os
import numpy as np
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from nilearn import image, datasets, plotting, masking
import pandas as pd
from scipy.spatial import distance



def plot_rdm(rdm_data, image_labels=None, title="RDM", cmap="viridis", xlabel="Image", ylabel="Image", cbar_label="Dissimilarity"):    
        
    # Create the plot
    plt.imshow(rdm_data)
    plt.xlabel("Image", fontsize=15)
    plt.ylabel("Image", fontsize=15)
    
    plt.title("RDM", fontsize=20)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Dissimilarity', fontsize=15)
    plt.show()
