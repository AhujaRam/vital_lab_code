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
    # Define image labels as a list
    image_labels = ["Image 1", "Image 2", "Image 3"]
    
    # Create the plot
    plt.imshow(rdm_data)
    plt.yticks(np.arange(3))
    plt.xlabel("Image", fontsize=15)
    plt.ylabel("Image", fontsize=15)
    
    
    # plot with images
    plt.yticks(np.arange(len(image_labels)), image_labels)
    plt.xticks(np.arange(len(image_labels)), image_labels)
    
    plt.title("RDM", fontsize=20)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Dissimilarity', fontsize=15)
    plt.show()
