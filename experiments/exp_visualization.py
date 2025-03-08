# -*- coding: utf-8 -*-
"""
@author: Nina

Builds the histogram of the years of the songs from the training set and
export the figure to the image file hist_train.png.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from funcs.data_utils import load_data

#Graphics settings
sns.set_style("darkgrid")
sns.set_context("paper")

#Import data
X_labeled, y_labeled, X_unlabeled = load_data(
        "../data/YearPredictionMSD_100.npz")

#Plot and save in image directory
fig, ax = plt.subplots()
ax.hist(y_labeled, bins=(2011-1922), edgecolor='white')
ax.set_xlabel('Year')
ax.set_ylabel('Count')
plt.savefig(r"../figures/hist_year.png") 
plt.tight_layout()

plt.close("all")
