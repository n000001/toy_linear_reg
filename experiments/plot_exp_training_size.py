# -*- coding: utf-8 -*-
"""
@author: Nina 

Plots the errors and learning time of the simple and lean_all methods 
computed according to the size of the training set. 
Saves the plots in the 'figures' directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Graphic settings
sns.set_style("darkgrid")
sns.set_context("paper")


def plot_learn_all(file_path) :
    """
    Plot and save two figures: MSE and training time 
    according to the size of the training set, for Ridge regression,
    Matching Pursuit and Orthogonal Matching Pursuit methods.
    
    Parameters
    ----------
    file_path : str
        path of file to load the performance data from
    """
    #Loading performance from the file
    file = np.load(file_path)
    N_learn_all = file["N"]
    valid_error_learn_all = file["valid_error"]
    train_error_learn_all = file["train_error"]
    learning_time_learn_all = file["learning_time"]
    
    #Defining suffix for image name to save the figure at
    if file_path == "exp_training_size_learn_all.npz" :
        suffix = ""
    elif file_path == "exp_training_size_learn_all_2.npz" :
        suffix = "_2"
    elif file_path == "exp_training_size_learn_all_2_12.npz" :
        suffix = "_2_12"
    else : 
        suffix = "_unknown"
    
    #Define color palette for the plots
    palette = ['red', 'seagreen', 'slateblue']
    
    #Error plot of the 3 methods according to the size of training set
    fig, ax = plt.subplots()
    for i, method in enumerate(["Ridge", "MP", "OMP"]) :
        ax.plot(N_learn_all, valid_error_learn_all[:, i], 
                label=method + ' - Validation', color=palette[i])
        ax.plot(N_learn_all, train_error_learn_all[:, i],
                '--', label=method + ' - Train', color=palette[i])
    ax.set_title("MSE according to the size of the training set")
    ax.set_xlabel("Size of the training set")
    ax.set_ylabel("Mean squared error")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(r"../figures/mse_learn_all" + suffix + ".png")

    #Learning time plot of the 3 methods according to the size of training set
    fig, ax = plt.subplots()
    for i, method in enumerate(["Ridge", "MP", "OMP"]) :
        ax.plot(N_learn_all, learning_time_learn_all[:, i]*1000,
                label=method, color=palette[i])
    ax.set_title("Learning time according to the size of the training set")
    ax.set_xlabel("Size of the training set")
    ax.set_ylabel("Learning time (milliseconds)")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(r"../figures/learning_time_learn_all" + suffix + ".png")
    
    
if __name__ == '__main__':
        
    #Loading results of error computation in different settings
    #From exp_training_size.npz file saved in current directory
    file = np.load("exp_training_size.npz")
    N = file["N"]
    valid_error = file["valid_error"]
    train_error = file["train_error"]
    learning_time = file["learning_time"]
    
    #Define color palette for the plots
    palette = ["sienna", "orange", "green", "darkviolet"]
    
    #Only the constant regression methods (Least squares too high we can't see well)
    fig, ax = plt.subplots()
    for i, method in enumerate(["Mean", "Median", "Majority"]) :
        ax.plot(N, valid_error[:, i], label=method + ' - Validation', 
                color=palette[i])
        ax.plot(N, train_error[:, i], '--', label=method + ' - Train', 
                color=palette[i])
    ax.set_title("MSE according to the size of the training set")
    ax.set_xlabel("Size of the training set")
    ax.set_ylabel("Mean squared error")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(r"../figures/mse_cst.png")
    
    #Least squares methods only
    fig, ax = plt.subplots()
    ax.plot(N, valid_error[:, 3], label="Least squares" + ' - Validation', color=palette[3])
    ax.plot(N, train_error[:, 3], '--', label="Least squares" + ' - Train', color=palette[3])
    ax.legend(frameon=True)
    ax.set_title("MSE according to the size of the training set")
    ax.set_xlabel("Size of the training set")
    ax.set_ylabel("Mean squared error")
    ax.semilogy()
    plt.tight_layout()
    plt.savefig(r"../figures/mse_least_squares.png")
    
    #Everything together, constant estimators and LS
    fig, ax = plt.subplots()
    for i, method in enumerate(["Mean", "Median", "Majority", "Least squares"]) :
        ax.plot(N, valid_error[:, i], label=method + ' - Validation', 
                color=palette[i])
        ax.plot(N, train_error[:, i], '--', label=method + ' - Train', 
                color=palette[i])
    ax.set_title("MSE according to the size of the training set")
    ax.set_xlabel("Size of the training set")
    ax.set_ylabel("Mean squared error")
    ax.legend(frameon=True)
    ax.semilogy()
    plt.tight_layout()
    plt.savefig(r"../figures/mse_all.png")
    
    #Training time: everything together, constant estimators and LS
    style = ['-', ':', ':', '-']
    transparency = [0.5, 0.5, 0.5, 1]
    fig, ax = plt.subplots()
    for i, method in enumerate(["Mean", "Median", "Majority", "Least squares"]) :
        ax.plot(N, learning_time[:, i]*1000, label=method, color=palette[i],
                linestyle=style[i])
    ax.set_title("Learning time evolution according to the size of the training set") 
    ax.set_xlabel("Size of the training set")
    ax.set_ylabel("Learning time (milliseconds)")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(r"../figures/learning_time.png")
    
    
    #Ploting MSE and training time according to the size of the training set
    #for Ridge, Matching Pursuit and Orthogonal Matching Pursuit
    plot_learn_all("exp_training_size_learn_all.npz")
    #----Optional: comparison with the other data sets provided on Ametice:
    #YearPredictionMSD_2_100.npz and YearPredictionMSD_2_12_100.npz 
    #which contain data from the same songs but with a larger number of 
    #descriptive variables
    plot_learn_all("exp_training_size_learn_all_2.npz")
    plot_learn_all("exp_training_size_learn_all_2_12.npz")
    
    plt.close("all") 
    #The figures are closed to avoid having to much figures open
    #They can be accessed in the figures directory since they have been saved 

    
