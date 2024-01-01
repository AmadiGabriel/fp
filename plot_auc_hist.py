#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %%
# Function to plot ROC curves
def plot_roc_curves(roc_data, ax):
    # Create an array of false positive rate values for the mean ROC curve plot
    mean_fpr = np.linspace(0, 1, 100)

    # Lists to store true positive rates (tprs) and AUC values (aucs) for each fold
    tprs = []
    aucs = []

    # Iterate over each fold
    for fold in range(5):
        fpr_column = f'fpr_{fold}'
        tpr_column = f'tpr_{fold}'
        
        # Retrieve FPR and TPR values from the DataFrame
        fpr = roc_data[fpr_column]
        tpr = roc_data[tpr_column]
        
        # Interpolate true positive rates for the mean ROC calculation, set TPR at FPR=0 to 0.0
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        
        # Append the interpolated TPR values to the tprs list and AUC value to the aucs list
        tprs.append(interp_tpr)
        aucs.append(np.trapz(interp_tpr, mean_fpr))

    # Calculate the mean true positive rate across all folds
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    # Plot the mean ROC curve on the axis
    ax.plot(
        mean_fpr,
        mean_tpr,
        lw=0.2,
        ls='-',
        color='k' 
    )


# %%
# Function to plot Main ROC curves
def plot_roc_curves_main(roc_data, ax):
    # Create an array of false positive rate values for the mean ROC curve plot
    mean_fpr = np.linspace(0, 1, 100)

    # Initialise lists to store true positive rates (tprs) and AUC values (aucs) for each fold
    tprs = []
    aucs = []

    # Iterate over each fold
    for fold in range(5):
        fpr_column = f'fpr_{fold}'
        tpr_column = f'tpr_{fold}'
        
        # Retrieve FPR and TPR values from the DataFrame
        fpr = roc_data[fpr_column]
        tpr = roc_data[tpr_column]
        
        # Interpolate true positive rates for the mean ROC calculation, set TPR at FPR=0 to 0.0
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        
        # Append the interpolated TPR values to the tprs list and AUC value to the aucs list
        tprs.append(interp_tpr)
        aucs.append(np.trapz(interp_tpr, mean_fpr))

    # Calculate the mean true positive rate across all folds
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    # Plot the mean ROC curve on the axis
    ax.plot(
        mean_fpr,
        mean_tpr,
        color='black', ls='-',  linewidth=2.5
    )


# %%
# Function to plot ROC curves for each fold
def plot_roc_curves_folds(roc_data, ax):
    # Iterate over each fold
    for fold in range(5):
        fpr_column = f'fpr_{fold}'
        tpr_column = f'tpr_{fold}'
        
        # Retrieve FPR and TPR values from the DataFrame
        fpr = roc_data[fpr_column]
        tpr = roc_data[tpr_column]
        
        # Plot the ROC curve for the current fold
        ax.plot(
            fpr.values,  # Convert Series to NumPy array
            tpr.values,  # Convert Series to NumPy array
            lw=1,
            color='black',
            alpha=0.8,
            ls=':'
        )


# %%
prob = 'FD' # Change to FP or FPs as necessary

file_main = f'lgbm_fi_ftpr_Problem_{prob}.csv' 
df_main = pd.read_csv(file_main)

plt.rc('font', family='Times New Roman', size=14)
num_files = 90  # Number of random auc computed

# Create a list of file names
file_names = [f'lgbm_fi_ftpr_Problem_{prob}_{i}.csv' for i in range(1, num_files + 1)] 

# Create an empty list to store DataFrames
dfs = []

# Load each CSV file into a DataFrame and append to the list
for file_name in file_names:
    df = pd.read_csv(file_name)
    dfs.append(df)


# Create a plot figure and axis using matplotlib
fig, ax = plt.subplots(figsize=(6, 6))

# Call the function for each DataFrame
plot_roc_curves_main(df_main, ax)

# Call the function for each DataFrame
for df in dfs:
    plot_roc_curves(df, ax)
    

# Customize plot attributes, labels, and title
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
)

ax.plot(color='black', ls='-', linewidth=0.5)

# Set the x and y axis labels
ax.set_xlabel("1 - Specificity", fontsize=14)
ax.set_ylabel("Sensitivity", fontsize=14)

plt.grid(lw=0.7)

# Set the x and y axis limits
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])

# Set the x and y axis tick intervals to 0.2
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax.plot([0, 1], [0, 0], color='grey')
ax.plot([1, 1], [0, 1], color='grey')

# Add graduation marks on left and right y-axis
ax.tick_params(axis='y', which='both', length=7, width=2, direction='in', labelsize=14, labelcolor='k',
               right=True, left=True)

plt.text(0.5, -0.25, '(a)', fontsize=14, ha='center', va='center')

# Display the plot
plt.show()

b = f'lgbm_auc_random_Problem_{prob}.png' 
fig.savefig(b, dpi=200, bbox_inches='tight', transparent=True)


# %%
dfs = []  # Create an empty list to store dataframes

for i in range(1, 91):
    filename = f'lgbm_fi_Problem_{prob}_{i}.csv'
    pattern = pd.read_csv(filename)
    dfs.append(pattern)

# Concatenate the dataframes in the list into a single dataframe
combined_df = pd.concat(dfs, ignore_index=True)


# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# Set the font family and size for the entire plot
font = {'family': 'Times New Roman', 'size': 14}
mpl.rc('font', **font)

# Set the font properties for tick labels
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)

column_name = 'AUC_ROC'  

other_df = pd.read_csv(f'lgbm_fi_Problem_{prob}.csv')

# Create the distribution plot using matplotlib's hist function
plt.figure(figsize=(10, 6))
plt.hist(combined_df[column_name], alpha=0.6, edgecolor='black', color='grey')
plt.xlabel('AUC Score')
plt.ylabel('Frequency')

# Set x-axis ticks with an interval of 0.1
plt.xticks([i / 10 for i in range(11)])
# plt.yticks([j / 0.2 for j in range(81)])

# Set x-axis limits
plt.xlim(0, 1)
plt.ylim(0, 24)
plt.grid()

auc_true = other_df.iloc[0,4]

plt.scatter([auc_true], [0.15], marker='*', color='k', s=100)

plt.text(0.5, -4.5, '(b)', fontsize=14, ha='center', va='center')


b = f'Hist_Problem_{prob}.png'
plt.savefig(b, dpi=200, bbox_inches='tight', transparent=True)

plt.show()


# %%
