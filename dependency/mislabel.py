#!/usr/bin/env python
# coding: utf-8
# %%
def randomly_misclassify_labels(y1):
    import numpy as np
    #Find the unique class labels
    unique_labels = np.unique(y1) # e.g [0 1] 
    
    # Returns the total number of samples.
    num_samples = len(y1) 
    
    # Count the no. of occurrences of each class label in y1 
    class_counts = np.bincount(y1) #e.g. for 15 len = [11 4]

    # Returns an array for shuffling the class labels (0 to num_samples -1)
    # Create an array of shuffled indices
    shuffled_indices = np.arange(num_samples) #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]

    # Random shuffle of the indices, based on original value counts of each label
    np.random.shuffle(shuffled_indices)  
    
    # Create an array to store randomly misclassified labels
    randomly_misclassified = np.zeros_like(y1) 
    
    # Initialise an index variable to help monitor position in the shuffled indices
    index = 0
    
    # For-loop to iterate through unique labels and assign them while maintaining counts
    for class_label in unique_labels:
        count = class_counts[class_label] #Retrieves no. of occurences for the current class label
        randomly_misclassified[shuffled_indices[index:index+count]] = class_label #Assign the current class label to the randomly shuffled indices
        index += count #Increase index, which moves to next shuffled indices for the next class label
    
    return randomly_misclassified

# %%
