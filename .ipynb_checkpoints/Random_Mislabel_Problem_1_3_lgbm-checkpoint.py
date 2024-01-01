#!/usr/bin/env python
# coding: utf-8
# %%
# Loading required libraries
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
random_seed = 1


# %%
# List to store data frames
dfs = []

# Import all flight data
for i in range(1, 11):
    filename = f'eng{i}.csv'
    df = pd.read_csv(filename)
    dfs.append(df)

# Concatenate data frames
df_d = pd.concat(dfs, ignore_index=True).fillna(0.1)

# Drop features VIF >> 5
df_d = df_d[df_d.columns.drop(['L_-b/2a True Airspeed (knots)','L_-b/2a CHT 3 (deg C)',
                               'D_-b/2a Oil Pressure (PSI)','L_-b/2a CHT 6 (deg C)',
                              'C_-b/2a Barometer Setting (inHg)','TO_-b/2a Barometer Setting (inHg)',
                               'L_-b/2a Barometer Setting (inHg)','D_-b/2a Barometer Setting (inHg)'])]

# %%
#Separate classes into 3
Xy = df_d

#Class 0
#------------------------------#
Xy0 = Xy[Xy.Fault == 0]
y0 = Xy0['Fault']
X0 = Xy0.drop('Fault', axis=1)

#Class 1
#------------------------------#
Xy1 = Xy[Xy.Fault == 1]
y1 = Xy1['Fault']
X1 = Xy1.drop('Fault', axis=1)

#Class 2
#------------------------------#
Xy2 = Xy[Xy.Fault == 2]
Xy2['Fault'].replace({2: 1}, inplace = True)
y2 = Xy2['Fault']
X2 = Xy2.drop('Fault', axis=1)

# %%
# Split Class 0 into 5 folds: Fold 1
X_train, X_test_1, y_train, y_test_1 = train_test_split(X0, y0, test_size=0.2, random_state=random_seed)
cv_0_0 = pd.concat([X_test_1,y_test_1], axis=1)

# Split Class 0 into 5 folds: Fold 2
X_train, X_test_2, y_train, y_test_2 = train_test_split(X_train,y_train, test_size=0.25, random_state=random_seed)
cv_0_1 = pd.concat([X_test_2,y_test_2], axis=1) 

# Split Class 0 into 5 folds: Fold 3
X_train, X_test_3, y_train, y_test_3 = train_test_split(X_train,y_train, test_size=0.33, random_state=random_seed)
cv_0_2 = pd.concat([X_test_3,y_test_3], axis=1) 

# Split Class 0 into 5 folds: Fold 4 and 5
X_train, X_test_4, y_train, y_test_4 = train_test_split(X_train,y_train, test_size=0.50, random_state=random_seed)
cv_0_3 = pd.concat([X_test_4,y_test_4], axis=1) 
cv_0_4 = pd.concat([X_train,y_train], axis=1) 


# %%
# Split Class 1 into 5 folds: Fold 1
X_train, X_test_1, y_train, y_test_1 = train_test_split(X1, y1, test_size=0.2, random_state=random_seed)
cv_1_0 = pd.concat([X_test_1,y_test_1], axis=1)

# Split Class 1 into 3 folds: Fold 2
X_train, X_test_2, y_train, y_test_2 = train_test_split(X_train,y_train, test_size=0.25, random_state=random_seed)
cv_1_1 = pd.concat([X_test_2,y_test_2], axis=1) 

# Split Class 1 into 3 folds: Fold 3
X_train, X_test_3, y_train, y_test_3 = train_test_split(X_train,y_train, test_size=0.33, random_state=random_seed)
cv_1_2 = pd.concat([X_test_3,y_test_3], axis=1) 

# Split Class 1 into 3 folds: Fold 4 and 5
X_train, X_test_4, y_train, y_test_4 = train_test_split(X_train,y_train, test_size=0.50, random_state=random_seed)
cv_1_3 = pd.concat([X_test_4,y_test_4], axis=1) 
cv_1_4 = pd.concat([X_train,y_train], axis=1) 

# %%
# Split Class 2 into 5 folds: Fold 1
X_train, X_test_1, y_train, y_test_1 = train_test_split(X2, y2, test_size=0.2, random_state=random_seed)
cv_2_0 = pd.concat([X_test_1,y_test_1], axis=1)

# Split Class 2 into 5 folds: Fold 2
X_train, X_test_2, y_train, y_test_2 = train_test_split(X_train,y_train, test_size=0.25, random_state=random_seed)
cv_2_1 = pd.concat([X_test_2,y_test_2], axis=1) 

# Split Class 2 into 5 folds: Fold 3
X_train, X_test_3, y_train, y_test_3 = train_test_split(X_train,y_train, test_size=0.33, random_state=random_seed)
cv_2_2 = pd.concat([X_test_3,y_test_3], axis=1) 

# Split Class 2 into 5 folds: Fold 4 and 5
X_train, X_test_4, y_train, y_test_4 = train_test_split(X_train,y_train, test_size=0.50, random_state=random_seed)
cv_2_3 = pd.concat([X_test_4,y_test_4], axis=1) 
cv_2_4 = pd.concat([X_train,y_train], axis=1)

# %%
#Set up parallel processing for loops
import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# %%
def randomly_misclassify_labels(y1):

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


# %% [markdown]
# ## Problem FD

# %%
# ---------------------------------------------------------------------
# # Uncomment for Random_Mislabelling
# for rand in range(100):   
# ---------------------------------------------------------------------
    
    # Split data into 5 folds
    cv_0 = pd.concat([cv_0_0, cv_2_0], axis = 0) #fold 0  
    cv_1 = pd.concat([cv_0_1, cv_2_1], axis = 0) #fold 1 
    cv_2 = pd.concat([cv_0_2, cv_2_2], axis = 0) #fold 2 
    cv_3 = pd.concat([cv_0_3, cv_2_3], axis = 0) #fold 3 
    cv_4 = pd.concat([cv_0_4, cv_2_4], axis = 0) #fold 4 

    all_df = pd.concat([cv_0, cv_1, cv_2, cv_3, cv_4], axis = 0)
    y_df = all_df['Fault']

# ---------------------------------------------------------------------    
# # Uncomment for Random Mislabelling
#     original_labels = y_df.to_numpy()
#     misclassified_labels = randomly_misclassify_labels(original_labels)
#     y_df = pd.Series(misclassified_labels) 
# ---------------------------------------------------------------------

    x_df = all_df.drop('Fault', axis=1)

# ---------------------------------------------------------------------    
# # Uncomment for Random Mislabelling
#     # Reset the index of x_df and y_df
#     x_df.reset_index(drop=True, inplace=True)
#     y_df.reset_index(drop=True, inplace=True)
# ---------------------------------------------------------------------

    all_df = pd.concat([x_df, y_df], axis = 1)
    all_df.rename(columns={0: "Fault"}, inplace=True)
    split_df = np.array_split(all_df, 5)

    # Save each part as a separate DataFrame
    cv_0, cv_1, cv_2, cv_3, cv_4 = split_df

    # Prepare training and test folds
    k_fold_train_0 = pd.concat([cv_0, cv_1, cv_2, cv_3], axis = 0) #skip fold 4 
    ky0_train = k_fold_train_0['Fault']
    kX0_train = k_fold_train_0.drop('Fault', axis=1)

    ky0_test = cv_4['Fault']  #include fold 4
    kX0_test =cv_4.drop('Fault', axis=1)  #include fold 4

    k_fold_train_1 = pd.concat([cv_0, cv_1, cv_2, cv_4], axis = 0) #skip fold 3
    ky1_train = k_fold_train_1['Fault']
    kX1_train = k_fold_train_1.drop('Fault', axis=1)
    # --------------------------------------------------------------------------
    ky1_test = cv_3['Fault']  #include fold 3
    kX1_test =cv_3.drop('Fault', axis=1)  #include fold 3

    k_fold_train_2 = pd.concat([cv_0, cv_1, cv_3, cv_4], axis = 0) #skip fold 2
    ky2_train = k_fold_train_2['Fault']
    kX2_train = k_fold_train_2.drop('Fault', axis=1)
    # --------------------------------------------------------------------------
    ky2_test = cv_2['Fault']  #include fold 2
    kX2_test =cv_2.drop('Fault', axis=1)  #include fold 2

    k_fold_train_3 = pd.concat([cv_0, cv_2, cv_3, cv_4], axis = 0) #skip fold 1
    ky3_train = k_fold_train_3['Fault']
    kX3_train = k_fold_train_3.drop('Fault', axis=1)
    # --------------------------------------------------------------------------
    ky3_test = cv_1['Fault']  #include fold 1
    kX3_test =cv_1.drop('Fault', axis=1)  #include fold 1


    k_fold_train_4 = pd.concat([cv_1, cv_2, cv_3, cv_4], axis = 0) #skip fold 0
    ky4_train = k_fold_train_4['Fault']
    kX4_train = k_fold_train_4.drop('Fault', axis=1)
    # --------------------------------------------------------------------------
    ky4_test = cv_0['Fault']  #include fold 0
    kX4_test =cv_0.drop('Fault', axis=1)  #include fold 0


    # Initialise LightGradientBoost Classifier
    lgbm = LGBMClassifier(random_state=random_seed, n_jobs=-1)
    
    Xy1 = df_d[df_d.Fault != 1] #Remove faults class 1 to leave only 0 and 2
    Xy1['Fault'].replace({2: 1}, inplace = True) #Replace class 2 labels to 1
    y1 = Xy1['Fault'] # Extract target class

# ---------------------------------------------------------------------
# # Uncomment for Random_Mislabelling
#     original_labels = y1.to_numpy()
#     misclassified_labels = randomly_misclassify_labels(original_labels)
#     y1 = pd.Series(misclassified_labels)
# ---------------------------------------------------------------------    

    X1 = Xy1.drop('Fault', axis=1) #
    #-------------------------------------------------------------


    # Undertake Feature Importance
    lgbm.fit(X1,y1)
    importance_1 = lgbm.feature_importances_


    # Get the scores with the highest importance and make into a list
    df_fi_1 = {
        'Feature': range(len(importance_1)),
        'Score': importance_1
    }
    df_fi_1 = pd.DataFrame(df_fi_1)
    idx_1 = df_fi_1.sort_values('Score', ascending=False) #Feature Importance
    aa_1 = idx_1['Feature'].to_list()

    Feature_Lenght_1= len(aa_1) +1
    result_1, cols_1 = [], []

    for i in range(Feature_Lenght_1):
        cols_1 = aa_1[0:i]
        result_1.append(cols_1)
    result_1_lgbm = result_1[1:]

    if len(result_1_lgbm) < 100:
        num_features = len(result_1_lgbm)
    else:
        num_features = 100


    ## Iteration for Problem FD

    def Problem_FD_lgbm(i):

        idx_1 = result_1_lgbm[i] 
        kX0_traini, kX0_testi = kX0_train.iloc[:, idx_1], kX0_test.iloc[:, idx_1] 
        kX1_traini, kX1_testi = kX1_train.iloc[:, idx_1], kX1_test.iloc[:, idx_1] 
        kX2_traini, kX2_testi = kX2_train.iloc[:, idx_1], kX2_test.iloc[:, idx_1]  
        kX3_traini, kX3_testi = kX3_train.iloc[:, idx_1], kX3_test.iloc[:, idx_1]  
        kX4_traini, kX4_testi = kX4_train.iloc[:, idx_1], kX4_test.iloc[:, idx_1]  


        # Initialise Pandas Series to store data     
        outSeries = pd.Series()
        outSeries['Problem'] = 'FD'
        outSeries['Class'] = '0 and 2'
        outSeries['Model'] = 'lgbm'
        outSeries['FS_Type'] = 'lgbm_fi'

        # Fold 0
        lgbm.fit(kX0_traini,ky0_train) #Fit train,test data
        lgbm_probs = lgbm.predict_proba(kX0_testi)[:,1] #Retain probabilities for the positive outcome only (1)
        lgbm_auc_0 = roc_auc_score(ky0_test, lgbm_probs) # Calculate AUC_ROC
        lgbm_fpr0 ,lgbm_tpr0, _ = roc_curve(ky0_test, lgbm_probs) #get fpr and tpr from roc_curve

        # calculate the g-mean for each threshold on fold 0
        gmeans_lgbm0 = np.sqrt(lgbm_tpr0 * (1-lgbm_fpr0))
        ix_lgbm0 = np.argmax(gmeans_lgbm0)
        lgbm_gmean_0 = gmeans_lgbm0[ix_lgbm0]

        # Fold 1
        lgbm.fit(kX1_traini,ky1_train)
        lgbm_probs = lgbm.predict_proba(kX1_testi)[:,1]
        lgbm_auc_1 = roc_auc_score(ky1_test, lgbm_probs)    
        lgbm_fpr1 ,lgbm_tpr1, _ = roc_curve(ky1_test, lgbm_probs)
        gmeans_lgbm1 = np.sqrt(lgbm_tpr1 * (1-lgbm_fpr1))
        ix_lgbm1 = np.argmax(gmeans_lgbm1)
        lgbm_gmean_1 = gmeans_lgbm1[ix_lgbm1]

        # Fold 2
        lgbm.fit(kX2_traini,ky2_train)
        lgbm_probs = lgbm.predict_proba(kX2_testi)[:,1]
        lgbm_auc_2 = roc_auc_score(ky2_test, lgbm_probs)  
        lgbm_fpr2 ,lgbm_tpr2, _ = roc_curve(ky2_test, lgbm_probs)
        gmeans_lgbm2 = np.sqrt(lgbm_tpr2 * (1-lgbm_fpr2))
        ix_lgbm2 = np.argmax(gmeans_lgbm2)
        lgbm_gmean_2 = gmeans_lgbm2[ix_lgbm2]

        # Fold 3
        lgbm.fit(kX3_traini,ky3_train)
        lgbm_probs = lgbm.predict_proba(kX3_testi)[:,1]
        lgbm_auc_3 = roc_auc_score(ky3_test, lgbm_probs)
        lgbm_fpr3 ,lgbm_tpr3, _ = roc_curve(ky3_test, lgbm_probs)
        gmeans_lgbm3 = np.sqrt(lgbm_tpr3 * (1-lgbm_fpr3))
        ix_lgbm3 = np.argmax(gmeans_lgbm3)
        lgbm_gmean_3 = gmeans_lgbm3[ix_lgbm3]

        # Fold 4
        lgbm.fit(kX4_traini,ky4_train)
        lgbm_probs = lgbm.predict_proba(kX4_testi)[:,1]
        lgbm_auc_4 = roc_auc_score(ky4_test, lgbm_probs)
        lgbm_fpr4 ,lgbm_tpr4, _ = roc_curve(ky4_test, lgbm_probs)
        gmeans_lgbm4 = np.sqrt(lgbm_tpr4 * (1-lgbm_fpr4))
        ix_lgbm4 = np.argmax(gmeans_lgbm4)
        lgbm_gmean_4 = gmeans_lgbm4[ix_lgbm4]

        a = [lgbm_auc_0, lgbm_auc_1, lgbm_auc_2, lgbm_auc_3, lgbm_auc_4]
        outSeries['AUC_ROC'] = round(np.mean(a),3)

        g = [lgbm_gmean_0, lgbm_gmean_1, lgbm_gmean_2, lgbm_gmean_3, lgbm_gmean_4] 
        outSeries['G_Mean'] = round(np.mean(g),3)
        outSeries['Number_of_Features'] = kX4_traini.shape[1]

        X_TO, X_C = kX0_traini.filter(regex='TO_'), kX0_traini.filter(regex='C_') 
        X_D, X_L = kX0_traini.filter(regex='D_'), kX0_traini.filter(regex='L_')

        X_Kurtosis = kX0_traini.filter(regex='Kurtosis')
        X_Skewness = kX0_traini.filter(regex='Skewness')
        X_RMS = kX0_traini.filter(regex='RMS')

        X_MeanF = kX0_traini.filter(regex='MeanF')
        X_MedF = kX0_traini.filter(regex='MedF')
        X_SF = kX0_traini.filter(regex='SF')

        X_a = kX0_traini.filter(regex='_a')
        X_b_2a = kX0_traini.filter(regex='-b/2a')
        X_c = kX0_traini.filter(regex='_c')

        outSeries['Take_Off'] = X_TO.shape[1]
        outSeries['Climb'] = X_C.shape[1]
        outSeries['Descent'] = X_D.shape[1]
        outSeries['Landing'] = X_L.shape[1]

        outSeries['Kurtosis'] = X_Kurtosis.shape[1]
        outSeries['Skewness'] = X_Skewness.shape[1]
        outSeries['RMS'] = X_RMS.shape[1]

        outSeries['MeanFreq'] = X_MeanF.shape[1]
        outSeries['MedFreq'] = X_MedF.shape[1]
        outSeries['SpecFlat'] = X_SF.shape[1]

        outSeries['X_a'] = X_a.shape[1]
        outSeries['X_b_2a'] = X_b_2a.shape[1]
        outSeries['X_c'] = X_c.shape[1]


        return outSeries
    with tqdm_joblib(tqdm(desc="Percentage Completion", total=num_features)) as progress_bar:
        lgbm_fi_iter_Problem_FD = pd.DataFrame(Parallel(n_jobs=-1)(delayed(Problem_FD_lgbm)(i) for i in range(num_features)))
    lgbm_fi_iter_Problem_FD.to_csv('lgbm_fi_iter_Problem_FD.csv', index=False)

# ---------------------------------------------------------------------
## Replace previous line with this for random_mislabelling.
#     lgbm_fi_iter_Problem_FD.to_csv('lgbm_fi_iter_Problem_FD_'+str(rand)+'.csv', index=False) 
# ---------------------------------------------------------------------
    
    # Retrieve index with max AUC value 
    maxrowindex_FD = lgbm_fi_iter_Problem_FD["AUC_ROC"].idxmax()


    # Retrieve row with max index
    maxrow_Problem_FD = lgbm_fi_iter_Problem_FD.loc[maxrowindex_FD]

    # save selected row as new dataframe
    df_lgbm_fi_FD = pd.DataFrame(maxrow_Problem_FD).transpose()
    
    lgbm_fi_FD = 'lgbm_fi_Problem_FD.csv'
# ---------------------------------------------------------------------
## Replace previous line with this for random_mislabelling.
#     lgbm_fi_FD = 'lgbm_fi_Problem_FD_'+str(rand)+'.csv'
# ---------------------------------------------------------------------
    df_lgbm_fi_FD.to_csv(lgbm_fi_FD, index=False)
    
    # Retrieve index of Max Score to Plot AUC 
    kX0_traini, kX0_testi = kX0_train.iloc[:,result_1_lgbm[maxrowindex_FD]], kX0_test.iloc[:,result_1_lgbm[maxrowindex_FD]]  
    kX1_traini, kX1_testi = kX1_train.iloc[:,result_1_lgbm[maxrowindex_FD]], kX1_test.iloc[:,result_1_lgbm[maxrowindex_FD]]  
    kX2_traini, kX2_testi = kX2_train.iloc[:,result_1_lgbm[maxrowindex_FD]], kX2_test.iloc[:,result_1_lgbm[maxrowindex_FD]]  
    kX3_traini, kX3_testi = kX3_train.iloc[:,result_1_lgbm[maxrowindex_FD]], kX3_test.iloc[:,result_1_lgbm[maxrowindex_FD]] 
    kX4_traini, kX4_testi = kX4_train.iloc[:,result_1_lgbm[maxrowindex_FD]], kX4_test.iloc[:,result_1_lgbm[maxrowindex_FD]] 

    # Fold 0
    lgbm.fit(kX0_traini,ky0_train) # Fit training data
    lgbm_probs = lgbm.predict_proba(kX0_testi)[:,1] #Retrain probabilities for the positive outcome only (1)
    lgbm_auc_0 = roc_auc_score(ky0_test, lgbm_probs) # Compute AUC score
    lgbm_fpr0 ,lgbm_tpr0, _ = roc_curve(ky0_test, lgbm_probs)

    # Fold 1
    lgbm.fit(kX1_traini,ky1_train)
    lgbm_probs = lgbm.predict_proba(kX1_testi)[:,1]
    lgbm_auc_1 = roc_auc_score(ky1_test, lgbm_probs)    
    lgbm_fpr1 ,lgbm_tpr1, _ = roc_curve(ky1_test, lgbm_probs)

    # Fold 2
    lgbm.fit(kX2_traini,ky2_train)
    lgbm_probs = lgbm.predict_proba(kX2_testi)[:,1]
    lgbm_auc_2 = roc_auc_score(ky2_test, lgbm_probs)  
    lgbm_fpr2 ,lgbm_tpr2, _ = roc_curve(ky2_test, lgbm_probs)

    # Fold 3
    lgbm.fit(kX3_traini,ky3_train)
    lgbm_probs = lgbm.predict_proba(kX3_testi)[:,1]
    lgbm_auc_3 = roc_auc_score(ky3_test, lgbm_probs)
    lgbm_fpr3 ,lgbm_tpr3, _ = roc_curve(ky3_test, lgbm_probs)

    # Fold 4
    lgbm.fit(kX4_traini,ky4_train)
    lgbm_probs = lgbm.predict_proba(kX4_testi)[:,1]
    lgbm_auc_4 = roc_auc_score(ky4_test, lgbm_probs)
    lgbm_fpr4 ,lgbm_tpr4, _ = roc_curve(ky4_test, lgbm_probs)


    tpr = [lgbm_tpr0, lgbm_tpr1, lgbm_tpr2, lgbm_tpr3, lgbm_tpr4]
    fpr = [lgbm_fpr0, lgbm_fpr1, lgbm_fpr2, lgbm_fpr3, lgbm_fpr4]


    # Save True and False Positive Rates for AUC Plot
    df_fpr, df_tpr = pd.DataFrame(fpr), pd.DataFrame(tpr)
    df_fpr_T, df_tpr_T  =  df_fpr.T, df_tpr.T
    df_fpr_T.columns, df_tpr_T.columns  =['fpr_0','fpr_1','fpr_2','fpr_3','fpr_4'], ['tpr_0','tpr_1','tpr_2','tpr_3','tpr_4']
    df = pd.concat([df_fpr_T, df_tpr_T], axis=1)
#     df.to_csv('lgbm_fi_ftpr_Problem_FD.csv', index=False)
# ---------------------------------------------------------------------
## Replace previous line with this for random_mislabelling.
    df.to_csv('lgbm_fi_ftpr_Problem_FD_'+str(rand)+'.csv', index=False) 
# ---------------------------------------------------------------------

# %% [markdown]
# ## Problem FP

# %%
# ---------------------------------------------------------------------
## Uncomment for Random_Mislabelling
# for rand in range(100):   
# ---------------------------------------------------------------------

    # Feature Importance for Case 2
    Xy2 = df_d[df_d.Fault != 2] #Remove faults class 2 to leave only 0 and 1
    y2 = Xy2['Fault'] # Extract target class
    X2 = Xy2.drop('Fault', axis=1) #Remove target class
    
# ---------------------------------------------------------------------    
## Uncomment for Random Mislabelling
#     original_labels = y2.to_numpy()
#     misclassified_labels = randomly_misclassify_labels(original_labels)
#     y2 = pd.Series(misclassified_labels) 
# ---------------------------------------------------------------------  

    # Undertake Feature Importance
    lgbm.fit(X2,y2)
    importance_1 = lgbm.feature_importances_


    # Get the scores with the highest importance and make into a list
    df_fi_1 = {
        'Feature': range(len(importance_1)),
        'Score': importance_1
    }
    df_fi_1 = pd.DataFrame(df_fi_1)
    idx_1 = df_fi_1.sort_values('Score', ascending=False) #Feature Importance
    aa_1 = idx_1['Feature'].to_list()

    Feature_Lenght_1= len(aa_1) +1
    result_1 = []
    cols_1 = []
    for i in range(Feature_Lenght_1):
        cols_1 = aa_1[0:i]
        result_1.append(cols_1)
    result_2_lgbm = result_1[1:]



    if len(result_2_lgbm) < 100:
        num_features = len(result_2_lgbm)

    else:
        num_features = 100

    # Split data into 5 folds
    cv_0 = pd.concat([cv_0_0, cv_1_0], axis = 0) #fold 0  
    cv_1 = pd.concat([cv_0_1, cv_1_1], axis = 0) #fold 1 
    cv_2 = pd.concat([cv_0_2, cv_1_2], axis = 0) #fold 2 
    cv_3 = pd.concat([cv_0_3, cv_1_3], axis = 0) #fold 3 
    cv_4 = pd.concat([cv_0_4, cv_1_4], axis = 0) #fold 4 

    all_df = pd.concat([cv_0, cv_1, cv_2, cv_3, cv_4], axis = 0)
    y_df = all_df['Fault']

# ---------------------------------------------------------------------    
## Uncomment for Random Mislabelling
#     original_labels = y_df.to_numpy()
#     misclassified_labels = randomly_misclassify_labels(original_labels, misclassification_rate)
#     y_df = pd.Series(misclassified_labels) 
# ---------------------------------------------------------------------    

    x_df = all_df.drop('Fault', axis=1)

# ---------------------------------------------------------------------    
## Uncomment for Random Mislabelling
    # Reset the index of x_df and y_df
#     x_df.reset_index(drop=True, inplace=True)
#     y_df.reset_index(drop=True, inplace=True)
# --------------------------------------------------------------------- 

    all_df = pd.concat([x_df, y_df], axis = 1)
    all_df.rename(columns={0: "Fault"}, inplace=True)
    split_df = np.array_split(all_df, 5)

    # Save each part as a separate DataFrame
    cv_0, cv_1, cv_2, cv_3, cv_4 = split_df


    # Prepare training and validation folds
    k_fold_train_0 = pd.concat([cv_0, cv_1, cv_2, cv_3], axis = 0) #skip fold 4 
    ky0_train = k_fold_train_0['Fault']
    kX0_train = k_fold_train_0.drop('Fault', axis=1)
    
    ky0_test = cv_4['Fault']  #include fold 4
    kX0_test = cv_4.drop('Fault', axis=1)  #include fold 4

    k_fold_train_1 = pd.concat([cv_0, cv_1, cv_2, cv_4], axis = 0) #skip fold 3
    ky1_train = k_fold_train_1['Fault']
    kX1_train = k_fold_train_1.drop('Fault', axis=1)

    ky1_test = cv_3['Fault']  #include fold 3
    kX1_test = cv_3.drop('Fault', axis=1)  #include fold 3

    k_fold_train_2 = pd.concat([cv_0, cv_1, cv_3, cv_4], axis = 0) #skip fold 2
    ky2_train = k_fold_train_2['Fault']
    kX2_train = k_fold_train_2.drop('Fault', axis=1)

    ky2_test = cv_2['Fault']  #include fold 2
    kX2_test = cv_2.drop('Fault', axis=1)  #include fold 2


    k_fold_train_3 = pd.concat([cv_0, cv_2, cv_3, cv_4], axis = 0) #skip fold 1
    ky3_train = k_fold_train_3['Fault']
    kX3_train = k_fold_train_3.drop('Fault', axis=1)
 
    ky3_test = cv_1['Fault']  #include fold 1
    kX3_test = cv_1.drop('Fault', axis=1)  #include fold 1


    k_fold_train_4 = pd.concat([cv_1, cv_2, cv_3, cv_4], axis = 0) #skip fold 0
    ky4_train = k_fold_train_4['Fault']
    kX4_train = k_fold_train_4.drop('Fault', axis=1)

    ky4_test = cv_0['Fault']  #include fold 0
    kX4_test = cv_0.drop('Fault', axis=1)  #include fold 0

    ## Iteration for Problem FP: (0 vs 1)
    def Problem_FP_lgbm(i):

        idx_2 = result_2_lgbm[i] 
        kX0_traini, kX0_testi = kX0_train.iloc[:, idx_2], kX0_test.iloc[:, idx_2] 
        kX1_traini, kX1_testi = kX1_train.iloc[:, idx_2], kX1_test.iloc[:, idx_2] 
        kX2_traini, kX2_testi = kX2_train.iloc[:, idx_2], kX2_test.iloc[:, idx_2]  
        kX3_traini, kX3_testi = kX3_train.iloc[:, idx_2], kX3_test.iloc[:, idx_2]  
        kX4_traini, kX4_testi = kX4_train.iloc[:, idx_2], kX4_test.iloc[:, idx_2]  


        outSeries = pd.Series()
        outSeries['Problem'] = 'FP'
        outSeries['Class'] = '0 and 1'
        outSeries['Model'] = 'lgbm'
        outSeries['FS_Type'] = 'lgbm_fi'

        lgbm.fit(kX0_traini,ky0_train)
        lgbm_probs = lgbm.predict_proba(kX0_testi)[:,1]
        lgbm_auc_0 = roc_auc_score(ky0_test, lgbm_probs)

        lgbm_fpr0 ,lgbm_tpr0, _ = roc_curve(ky0_test, lgbm_probs)
        # calculate the g-mean for each threshold
        gmeans_lgbm0 = np.sqrt(lgbm_tpr0 * (1-lgbm_fpr0))
        ix_lgbm0 = np.argmax(gmeans_lgbm0)
        lgbm_gmean_0 = gmeans_lgbm0[ix_lgbm0]

        lgbm.fit(kX1_traini,ky1_train)
        lgbm_probs = lgbm.predict_proba(kX1_testi)[:,1]
        lgbm_auc_1 = roc_auc_score(ky1_test, lgbm_probs)    
        lgbm_fpr1 ,lgbm_tpr1, _ = roc_curve(ky1_test, lgbm_probs)
        gmeans_lgbm1 = np.sqrt(lgbm_tpr1 * (1-lgbm_fpr1))
        ix_lgbm1 = np.argmax(gmeans_lgbm1)
        lgbm_gmean_1 = gmeans_lgbm1[ix_lgbm1]

        lgbm.fit(kX2_traini,ky2_train)
        lgbm_probs = lgbm.predict_proba(kX2_testi)[:,1]
        lgbm_probs = lgbm_probs
        lgbm_auc_2 = roc_auc_score(ky2_test, lgbm_probs)  
        lgbm_fpr2 ,lgbm_tpr2, _ = roc_curve(ky2_test, lgbm_probs)
        gmeans_lgbm2 = np.sqrt(lgbm_tpr2 * (1-lgbm_fpr2))
        ix_lgbm2 = np.argmax(gmeans_lgbm2)
        lgbm_gmean_2 = gmeans_lgbm2[ix_lgbm2]

        lgbm.fit(kX3_traini,ky3_train)
        lgbm_probs = lgbm.predict_proba(kX3_testi)[:,1]
        lgbm_probs = lgbm_probs
        lgbm_auc_3 = roc_auc_score(ky3_test, lgbm_probs)
        lgbm_fpr3 ,lgbm_tpr3, _ = roc_curve(ky3_test, lgbm_probs)
        gmeans_lgbm3 = np.sqrt(lgbm_tpr3 * (1-lgbm_fpr3))
        ix_lgbm3 = np.argmax(gmeans_lgbm3)
        lgbm_gmean_3 = gmeans_lgbm3[ix_lgbm3]

        lgbm.fit(kX4_traini,ky4_train)
        lgbm_probs = lgbm.predict_proba(kX4_testi)
        lgbm_probs = lgbm_probs[:,1]
        lgbm_auc_4 = roc_auc_score(ky4_test, lgbm_probs)
        lgbm_fpr4 ,lgbm_tpr4, _ = roc_curve(ky4_test, lgbm_probs)
        # calculate the g-mean for each threshold
        gmeans_lgbm4 = np.sqrt(lgbm_tpr4 * (1-lgbm_fpr4))
        ix_lgbm4 = np.argmax(gmeans_lgbm4)
        lgbm_gmean_4 = gmeans_lgbm4[ix_lgbm4]

        a = [lgbm_auc_0, lgbm_auc_1, lgbm_auc_2, lgbm_auc_3, lgbm_auc_4]
        outSeries['AUC_ROC'] = round(np.mean(a),3)

        g = [lgbm_gmean_0, lgbm_gmean_1, lgbm_gmean_2, lgbm_gmean_3, lgbm_gmean_4] 
        outSeries['G_Mean'] = round(np.mean(g),3)
        outSeries['Number_of_Features'] = str(kX4_traini.shape[1])


        X_TO, X_C = kX0_traini.filter(regex='TO_'), kX0_traini.filter(regex='C_')
        X_D, X_L = kX0_traini.filter(regex='D_'), kX0_traini.filter(regex='L_')

        X_Kurtosis = kX0_traini.filter(regex='Kurtosis')
        X_Skewness = kX0_traini.filter(regex='Skewness')
        X_RMS = kX0_traini.filter(regex='RMS')

        X_MeanF = kX0_traini.filter(regex='MeanF')
        X_MedF = kX0_traini.filter(regex='MedF')
        X_SF = kX0_traini.filter(regex='SF')

        X_a = kX0_traini.filter(regex='_a')
        X_b_2a = kX0_traini.filter(regex='-b/2a')
        X_c = kX0_traini.filter(regex='_c')

        outSeries['Take_Off'] = X_TO.shape[1]
        outSeries['Climb'] = X_C.shape[1]
        outSeries['Descent'] = X_D.shape[1]
        outSeries['Landing'] = X_L.shape[1]

        outSeries['Kurtosis'] = X_Kurtosis.shape[1]
        outSeries['Skewness'] = X_Skewness.shape[1]
        outSeries['RMS'] = X_RMS.shape[1]

        outSeries['MeanFreq'] = X_MeanF.shape[1]
        outSeries['MedFreq'] = X_MedF.shape[1]
        outSeries['SpecFlat'] = X_SF.shape[1]

        outSeries['X_a'] = X_a.shape[1]
        outSeries['X_b_2a'] = X_b_2a.shape[1]
        outSeries['X_c'] = X_c.shape[1]


        return outSeries
    with tqdm_joblib(tqdm(desc="Percentage Completion", total=num_features)) as progress_bar:
        lgbm_fi_iter_Problem_FP = pd.DataFrame(Parallel(n_jobs=-1)(delayed(Problem_FP_lgbm)(i) for i in range(num_features)))
    lgbm_fi_iter_Problem_FP.to_csv('lgbm_fi_iter_Problem_FP.csv', index=False)
# ---------------------------------------------------------------------
## Replace previous line with this for random_mislabelling.
    #lgbm_fi_iter_Problem_FP.to_csv('lgbm_fi_iter_Problem_FP_'+str(rand)+'.csv', index=False) 
# ---------------------------------------------------------------------
    
    maxrowindex_FP = lgbm_fi_iter_Problem_FP["AUC_ROC"].idxmax()
    maxrow_Problem_FP = lgbm_fi_iter_Problem_FP.loc[maxrowindex_FP]
    df_lgbm_fi_FP = pd.DataFrame(maxrow_Problem_FP).transpose()
    lgbm_fi_FP = 'lgbm_fi_Problem_FP.csv'
# ---------------------------------------------------------------------
## Replace previous line with this for random_mislabelling.
#     lgbm_fi_FP = 'lgbm_fi_Problem_FP_'+str(rand)+'.csv'
# ---------------------------------------------------------------------
    df_lgbm_fi_FP.to_csv(lgbm_fi_FP, index=False)
    
    kX0_traini, kX0_testi = kX0_train.iloc[:,result_2_lgbm[maxrowindex_FP]], kX0_test.iloc[:,result_2_lgbm[maxrowindex_FP]]  
    kX1_traini, kX1_testi = kX1_train.iloc[:,result_2_lgbm[maxrowindex_FP]], kX1_test.iloc[:,result_2_lgbm[maxrowindex_FP]]  
    kX2_traini, kX2_testi = kX2_train.iloc[:,result_2_lgbm[maxrowindex_FP]], kX2_test.iloc[:,result_2_lgbm[maxrowindex_FP]]  
    kX3_traini, kX3_testi = kX3_train.iloc[:,result_2_lgbm[maxrowindex_FP]], kX3_test.iloc[:,result_2_lgbm[maxrowindex_FP]] 
    kX4_traini, kX4_testi = kX4_train.iloc[:,result_2_lgbm[maxrowindex_FP]], kX4_test.iloc[:,result_2_lgbm[maxrowindex_FP]] 

    lgbm.fit(kX0_traini,ky0_train)
    lgbm_probs = lgbm.predict_proba(kX0_testi)[:,1]
    lgbm_auc_0 = roc_auc_score(ky0_test, lgbm_probs)
    lgbm_fpr0 ,lgbm_tpr0, _ = roc_curve(ky0_test, lgbm_probs)

    lgbm.fit(kX1_traini,ky1_train)
    lgbm_probs = lgbm.predict_proba(kX1_testi)[:,1]
    lgbm_auc_1 = roc_auc_score(ky1_test, lgbm_probs)    
    lgbm_fpr1 ,lgbm_tpr1, _ = roc_curve(ky1_test, lgbm_probs)

    lgbm.fit(kX2_traini,ky2_train)
    lgbm_probs = lgbm.predict_proba(kX2_testi)[:,1]
    lgbm_auc_2 = roc_auc_score(ky2_test, lgbm_probs)  
    lgbm_fpr2 ,lgbm_tpr2, _ = roc_curve(ky2_test, lgbm_probs)

    lgbm.fit(kX3_traini,ky3_train)
    lgbm_probs = lgbm.predict_proba(kX3_testi)[:,1]
    lgbm_auc_3 = roc_auc_score(ky3_test, lgbm_probs)
    lgbm_fpr3 ,lgbm_tpr3, _ = roc_curve(ky3_test, lgbm_probs)

    lgbm.fit(kX4_traini,ky4_train)
    lgbm_probs = lgbm.predict_proba(kX4_testi)[:,1]
    lgbm_auc_4 = roc_auc_score(ky4_test, lgbm_probs)
    lgbm_fpr4 ,lgbm_tpr4, _ = roc_curve(ky4_test, lgbm_probs)


    tpr = [lgbm_tpr0, lgbm_tpr1, lgbm_tpr2, lgbm_tpr3, lgbm_tpr4]
    fpr = [lgbm_fpr0, lgbm_fpr1, lgbm_fpr2, lgbm_fpr3, lgbm_fpr4]


    # Save True and False Positive Rates for AUC Plot
    df_fpr, df_tpr = pd.DataFrame(fpr), pd.DataFrame(tpr)
    df_fpr_T, df_tpr_T  =  df_fpr.T, df_tpr.T
    df_fpr_T.columns, df_tpr_T.columns  =['fpr_0','fpr_1','fpr_2','fpr_3','fpr_4'], ['tpr_0','tpr_1','tpr_2','tpr_3','tpr_4']
    df = pd.concat([df_fpr_T, df_tpr_T], axis=1)
    df.to_csv('lgbm_fi_ftpr_Problem_FP.csv', index=False)
# ---------------------------------------------------------------------
## Replace previous line with this for random_mislabelling.
    #df.to_csv('lgbm_fi_ftpr_Problem_FP_'+str(rand)+'.csv', index=False) 
# ---------------------------------------------------------------------

# %% [markdown]
# # Problem FPs

# %%
# ---------------------------------------------------------------------
## Uncomment for Random_Mislabelling
# for rand in range(100):   
# ---------------------------------------------------------------------

    # Set up training folds using classes 0, 1 and 2
    cv_0 = pd.concat([cv_0_0, cv_1_0, cv_2_0], axis = 0)
    cv_1 = pd.concat([cv_0_1, cv_1_1, cv_2_1], axis = 0)
    cv_2 = pd.concat([cv_0_2, cv_1_2, cv_2_2], axis = 0)
    cv_3 = pd.concat([cv_0_3, cv_1_3, cv_2_3], axis = 0)
    cv_4 = pd.concat([cv_0_4, cv_1_4, cv_2_4], axis = 0)

    all_df = pd.concat([cv_0, cv_1, cv_2, cv_3, cv_4], axis = 0)
    y_df = all_df['Fault']
    
# ---------------------------------------------------------------------    
## Uncomment for Random Mislabelling    
#     original_labels = y_df.to_numpy()
#     misclassified_labels = randomly_misclassify_labels(original_labels)
#     y_df = pd.Series(misclassified_labels) 
# ---------------------------------------------------------------------    
    
    x_df = all_df.drop('Fault', axis=1)

# ---------------------------------------------------------------------    
## Uncomment for Random Mislabelling  
    # Reset the index of x_df and y_df
#     x_df.reset_index(drop=True, inplace=True)
#     y_df.reset_index(drop=True, inplace=True)
# ---------------------------------------------------------------------  

    all_df = pd.concat([x_df, y_df], axis = 1)
    all_df.rename(columns={0: "Fault"}, inplace=True)
    split_df = np.array_split(all_df, 5)

    # Save each part as a separate DataFrame
    cv_0, cv_1, cv_2, cv_3, cv_4 = split_df


    k_fold_train_0 = pd.concat([cv_0, cv_1, cv_2, cv_3], axis = 0) #skip 4
    ky0_train = k_fold_train_0['Fault']
    kX0_train = k_fold_train_0.drop('Fault', axis=1)


    k_fold_train_1 = pd.concat([cv_0, cv_1, cv_2, cv_4], axis = 0) #skip 3
    ky1_train = k_fold_train_1['Fault']
    kX1_train = k_fold_train_1.drop('Fault', axis=1)


    k_fold_train_2 = pd.concat([cv_0, cv_1, cv_3, cv_4], axis = 0) #skip 2
    ky2_train = k_fold_train_2['Fault']
    kX2_train = k_fold_train_2.drop('Fault', axis=1)


    k_fold_train_3 = pd.concat([cv_0, cv_2, cv_3, cv_4], axis = 0) #skip 1
    ky3_train = k_fold_train_3['Fault']
    kX3_train = k_fold_train_3.drop('Fault', axis=1)


    k_fold_train_4 = pd.concat([cv_1, cv_2, cv_3, cv_4], axis = 0) #skip 0
    ky4_train = k_fold_train_4['Fault']
    kX4_train = k_fold_train_4.drop('Fault', axis=1)

    # Set up validation folds using classes 0 and 1
    cv_0 = pd.concat([cv_0_0, cv_1_0], axis = 0)
    cv_1 = pd.concat([cv_0_1, cv_1_1], axis = 0)
    cv_2 = pd.concat([cv_0_2, cv_1_2], axis = 0)
    cv_3 = pd.concat([cv_0_3, cv_1_3], axis = 0)
    cv_4 = pd.concat([cv_0_4, cv_1_4], axis = 0)

    all_df = pd.concat([cv_0, cv_1, cv_2, cv_3, cv_4], axis = 0)
    y_df = all_df['Fault']
    
# ---------------------------------------------------------------------    
## Uncomment for Random Mislabelling   
#     original_labels = y_df.to_numpy()
#     misclassified_labels = randomly_misclassify_labels(original_labels)
#     y_df = pd.Series(misclassified_labels) 
# --------------------------------------------------------------------- 

    x_df = all_df.drop('Fault', axis=1)

# ---------------------------------------------------------------------    
## Uncomment for Random Mislabelling  
    # Reset the index of x_df and y_df
#     x_df.reset_index(drop=True, inplace=True)
#     y_df.reset_index(drop=True, inplace=True)
# ---------------------------------------------------------------------  

    all_df = pd.concat([x_df, y_df], axis = 1)
    all_df.rename(columns={0: "Fault"}, inplace=True)
    split_df = np.array_split(all_df, 5)

    # Save each part as a separate DataFrame
    cv_0, cv_1, cv_2, cv_3, cv_4 = split_df


    k_fold_test_0 = cv_4 #include 4
    ky0_test = k_fold_test_0['Fault']
    kX0_test =k_fold_test_0.drop('Fault', axis=1)

    k_fold_test_1 = cv_3 #include 3
    ky1_test = k_fold_test_1['Fault']
    kX1_test =k_fold_test_1.drop('Fault', axis=1)

    k_fold_test_2 = cv_2 #include 2
    ky2_test = k_fold_test_2['Fault']
    kX2_test =k_fold_test_2.drop('Fault', axis=1)

    k_fold_test_3 = cv_1 #include 1
    ky3_test = k_fold_test_3['Fault']
    kX3_test =k_fold_test_3.drop('Fault', axis=1)

    k_fold_test_4 = cv_0 #include 0
    ky4_test = k_fold_test_4['Fault']
    kX4_test =k_fold_test_4.drop('Fault', axis=1)


    # Feature Importance for Case 3
    Xy3 = df_d
    Xy3['Fault'].replace({2: 1}, inplace = True) #Replace class 2 labels to 1
    y3 = Xy3['Fault']
    X3 = Xy3.drop('Fault', axis=1)

# ---------------------------------------------------------------------    
## Uncomment for Random Mislabelling
#     original_labels = y3.to_numpy()
#     misclassified_labels = randomly_misclassify_labels(original_labels)
#     y3 = pd.Series(misclassified_labels)
# --------------------------------------------------------------------- 

    # Undertake Feature Importance
    lgbm.fit(X3,y3)
    importance_1 = lgbm.feature_importances_

    # Get the scores with the highest importance and make into a list
    df_fi_1 = {
        'Feature': range(len(importance_1)),
        'Score': importance_1
    }
    df_fi_1 = pd.DataFrame(df_fi_1)
    idx_1 = df_fi_1.sort_values('Score', ascending=False) #Feature Importance
    aa_1 = idx_1['Feature'].to_list()

    Feature_Lenght_1= len(aa_1) +1
    result_1 = []
    cols_1 = []
    for i in range(Feature_Lenght_1):
        cols_1 = aa_1[0:i]
        result_1.append(cols_1)
    result_3_lgbm = result_1[1:]

    if len(result_3_lgbm) < 100:
        num_features = len(result_3_lgbm)

    else:
        num_features = 100


    ## Iteration for Problem FPs: (0 vs 1s2)

    def Problem_FPs_lgbm(i):

        idx_3 = result_3_lgbm[i] 
        kX0_traini, kX0_testi = kX0_train.iloc[:, idx_3], kX0_test.iloc[:, idx_3] 
        kX1_traini, kX1_testi = kX1_train.iloc[:, idx_3], kX1_test.iloc[:, idx_3] 
        kX2_traini, kX2_testi = kX2_train.iloc[:, idx_3], kX2_test.iloc[:, idx_3]  
        kX3_traini, kX3_testi = kX3_train.iloc[:, idx_3], kX3_test.iloc[:, idx_3]  
        kX4_traini, kX4_testi = kX4_train.iloc[:, idx_3], kX4_test.iloc[:, idx_3]  

        outSeries = pd.Series()
        outSeries['Problem'] = 'FPs'
        outSeries['Class'] = '0 and 1s2'
        outSeries['Model'] = 'lgbm'
        outSeries['FS_Type'] = 'lgbm_fi'

        lgbm.fit(kX0_traini,ky0_train)
        lgbm_probs = lgbm.predict_proba(kX0_testi)[:,1]
        lgbm_auc_0 = roc_auc_score(ky0_test, lgbm_probs)
        lgbm_fpr0 ,lgbm_tpr0, _ = roc_curve(ky0_test, lgbm_probs)
        gmeans_lgbm0 = np.sqrt(lgbm_tpr0 * (1-lgbm_fpr0))
        ix_lgbm0 = np.argmax(gmeans_lgbm0)
        lgbm_gmean_0 = gmeans_lgbm0[ix_lgbm0]

        lgbm.fit(kX1_traini,ky1_train)
        lgbm_probs = lgbm.predict_proba(kX1_testi)[:,1]
        lgbm_auc_1 = roc_auc_score(ky1_test, lgbm_probs)    
        lgbm_fpr1 ,lgbm_tpr1, _ = roc_curve(ky1_test, lgbm_probs)
        gmeans_lgbm1 = np.sqrt(lgbm_tpr1 * (1-lgbm_fpr1))
        ix_lgbm1 = np.argmax(gmeans_lgbm1)
        lgbm_gmean_1 = gmeans_lgbm1[ix_lgbm1]

        lgbm.fit(kX2_traini,ky2_train)
        lgbm_probs = lgbm.predict_proba(kX2_testi)[:,1]
        lgbm_auc_2 = roc_auc_score(ky2_test, lgbm_probs)  
        lgbm_fpr2 ,lgbm_tpr2, _ = roc_curve(ky2_test, lgbm_probs)
        gmeans_lgbm2 = np.sqrt(lgbm_tpr2 * (1-lgbm_fpr2))
        ix_lgbm2 = np.argmax(gmeans_lgbm2)
        lgbm_gmean_2 = gmeans_lgbm2[ix_lgbm2]

        lgbm.fit(kX3_traini,ky3_train)
        lgbm_probs = lgbm.predict_proba(kX3_testi)[:,1]
        lgbm_auc_3 = roc_auc_score(ky3_test, lgbm_probs)
        lgbm_fpr3 ,lgbm_tpr3, _ = roc_curve(ky3_test, lgbm_probs)
        gmeans_lgbm3 = np.sqrt(lgbm_tpr3 * (1-lgbm_fpr3))
        ix_lgbm3 = np.argmax(gmeans_lgbm3)
        lgbm_gmean_3 = gmeans_lgbm3[ix_lgbm3]

        lgbm.fit(kX4_traini,ky4_train)
        lgbm_probs = lgbm.predict_proba(kX4_testi)[:,1]
        lgbm_auc_4 = roc_auc_score(ky4_test, lgbm_probs)
        lgbm_fpr4 ,lgbm_tpr4, _ = roc_curve(ky4_test, lgbm_probs)
        gmeans_lgbm4 = np.sqrt(lgbm_tpr4 * (1-lgbm_fpr4))
        ix_lgbm4 = np.argmax(gmeans_lgbm4)
        lgbm_gmean_4 = gmeans_lgbm4[ix_lgbm4]

        a = [lgbm_auc_0, lgbm_auc_1, lgbm_auc_2, lgbm_auc_3, lgbm_auc_4]
        outSeries['AUC_ROC'] = round(np.mean(a),3)

        g = [lgbm_gmean_0, lgbm_gmean_1, lgbm_gmean_2, lgbm_gmean_3, lgbm_gmean_4] 
        outSeries['G_Mean'] = round(np.mean(g),3)
        outSeries['Number_of_Features'] = str(kX4_traini.shape[1])


        X_TO,X_C = kX0_traini.filter(regex='TO_'), kX0_traini.filter(regex='C_')
        X_D, X_L = kX0_traini.filter(regex='D_'), kX0_traini.filter(regex='L_')

        X_Kurtosis = kX0_traini.filter(regex='Kurtosis')
        X_Skewness = kX0_traini.filter(regex='Skewness')
        X_RMS = kX0_traini.filter(regex='RMS')

        X_MeanF = kX0_traini.filter(regex='MeanF')
        X_MedF = kX0_traini.filter(regex='MedF')
        X_SF = kX0_traini.filter(regex='SF')

        X_a = kX0_traini.filter(regex='_a')
        X_b_2a = kX0_traini.filter(regex='-b/2a')
        X_c = kX0_traini.filter(regex='_c')

        outSeries['Take_Off'] = X_TO.shape[1]
        outSeries['Climb'] = X_C.shape[1]
        outSeries['Descent'] = X_D.shape[1]
        outSeries['Landing'] = X_L.shape[1]

        outSeries['Kurtosis'] = X_Kurtosis.shape[1]
        outSeries['Skewness'] = X_Skewness.shape[1]
        outSeries['RMS'] = X_RMS.shape[1]

        outSeries['MeanFreq'] = X_MeanF.shape[1]
        outSeries['MedFreq'] = X_MedF.shape[1]
        outSeries['SpecFlat'] = X_SF.shape[1]

        outSeries['X_a'] = X_a.shape[1]
        outSeries['X_b_2a'] = X_b_2a.shape[1]
        outSeries['X_c'] = X_c.shape[1]

        return outSeries
    with tqdm_joblib(tqdm(desc="Percentage Completion", total=num_features)) as progress_bar:
        lgbm_fi_iter_Problem_FPs = pd.DataFrame(Parallel(n_jobs=-1)(delayed(Problem_FPs_lgbm)(i) for i in range(num_features)))
    lgbm_fi_iter_Problem_FPs.to_csv('lgbm_fi_iter_Problem_FPs.csv', index=False)
# ---------------------------------------------------------------------
## Replace previous line with this for random_mislabelling.
    #lgbm_fi_iter_Problem_FPs.to_csv('lgbm_fi_iter_Problem_FPs_'+str(rand)+'.csv', index=False) 
# ---------------------------------------------------------------------

    maxrowindex_FPs = lgbm_fi_iter_Problem_FPs["AUC_ROC"].idxmax()
    
# ---------------------------------------------------------------------
## Comment out for random_mislabelling    
    maxrow_Problem_FPs = lgbm_fi_iter_Problem_FPs.loc[maxrowindex_FPs]
    df_lgbm_fi_FPs = pd.DataFrame(maxrow_Problem_FPs).transpose()
    lgbm_fi_FPs = 'lgbm_fi_Problem_FPs.csv'
# ---------------------------------------------------------------------
## Replace previous line with this for random_mislabelling.
#     lgbm_fi_FPs = 'lgbm_fi_Problem_FPs_'+str(rand)+'.csv'
# ---------------------------------------------------------------------
    df_lgbm_fi_FPs.to_csv(lgbm_fi_FPs, index=False)

    kX0_traini, kX0_testi = kX0_train.iloc[:,result_3_lgbm[maxrowindex_FP]], kX0_test.iloc[:,result_3_lgbm[maxrowindex_FP]]  
    kX1_traini, kX1_testi = kX1_train.iloc[:,result_3_lgbm[maxrowindex_FP]], kX1_test.iloc[:,result_3_lgbm[maxrowindex_FP]]  
    kX2_traini, kX2_testi = kX2_train.iloc[:,result_3_lgbm[maxrowindex_FP]], kX2_test.iloc[:,result_3_lgbm[maxrowindex_FP]]  
    kX3_traini, kX3_testi = kX3_train.iloc[:,result_3_lgbm[maxrowindex_FP]], kX3_test.iloc[:,result_3_lgbm[maxrowindex_FP]] 
    kX4_traini, kX4_testi = kX4_train.iloc[:,result_3_lgbm[maxrowindex_FP]], kX4_test.iloc[:,result_3_lgbm[maxrowindex_FP]] 

    lgbm.fit(kX0_traini,ky0_train)
    lgbm_probs = lgbm.predict_proba(kX0_testi)[:,1]
    lgbm_auc_0 = roc_auc_score(ky0_test, lgbm_probs)
    lgbm_fpr0 ,lgbm_tpr0, _ = roc_curve(ky0_test, lgbm_probs)

    lgbm.fit(kX1_traini,ky1_train)
    lgbm_probs = lgbm.predict_proba(kX1_testi)[:,1]
    lgbm_auc_1 = roc_auc_score(ky1_test, lgbm_probs)    
    lgbm_fpr1 ,lgbm_tpr1, _ = roc_curve(ky1_test, lgbm_probs)

    lgbm.fit(kX2_traini,ky2_train)
    lgbm_probs = lgbm.predict_proba(kX2_testi)[:,1]
    lgbm_auc_2 = roc_auc_score(ky2_test, lgbm_probs)  
    lgbm_fpr2 ,lgbm_tpr2, _ = roc_curve(ky2_test, lgbm_probs)

    lgbm.fit(kX3_traini,ky3_train)
    lgbm_probs = lgbm.predict_proba(kX3_testi)[:,1]
    lgbm_auc_3 = roc_auc_score(ky3_test, lgbm_probs)
    lgbm_fpr3 ,lgbm_tpr3, _ = roc_curve(ky3_test, lgbm_probs)

    lgbm.fit(kX4_traini,ky4_train)
    lgbm_probs = lgbm.predict_proba(kX4_testi)[:,1]
    lgbm_auc_4 = roc_auc_score(ky4_test, lgbm_probs)
    lgbm_fpr4 ,lgbm_tpr4, _ = roc_curve(ky4_test, lgbm_probs)


    tpr = [lgbm_tpr0, lgbm_tpr1, lgbm_tpr2, lgbm_tpr3, lgbm_tpr4]
    fpr = [lgbm_fpr0, lgbm_fpr1, lgbm_fpr2, lgbm_fpr3, lgbm_fpr4]


    # Save True and False Positive Rates for AUC Plot
    df_fpr, df_tpr = pd.DataFrame(fpr), pd.DataFrame(tpr)
    df_fpr_T, df_tpr_T  =  df_fpr.T, df_tpr.T
    df_fpr_T.columns, df_tpr_T.columns  =['fpr_0','fpr_1','fpr_2','fpr_3','fpr_4'], ['tpr_0','tpr_1','tpr_2','tpr_3','tpr_4']
    df = pd.concat([df_fpr_T, df_tpr_T], axis=1)
    df.to_csv('lgbm_fi_ftpr_Problem_FPs.csv', index=False)
# ---------------------------------------------------------------------
## Replace previous line with this for random_mislabelling.
    #df.to_csv('lgbm_fi_ftpr_Problem_FPs_'+str(rand)+'.csv', index=False) 
# ---------------------------------------------------------------------
