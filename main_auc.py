#!/usr/bin/env python
# coding: utf-8
# %%
permute = True # Set permute to True to build the model from one instance of shuffled labels

n_iter = 90 # Set number of iterations for permutation (1 - 100)

# -----------------------------------------
# Loading required libraries
from joblib import Parallel, delayed
from dependency.mislabel import randomly_misclassify_labels
from dependency.progress import tqdm_joblib
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
random_seed = 208


# -----------------------------------------
# Load all engine feature data
dfs = [] # List to store data frames

for i in range(1, 11):
    filename = f'engine_data/eng{i}.csv'
    df = pd.read_csv(filename)
    dfs.append(df)

# Concatenate dataframes
df_d = pd.concat(dfs, ignore_index=True).fillna(0.1)

# Drop features VIF >> 5
df_d = df_d[df_d.columns.drop(['L_-b/2a True Airspeed (knots)','L_-b/2a CHT 3 (deg C)',
                               'D_-b/2a Oil Pressure (PSI)','L_-b/2a CHT 6 (deg C)',
                              'C_-b/2a Barometer Setting (inHg)','TO_-b/2a Barometer Setting (inHg)',
                               'L_-b/2a Barometer Setting (inHg)','D_-b/2a Barometer Setting (inHg)'])]


# Creating table to populate results
Case_1_3_lgbm_fi_casc = pd.DataFrame({'Case': [],'Model': [], 'FS_Type': [],'AUC_ROC': []})

# -----------------------------------------
#Separate data classes into 3
Xy = df_d

# -----------------------------------------
#Class 0
Xy0 = Xy[Xy.Fault == 0]
y0 = Xy0['Fault']
X0 = Xy0.drop('Fault', axis=1)

# -----------------------------------------
#Class 1
Xy1 = Xy[Xy.Fault == 1]
y1 = Xy1['Fault']
X1 = Xy1.drop('Fault', axis=1)

# -----------------------------------------
#Class 2
Xy2 = Xy[Xy.Fault == 2]
Xy2['Fault'].replace({2: 1}, inplace = True)
y2 = Xy2['Fault']
X2 = Xy2.drop('Fault', axis=1)

# -----------------------------------------
# Split Class 0 into 5 folds:
# Fold 1
X_train, X_test_1, y_train, y_test_1 = train_test_split(X0, y0, test_size=0.2, random_state=random_seed)
cv_0_0 = pd.concat([X_test_1,y_test_1], axis=1)

# Fold 2
X_train, X_test_2, y_train, y_test_2 = train_test_split(X_train,y_train, test_size=0.25, random_state=random_seed)
cv_0_1 = pd.concat([X_test_2,y_test_2], axis=1) 

# Fold 3
X_train, X_test_3, y_train, y_test_3 = train_test_split(X_train,y_train, test_size=0.33, random_state=random_seed)
cv_0_2 = pd.concat([X_test_3,y_test_3], axis=1) 

# Fold 4 and 5
X_train, X_test_4, y_train, y_test_4 = train_test_split(X_train,y_train, test_size=0.50, random_state=random_seed)
cv_0_3 = pd.concat([X_test_4,y_test_4], axis=1) 
cv_0_4 = pd.concat([X_train,y_train], axis=1) 

# -----------------------------------------
# Split Class 1 into 5 folds:
# Fold 1
X_train, X_test_1, y_train, y_test_1 = train_test_split(X1, y1, test_size=0.2, random_state=random_seed)
cv_1_0 = pd.concat([X_test_1,y_test_1], axis=1)

# Fold 2
X_train, X_test_2, y_train, y_test_2 = train_test_split(X_train,y_train, test_size=0.25, random_state=random_seed)
cv_1_1 = pd.concat([X_test_2,y_test_2], axis=1) 

# Fold 3
X_train, X_test_3, y_train, y_test_3 = train_test_split(X_train,y_train, test_size=0.33, random_state=random_seed)
cv_1_2 = pd.concat([X_test_3,y_test_3], axis=1) 

# Fold 4 and 5
X_train, X_test_4, y_train, y_test_4 = train_test_split(X_train,y_train, test_size=0.50, random_state=random_seed)
cv_1_3 = pd.concat([X_test_4,y_test_4], axis=1) 
cv_1_4 = pd.concat([X_train,y_train], axis=1) 

# -----------------------------------------
# Split Class 2 into 5 folds:
# Fold 1
X_train, X_test_1, y_train, y_test_1 = train_test_split(X2, y2, test_size=0.2, random_state=random_seed)
cv_2_0 = pd.concat([X_test_1,y_test_1], axis=1)

# Fold 2
X_train, X_test_2, y_train, y_test_2 = train_test_split(X_train,y_train, test_size=0.25, random_state=random_seed)
cv_2_1 = pd.concat([X_test_2,y_test_2], axis=1) 

# Fold 3
X_train, X_test_3, y_train, y_test_3 = train_test_split(X_train,y_train, test_size=0.33, random_state=random_seed)
cv_2_2 = pd.concat([X_test_3,y_test_3], axis=1) 

# Fold 4 and 5
X_train, X_test_4, y_train, y_test_4 = train_test_split(X_train,y_train, test_size=0.50, random_state=random_seed)
cv_2_3 = pd.concat([X_test_4,y_test_4], axis=1) 
cv_2_4 = pd.concat([X_train,y_train], axis=1)


for rand in range(n_iter):
    ## Case 3: (0 vs 1s2)

    # Set up training folds using classes 0, 1 and 2
    cv_0 = pd.concat([cv_0_0, cv_1_0, cv_2_0], axis = 0)
    cv_1 = pd.concat([cv_0_1, cv_1_1, cv_2_1], axis = 0)
    cv_2 = pd.concat([cv_0_2, cv_1_2, cv_2_2], axis = 0)
    cv_3 = pd.concat([cv_0_3, cv_1_3, cv_2_3], axis = 0)
    cv_4 = pd.concat([cv_0_4, cv_1_4, cv_2_4], axis = 0)

    all_df = pd.concat([cv_0, cv_1, cv_2, cv_3, cv_4], axis = 0)
    y_df = all_df['Fault']
    x_df = all_df.drop('Fault', axis=1)
    
    if permute == True:   
        original_labels = y_df.to_numpy()
        misclassified_labels = randomly_misclassify_labels(original_labels)
        y_df = pd.Series(misclassified_labels)    
        x_df.reset_index(drop=True, inplace=True)
        y_df.reset_index(drop=True, inplace=True)
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

    # Set up validation folds using classes 0 and 1 (remove class 2)
    cv_0, cv_1, cv_2, cv_3, cv_4 = (cv_0[cv_0.Fault != 2], cv_1[cv_1.Fault != 2], cv_2[cv_2.Fault != 2], 
                                    cv_3[cv_3.Fault != 2], cv_4[cv_4.Fault != 2])
    
    all_df_val = pd.concat([cv_0, cv_1, cv_2, cv_3, cv_4], axis = 0)
    y_df = all_df_val['Fault']
    x_df = all_df_val.drop('Fault', axis=1)

    split_df = np.array_split(all_df_val, 5)

    # Save each part as a separate DataFrame
    cv_0, cv_1, cv_2, cv_3, cv_4 = split_df


    k_fold_val_0 = cv_4 #include 4
    ky0_val = k_fold_val_0['Fault']
    kX0_val =k_fold_val_0.drop('Fault', axis=1)

    k_fold_val_1 = cv_3 #include 3
    ky1_val = k_fold_val_1['Fault']
    kX1_val =k_fold_val_1.drop('Fault', axis=1)

    k_fold_val_2 = cv_2 #include 2
    ky2_val = k_fold_val_2['Fault']
    kX2_val =k_fold_val_2.drop('Fault', axis=1)

    k_fold_val_3 = cv_1 #include 1
    ky3_val = k_fold_val_3['Fault']
    kX3_val =k_fold_val_3.drop('Fault', axis=1)

    k_fold_val_4 = cv_0 #include 0
    ky4_val = k_fold_val_4['Fault']
    kX4_val =k_fold_val_4.drop('Fault', axis=1)


    # Feature Importance for Case 3
    Xy3 = all_df
    Xy3['Fault'].replace({2: 1}, inplace = True) #Replace class 2 labels to 1
    y3 = Xy3['Fault']
    X3 = Xy3.drop('Fault', axis=1)
        # Initialise LGBM Classifier        
    lgbm = LGBMClassifier(random_state=random_seed, n_jobs=-1)
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


    ## Case 3: (0 vs 1s2) Scenario (a): Iteration

    def Case_3_lgbm_nr(i):

        idx_3 = result_3_lgbm[i] #fi_thresh_iterest
        kX0_traini = kX0_train.iloc[:, idx_3] #fi_thresh_iterest
        kX1_traini = kX1_train.iloc[:, idx_3] #fi_thresh_iterest
        kX2_traini = kX2_train.iloc[:, idx_3] #fi_thresh_iterest
        kX3_traini = kX3_train.iloc[:, idx_3] #fi_thresh_iterest
        kX4_traini = kX4_train.iloc[:, idx_3] #fi_thresh_iterest

        kX0_vali = kX0_val.iloc[:, idx_3] #fi_thresh_iterest
        kX1_vali = kX1_val.iloc[:, idx_3] #fi_thresh_iterest
        kX2_vali = kX2_val.iloc[:, idx_3] #fi_thresh_iterest
        kX3_vali = kX3_val.iloc[:, idx_3] #fi_thresh_iterest
        kX4_vali = kX4_val.iloc[:, idx_3] #fi_thresh_iterest


        outSeries = pd.Series()
        outSeries['Case'] = '3'
        outSeries['Class'] = '0 and 1s2'
        outSeries['Model'] = 'lgbm'
        outSeries['FS_Type'] = 'lgbm_fi_casc'

        lgbm.fit(kX0_traini,ky0_train)

        # Prediction Probabilities
        lgbm_probs = lgbm.predict_proba(kX0_vali)

        #keep probabilities for the positive outcome only (1)
        lgbm_probs = lgbm_probs[:,1]

        # Calculate AUC_ROC
        lgbm_auc_0 = roc_auc_score(ky0_val, lgbm_probs)
        lgbm_fpr0 ,lgbm_tpr0, _ = roc_curve(ky0_val, lgbm_probs)


        lgbm.fit(kX1_traini,ky1_train)
        lgbm_probs = lgbm.predict_proba(kX1_vali)
        lgbm_probs = lgbm_probs[:,1]
        lgbm_auc_1 = roc_auc_score(ky1_val, lgbm_probs)    
        lgbm_fpr1 ,lgbm_tpr1, _ = roc_curve(ky1_val, lgbm_probs)


        lgbm.fit(kX2_traini,ky2_train)
        lgbm_probs = lgbm.predict_proba(kX2_vali)
        lgbm_probs = lgbm_probs[:,1]
        lgbm_auc_2 = roc_auc_score(ky2_val, lgbm_probs)  
        lgbm_fpr2 ,lgbm_tpr2, _ = roc_curve(ky2_val, lgbm_probs)


        lgbm.fit(kX3_traini,ky3_train)
        lgbm_probs = lgbm.predict_proba(kX3_vali)
        lgbm_probs = lgbm_probs[:,1]
        lgbm_auc_3 = roc_auc_score(ky3_val, lgbm_probs)
        lgbm_fpr3 ,lgbm_tpr3, _ = roc_curve(ky3_val, lgbm_probs)

        lgbm.fit(kX4_traini,ky4_train)
        lgbm_probs = lgbm.predict_proba(kX4_vali)
        lgbm_probs = lgbm_probs[:,1]
        lgbm_auc_4 = roc_auc_score(ky4_val, lgbm_probs)
        lgbm_fpr4 ,lgbm_tpr4, _ = roc_curve(ky4_val, lgbm_probs)


        a = [lgbm_auc_0, lgbm_auc_1, lgbm_auc_2, lgbm_auc_3, lgbm_auc_4]
        outSeries['AUC_ROC'] = round(np.mean(a),3)

        return outSeries
                                     
    with tqdm_joblib(tqdm(desc="Percentage Completion", total=num_features)) as progress_bar:
        Case_3_lgbm_fi_casc_nr = pd.DataFrame(Parallel(n_jobs=-1)(delayed(Case_3_lgbm_nr)(i) for i in range(num_features)))

    # ## Case 3: (0 vs 1s2) Scenario (a): MaxIndex

    maxrowindex_3nr_lgbm_fi_casc = Case_3_lgbm_fi_casc_nr["AUC_ROC"].idxmax()
    Case_1_3_lgbm_fi_casc.loc[2,['FS_Type']] = 'lgbm_fi_casc' # CHECK FEATURE IMPORTANCE TYPE USED
    Case_1_3_lgbm_fi_casc.loc[2,['Case']] = '3'
    Case_1_3_lgbm_fi_casc.loc[2,['Class']] = '0 and 1s2'
    Case_1_3_lgbm_fi_casc.loc[2,['Model']] = 'lgbm'

    kX0_traini = kX0_train.iloc[:,result_3_lgbm[maxrowindex_3nr_lgbm_fi_casc]] #fi_thresh_iterest
    kX1_traini = kX1_train.iloc[:,result_3_lgbm[maxrowindex_3nr_lgbm_fi_casc]] #fi_thresh_iterest
    kX2_traini = kX2_train.iloc[:,result_3_lgbm[maxrowindex_3nr_lgbm_fi_casc]] #fi_thresh_iterest
    kX3_traini = kX3_train.iloc[:,result_3_lgbm[maxrowindex_3nr_lgbm_fi_casc]] #fi_thresh_iterest
    kX4_traini = kX4_train.iloc[:,result_3_lgbm[maxrowindex_3nr_lgbm_fi_casc]] #fi_thresh_iterest
    Case_1_3_lgbm_fi_casc.loc[2,['Number_of_Features']] = str(kX4_traini.shape[1])

    kX0_vali = kX0_val.iloc[:,result_3_lgbm[maxrowindex_3nr_lgbm_fi_casc]] #fi_thresh_iterest
    kX1_vali = kX1_val.iloc[:,result_3_lgbm[maxrowindex_3nr_lgbm_fi_casc]] #fi_thresh_iterest
    kX2_vali = kX2_val.iloc[:,result_3_lgbm[maxrowindex_3nr_lgbm_fi_casc]] #fi_thresh_iterest
    kX3_vali = kX3_val.iloc[:,result_3_lgbm[maxrowindex_3nr_lgbm_fi_casc]] #fi_thresh_iterest
    kX4_vali = kX4_val.iloc[:,result_3_lgbm[maxrowindex_3nr_lgbm_fi_casc]] #fi_thresh_iterest



    lgbm.fit(kX0_traini,ky0_train)

    # Prediction Probabilities
    lgbm_probs = lgbm.predict_proba(kX0_vali)

    #keep probabilities for the positive outcome only (1)
    lgbm_probs = lgbm_probs[:,1]

    # Calculate AUC_ROC
    lgbm_auc_0 = roc_auc_score(ky0_val, lgbm_probs)


    lgbm_fpr0 ,lgbm_tpr0, _ = roc_curve(ky0_val, lgbm_probs)
    # calculate the g-mean for each threshold
    gmeans_lgbm0 = np.sqrt(lgbm_tpr0 * (1-lgbm_fpr0))
    ix_lgbm0 = np.argmax(gmeans_lgbm0)
    lgbm_gmean_0 = gmeans_lgbm0[ix_lgbm0]

    lgbm.fit(kX1_traini,ky1_train)
    lgbm_probs = lgbm.predict_proba(kX1_vali)
    lgbm_probs = lgbm_probs[:,1]
    lgbm_auc_1 = roc_auc_score(ky1_val, lgbm_probs)    
    lgbm_fpr1 ,lgbm_tpr1, _ = roc_curve(ky1_val, lgbm_probs)
    # calculate the g-mean for each threshold
    gmeans_lgbm1 = np.sqrt(lgbm_tpr1 * (1-lgbm_fpr1))
    ix_lgbm1 = np.argmax(gmeans_lgbm1)
    lgbm_gmean_1 = gmeans_lgbm1[ix_lgbm1]

    lgbm.fit(kX2_traini,ky2_train)
    lgbm_probs = lgbm.predict_proba(kX2_vali)
    lgbm_probs = lgbm_probs[:,1]
    lgbm_auc_2 = roc_auc_score(ky2_val, lgbm_probs)  
    lgbm_fpr2 ,lgbm_tpr2, _ = roc_curve(ky2_val, lgbm_probs)
    # calculate the g-mean for each threshold
    gmeans_lgbm2 = np.sqrt(lgbm_tpr2 * (1-lgbm_fpr2))
    ix_lgbm2 = np.argmax(gmeans_lgbm2)
    lgbm_gmean_2 = gmeans_lgbm2[ix_lgbm2]

    lgbm.fit(kX3_traini,ky3_train)
    lgbm_probs = lgbm.predict_proba(kX3_vali)
    lgbm_probs = lgbm_probs[:,1]
    lgbm_auc_3 = roc_auc_score(ky3_val, lgbm_probs)
    lgbm_fpr3 ,lgbm_tpr3, _ = roc_curve(ky3_val, lgbm_probs)
    # calculate the g-mean for each threshold
    gmeans_lgbm3 = np.sqrt(lgbm_tpr3 * (1-lgbm_fpr3))
    ix_lgbm3 = np.argmax(gmeans_lgbm3)
    lgbm_gmean_3 = gmeans_lgbm3[ix_lgbm3]

    lgbm.fit(kX4_traini,ky4_train)
    lgbm_probs = lgbm.predict_proba(kX4_vali)
    lgbm_probs = lgbm_probs[:,1]
    lgbm_auc_4 = roc_auc_score(ky4_val, lgbm_probs)
    lgbm_fpr4 ,lgbm_tpr4, _ = roc_curve(ky4_val, lgbm_probs)
    # calculate the g-mean for each threshold
    gmeans_lgbm4 = np.sqrt(lgbm_tpr4 * (1-lgbm_fpr4))
    ix_lgbm4 = np.argmax(gmeans_lgbm4)
    lgbm_gmean_4 = gmeans_lgbm4[ix_lgbm4]

    a = [lgbm_auc_0, lgbm_auc_1, lgbm_auc_2, lgbm_auc_3, lgbm_auc_4]
    Case_1_3_lgbm_fi_casc.loc[2,['AUC_ROC']] = [str("%.3f" % np.mean(a))]


    tpr = [lgbm_tpr0, lgbm_tpr1, lgbm_tpr2, lgbm_tpr3, lgbm_tpr4]
    fpr = [lgbm_fpr0, lgbm_fpr1, lgbm_fpr2, lgbm_fpr3, lgbm_fpr4]


    Case_1_3_lgbm_fi_casc.to_csv(f'rand_auc/new_lgbm_fi_casc_Case_3_misclass_rand_{str(rand)}.csv', index=False)

    # Save True and False Positive Rates for AUC Plot
    df_fpr, df_tpr = pd.DataFrame(fpr), pd.DataFrame(tpr)
    df_fpr_T, df_tpr_T  =  df_fpr.T, df_tpr.T
    df_fpr_T.columns, df_tpr_T.columns  =['fpr_0','fpr_1','fpr_2','fpr_3','fpr_4'], ['tpr_0','tpr_1','tpr_2','tpr_3','tpr_4']
    df = pd.concat([df_fpr_T, df_tpr_T], axis=1)
    df.to_csv(f'rand_ftpr/new_lgbm_fi_casc_ftpr_Case_3_nr_misclass_rand_{str(rand)}.csv', index=False)
# %%
# 0.956, 0.752
