#!/usr/bin/env python
# coding: utf-8
# %%
permute = True # Set permute to True to build the model from one instance of shuffled labels

n_iter = 90 # Set number of iterations for permutation (1 - 100)

# # -----------------------------------------
# Loading required libraries
from joblib import Parallel, delayed
from dependency.progress import tqdm_joblib
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
random_seed = 1


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

for rand in range(n_iter):  

    # -----------------------------------------
    Xy = df_d.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    if permute == True:   
        y = Xy['Fault'].values
        np.random.seed(rand) 
        np.random.shuffle(y)
        Xy['Fault'] = y

    #Separate data classes into 3
    # -----------------------------------------
    #Class 0
    Xy0 = Xy[Xy.Fault == 0]
    X0, y0 = Xy0.drop('Fault', axis=1), Xy0['Fault']

    #Class 2
    Xy2 = Xy[Xy.Fault == 2]
    X2, y2 = Xy2.drop('Fault', axis=1), Xy2['Fault'].replace({2: 1}) 
    # -----------------------------------------

    # Calculate the indices for splitting into 5 equal parts
    indices0, indices2  = (np.linspace(0, len(Xy0), num=6, dtype=int), 
                           np.linspace(0, len(Xy2), num=6, dtype=int))

    # -----------------------------------------
    ## Split Classes into 5 folds
    split_dfs0 = [Xy0.iloc[indices0[i]:indices0[i+1]] for i in range(len(indices0)-1)]
    cv_0_0, cv_0_1, cv_0_2, cv_0_3, cv_0_4 = split_dfs0

    split_dfs2 = [Xy2.iloc[indices2[i]:indices2[i+1]] for i in range(len(indices2)-1)]
    cv_2_0, cv_2_1, cv_2_2, cv_2_3, cv_2_4 = split_dfs2
    
    # Set up training and testing folds using classes 0 and 2; 
    tr_cv_0 = te_cv_0 =  pd.concat([cv_0_0, cv_2_0], axis = 0)
    tr_cv_1 = te_cv_1 =  pd.concat([cv_0_1, cv_2_1], axis = 0)
    tr_cv_2 = te_cv_2 =  pd.concat([cv_0_2, cv_2_2], axis = 0)
    tr_cv_3 = te_cv_3 =  pd.concat([cv_0_3, cv_2_3], axis = 0)
    tr_cv_4 = te_cv_4 =  pd.concat([cv_0_4, cv_2_4], axis = 0)

    k_fold_train_0 = pd.concat([tr_cv_0, tr_cv_1, tr_cv_2, tr_cv_3], axis = 0) #skip 4
    ky0_train = k_fold_train_0['Fault'].replace({2: 1})
    kX0_train = k_fold_train_0.drop('Fault', axis=1)

    k_fold_train_1 = pd.concat([tr_cv_0, tr_cv_1, tr_cv_2, tr_cv_4], axis = 0) #skip 3
    ky1_train = k_fold_train_1['Fault'].replace({2: 1})
    kX1_train = k_fold_train_1.drop('Fault', axis=1)

    k_fold_train_2 = pd.concat([tr_cv_0, tr_cv_1, tr_cv_3, tr_cv_4], axis = 0) #skip 2
    ky2_train = k_fold_train_2['Fault'].replace({2: 1})
    kX2_train = k_fold_train_2.drop('Fault', axis=1)

    k_fold_train_3 = pd.concat([tr_cv_0, tr_cv_2, tr_cv_3, tr_cv_4], axis = 0) #skip 1
    ky3_train = k_fold_train_3['Fault'].replace({2: 1})
    kX3_train = k_fold_train_3.drop('Fault', axis=1)

    k_fold_train_4 = pd.concat([tr_cv_1, tr_cv_2, tr_cv_3, tr_cv_4], axis = 0) #skip 0
    ky4_train = k_fold_train_4['Fault'].replace({2: 1})
    kX4_train = k_fold_train_4.drop('Fault', axis=1)

    k_fold_test_0 = te_cv_4 #include 4
    ky0_test = k_fold_test_0['Fault']
    kX0_test = k_fold_test_0.drop('Fault', axis=1)

    k_fold_test_1 = te_cv_3 #include 3
    ky1_test = k_fold_test_1['Fault']
    kX1_test = k_fold_test_1.drop('Fault', axis=1)

    k_fold_test_2 = te_cv_2 #include 2
    ky2_test = k_fold_test_2['Fault']
    kX2_test = k_fold_test_2.drop('Fault', axis=1)

    k_fold_test_3 = te_cv_1 #include 1
    ky3_test = k_fold_test_3['Fault']
    kX3_test = k_fold_test_3.drop('Fault', axis=1)

    k_fold_test_4 = te_cv_0 #include 0
    ky4_test = k_fold_test_4['Fault']
    kX4_test = k_fold_test_4.drop('Fault', axis=1)

    X3y3 = pd.concat([tr_cv_0, tr_cv_1, tr_cv_2, tr_cv_3, tr_cv_4], axis = 0) #skip 4
    X3, y3 = X3y3.drop('Fault', axis=1), X3y3['Fault']

    # Initialise LGBM Classifier        
    lgbm = LGBMClassifier(random_state=random_seed, n_jobs=-1, verbose=-1)
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

    # # Build feature array
    for i in range(Feature_Lenght_1):
        cols_1 = aa_1[0:i]
        result_1.append(cols_1)
    result_1_lgbm = result_1[1:]

    if len(result_1_lgbm) < 100:
        num_features = len(result_1_lgbm)
    else:
        num_features = 100

    # Feature selection based on feature array
    def Problem_FPs_lgbm(i):
        idx_3 = result_1_lgbm[i] 
        kX0_traini, kX0_testi = kX0_train.iloc[:, idx_3], kX0_test.iloc[:, idx_3] 
        kX1_traini, kX1_testi = kX1_train.iloc[:, idx_3], kX1_test.iloc[:, idx_3] 
        kX2_traini, kX2_testi = kX2_train.iloc[:, idx_3], kX2_test.iloc[:, idx_3]  
        kX3_traini, kX3_testi = kX3_train.iloc[:, idx_3], kX3_test.iloc[:, idx_3]  
        kX4_traini, kX4_testi = kX4_train.iloc[:, idx_3], kX4_test.iloc[:, idx_3]  

        outSeries = pd.Series()
        lgbm.fit(kX0_traini,ky0_train)
        lgbm_probs = lgbm.predict_proba(kX0_testi)[:,1]
        lgbm_auc_0 = roc_auc_score(ky0_test, lgbm_probs)

        lgbm.fit(kX1_traini,ky1_train)
        lgbm_probs = lgbm.predict_proba(kX1_testi)[:,1]
        lgbm_auc_1 = roc_auc_score(ky1_test, lgbm_probs)    

        lgbm.fit(kX2_traini,ky2_train)
        lgbm_probs = lgbm.predict_proba(kX2_testi)[:,1]
        lgbm_auc_2 = roc_auc_score(ky2_test, lgbm_probs)  

        lgbm.fit(kX3_traini,ky3_train)
        lgbm_probs = lgbm.predict_proba(kX3_testi)[:,1]
        lgbm_auc_3 = roc_auc_score(ky3_test, lgbm_probs)

        lgbm.fit(kX4_traini,ky4_train)
        lgbm_probs = lgbm.predict_proba(kX4_testi)[:,1]
        lgbm_auc_4 = roc_auc_score(ky4_test, lgbm_probs)

        # Compute mean AUC of 5 folds
        a = [lgbm_auc_0, lgbm_auc_1, lgbm_auc_2, lgbm_auc_3, lgbm_auc_4]
        outSeries['AUC_ROC'] = round(np.mean(a),3)

        return outSeries
    with tqdm_joblib(tqdm(desc="Percentage Completion", total=num_features)) as progress_bar:
        Case_1_lgbm = pd.DataFrame(Parallel(n_jobs=-1)(delayed(Problem_FPs_lgbm)(i) for i in range(num_features)))

    maxrowindex = Case_1_lgbm["AUC_ROC"].idxmax()
    Case_1_3_lgbm_fi_casc.loc[0,['FS_Type']] = 'lgbm_fi_casc' # CHECK FEATURE IMPORTANCE TYPE USED
    Case_1_3_lgbm_fi_casc.loc[0,['Case']] = '1'
    Case_1_3_lgbm_fi_casc.loc[0,['Class']] = '0 and 2'
    Case_1_3_lgbm_fi_casc.loc[0,['Model']] = 'lgbm'

    kX0_traini = kX0_train.iloc[:,result_1_lgbm[maxrowindex]] #fi_thresh_iterest
    kX1_traini = kX1_train.iloc[:,result_1_lgbm[maxrowindex]] #fi_thresh_iterest
    kX2_traini = kX2_train.iloc[:,result_1_lgbm[maxrowindex]] #fi_thresh_iterest
    kX3_traini = kX3_train.iloc[:,result_1_lgbm[maxrowindex]] #fi_thresh_iterest
    kX4_traini = kX4_train.iloc[:,result_1_lgbm[maxrowindex]] #fi_thresh_iterest

    kX0_testi = kX0_test.iloc[:,result_1_lgbm[maxrowindex]] #fi_thresh_iterest
    kX1_testi = kX1_test.iloc[:,result_1_lgbm[maxrowindex]] #fi_thresh_iterest
    kX2_testi = kX2_test.iloc[:,result_1_lgbm[maxrowindex]] #fi_thresh_iterest
    kX3_testi = kX3_test.iloc[:,result_1_lgbm[maxrowindex]] #fi_thresh_iterest
    kX4_testi = kX4_test.iloc[:,result_1_lgbm[maxrowindex]] #fi_thresh_iterest

    lgbm.fit(kX0_traini,ky0_train)

    # Prediction Probabilities
    lgbm_probs = lgbm.predict_proba(kX0_testi)

    #keep probabilities for the positive outcome only (1)
    lgbm_probs = lgbm_probs[:,1]

    # Calculate AUC_ROC
    lgbm_auc_0 = roc_auc_score(ky0_test, lgbm_probs)
    lgbm_fpr0 ,lgbm_tpr0, _ = roc_curve(ky0_test, lgbm_probs, pos_label=1)

    lgbm.fit(kX1_traini,ky1_train)
    lgbm_probs = lgbm.predict_proba(kX1_testi)
    lgbm_probs = lgbm_probs[:,1]
    lgbm_auc_1 = roc_auc_score(ky1_test, lgbm_probs)    
    lgbm_fpr1 ,lgbm_tpr1, _ = roc_curve(ky1_test, lgbm_probs, pos_label=1)

    lgbm.fit(kX2_traini,ky2_train)
    lgbm_probs = lgbm.predict_proba(kX2_testi)
    lgbm_probs = lgbm_probs[:,1]
    lgbm_auc_2 = roc_auc_score(ky2_test, lgbm_probs)  
    lgbm_fpr2 ,lgbm_tpr2, _ = roc_curve(ky2_test, lgbm_probs, pos_label=1)

    lgbm.fit(kX3_traini,ky3_train)
    lgbm_probs = lgbm.predict_proba(kX3_testi)
    lgbm_probs = lgbm_probs[:,1]
    lgbm_auc_3 = roc_auc_score(ky3_test, lgbm_probs)
    lgbm_fpr3 ,lgbm_tpr3, _ = roc_curve(ky3_test, lgbm_probs, pos_label=1)

    lgbm.fit(kX4_traini,ky4_train)
    lgbm_probs = lgbm.predict_proba(kX4_testi)
    lgbm_probs = lgbm_probs[:,1]
    lgbm_auc_4 = roc_auc_score(ky4_test, lgbm_probs)
    lgbm_fpr4 ,lgbm_tpr4, _ = roc_curve(ky4_test, lgbm_probs, pos_label=1)

    a = [lgbm_auc_0, lgbm_auc_1, lgbm_auc_2, lgbm_auc_3, lgbm_auc_4]
    Case_1_3_lgbm_fi_casc.loc[0,['AUC_ROC']] = [str("%.3f" % np.mean(a))]

    tpr = [lgbm_tpr0, lgbm_tpr1, lgbm_tpr2, lgbm_tpr3, lgbm_tpr4]
    fpr = [lgbm_fpr0, lgbm_fpr1, lgbm_fpr2, lgbm_fpr3, lgbm_fpr4]

    Case_1_3_lgbm_fi_casc.to_csv(f'rand_auc_case1/new_lgbm_fi_casc_Case_1_misclass_rand_{str(rand)}.csv', index=False)

    # Save True and False Positive Rates for AUC Plot
    df_fpr, df_tpr = pd.DataFrame(fpr), pd.DataFrame(tpr)
    df_fpr_T, df_tpr_T  =  df_fpr.T, df_tpr.T
    df_fpr_T.columns, df_tpr_T.columns  =['fpr_0','fpr_1','fpr_2','fpr_3','fpr_4'], ['tpr_0','tpr_1','tpr_2','tpr_3','tpr_4']
    df = pd.concat([df_fpr_T, df_tpr_T], axis=1)
    df.to_csv(f'rand_ftpr_case1/new_lgbm_fi_casc_ftpr_Case_1_nr_misclass_rand_{str(rand)}.csv', index=False)
# %%
