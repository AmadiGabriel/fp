#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# Loading required libraries
from dependency_1 import tqdm_joblib
from dependency_2 import randomly_misclassify_labels
from tqdm import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
import random
random_seed = 1

# -----------------------------------------
# Load all engine feature data
dfs = [] # List to store data frames

for i in range(1, 11):
    filename = f'eng{i}.csv'
    df = pd.read_csv(filename)
    dfs.append(df)

# Concatenate dataframes
df_d = pd.concat(dfs, ignore_index=True).fillna(0.1)

# Drop features VIF >> 5
df_d = df_d[df_d.columns.drop(['L_-b/2a True Airspeed (knots)','L_-b/2a CHT 3 (deg C)',
                               'D_-b/2a Oil Pressure (PSI)','L_-b/2a CHT 6 (deg C)',
                              'C_-b/2a Barometer Setting (inHg)','TO_-b/2a Barometer Setting (inHg)',
                               'L_-b/2a Barometer Setting (inHg)','D_-b/2a Barometer Setting (inHg)'])]

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

# -----------------------------------------
# Set up training folds using classes 0, 1 and 2
cv_0 = pd.concat([cv_0_0, cv_1_0, cv_2_0], axis = 0)
cv_1 = pd.concat([cv_0_1, cv_1_1, cv_2_1], axis = 0)
cv_2 = pd.concat([cv_0_2, cv_1_2, cv_2_2], axis = 0)
cv_3 = pd.concat([cv_0_3, cv_1_3, cv_2_3], axis = 0)
cv_4 = pd.concat([cv_0_4, cv_1_4, cv_2_4], axis = 0)

all_df = pd.concat([cv_0, cv_1, cv_2, cv_3, cv_4], axis = 0)
y_df = all_df['Fault']


original_labels = y_df.to_numpy()
misclassified_labels = randomly_misclassify_labels(original_labels)
y_df = pd.Series(misclassified_labels) 

x_df = all_df.drop('Fault', axis=1)
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

# Set up validation folds using classes 0 and 1
cv_0 = pd.concat([cv_0_0, cv_1_0], axis = 0)
cv_1 = pd.concat([cv_0_1, cv_1_1], axis = 0)
cv_2 = pd.concat([cv_0_2, cv_1_2], axis = 0)
cv_3 = pd.concat([cv_0_3, cv_1_3], axis = 0)
cv_4 = pd.concat([cv_0_4, cv_1_4], axis = 0)

all_df = pd.concat([cv_0, cv_1, cv_2, cv_3, cv_4], axis = 0)
y_df = all_df['Fault']

original_labels = y_df.to_numpy()
misclassified_labels = randomly_misclassify_labels(original_labels)
y_df = pd.Series(misclassified_labels) 

x_df = all_df.drop('Fault', axis=1)

x_df.reset_index(drop=True, inplace=True)
y_df.reset_index(drop=True, inplace=True)

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

# -----------------------------------------
# Feature Importance Computation
Xy3 = df_d
Xy3['Fault'].replace({2: 1}, inplace = True) #Replace class 2 labels to 1
y3 = Xy3['Fault']
X3 = Xy3.drop('Fault', axis=1)

original_labels = y3.to_numpy()
misclassified_labels = randomly_misclassify_labels(original_labels)
y3 = pd.Series(misclassified_labels)

# -----------------------------------------
# Initialise LightGradientBoost Classifier

for ijk in range(1,101):
    random_seed = random.randrange(1, 10**10)

    lgbm = LGBMClassifier(n_jobs=-1, random_state=random_seed, colsample_bytree=0.99, verbosity=-1)

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

    # -----------------------------------------
    # Build feature array
    for i in range(Feature_Lenght_1):
        cols_1 = aa_1[0:i]
        result_1.append(cols_1)
    result_3_lgbm = result_1[1:]

    # Select 
    if len(result_3_lgbm) < 100:
        num_features = len(result_3_lgbm)

    else:
        num_features = 100

    # -----------------------------------------
    # Feature selection based on feature array
    def Problem_FPs_lgbm(i):
        idx_fps = result_3_lgbm[i] 
        kX0_traini, kX0_testi = kX0_train.iloc[:, idx_fps], kX0_test.iloc[:, idx_fps] 
        kX1_traini, kX1_testi = kX1_train.iloc[:, idx_fps], kX1_test.iloc[:, idx_fps] 
        kX2_traini, kX2_testi = kX2_train.iloc[:, idx_fps], kX2_test.iloc[:, idx_fps]  
        kX3_traini, kX3_testi = kX3_train.iloc[:, idx_fps], kX3_test.iloc[:, idx_fps]  
        kX4_traini, kX4_testi = kX4_train.iloc[:, idx_fps], kX4_test.iloc[:, idx_fps]  

        outSeries = pd.Series()

        random_seed = random.randrange(1, 10**10)
        lgbm = LGBMClassifier(random_state=random_seed, n_jobs=-1, colsample_bytree=0.99, verbosity=-1)

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
        lgbm_fi_iter_Problem_FPs = pd.DataFrame(Parallel(n_jobs=-1, prefer="threads")(delayed(Problem_FPs_lgbm)(i) for i in range(num_features)))
    max_auc = lgbm_fi_iter_Problem_FPs["AUC_ROC"].max()
    with open(f'perm_files/permuted_rand_{ijk}.txt', 'w') as f:
        f.write(str(max_auc))


# %%




