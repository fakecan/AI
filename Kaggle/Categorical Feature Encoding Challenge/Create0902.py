import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

# ■■■■■■■■■■■■■■■■    Data set load   ■■■■■■■■■■■■■■■■
df_train = pd.read_csv('./data/cat-in-the-dat/train.csv')
df_test = pd.read_csv('./data/cat-in-the-dat/test.csv')
submission = pd.read_csv('./data/cat-in-the-dat/sample_submission.csv', index_col='id')

# ■■■■■■■■■■■■■■■■    Summary ■■■■■■■■■■■■■■■■
from scipy import stats
def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values    # 서로 다른 특징값의 개수 확인
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

# summary = resumetable(df_train)
# print(summary)

# print(df_train.info())
# print(df_train.shape, df_test.shape)    # (300000, 25) (200000, 24)

# ■■■■■■■■■■■■■■■■    Data Preprocessing  ■■■■■■■■■■■■■■■■
bin_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']
bin_dict = {'T':1, 'F':0,   # bin_3
            'Y':1, 'N':0}   # bin_4

df_train['bin_3'] = df_train['bin_3'].map(bin_dict)
df_train['bin_4'] = df_train['bin_4'].map(bin_dict)
df_test['bin_3'] = df_test['bin_3'].map(bin_dict)
df_test['bin_4'] = df_test['bin_4'].map(bin_dict)
# print(df_train[['bin_3', 'bin_4']].head())    # confirm

nom_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

df_test['target'] = 'test'
df = pd.concat([df_train, df_test], axis=0, sort=False)
# print('Shape before transformation:', df.shape) # (500000, 25)
df = pd.get_dummies(df, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],
                        prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],
                        drop_first=True)
# print('Shape after transformation:', df.shape)  # (500000, 40)
# print(df_train.head())

df_train = df[df['target'] != 'test']
df_test = df[df['target'] == 'test'].drop('target', axis=1)
del df
# print(df_train.head())

# ■■■■■■■■■■■■■■■■    ord_1 ~ ord_5   ■■■■■■■■■■■■■■■■
ord_cols = ['ord_0', 'ord_1', 'ord_2', 'ord_3']

df_train['ord_5_ot'] = 'Others'
df_train.loc[df_train['ord_5'].isin(df_train['ord_5'].value_counts()[:25].sort_index().index), 'ord_5_ot'] = df_train['ord_5']

ord_5_count = df_train['ord_5'].value_counts().reset_index()['ord_5'].values

from pandas.api.types import CategoricalDtype

ord_1 = CategoricalDtype(categories=['Novice', 'Contributor', 'Expert',
                                     'Master', 'Grandmaster'], ordered=True)
ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot',
                                     'Boiling Hot', 'Lava Hot'], ordered=True)
ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',
                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)
ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)

df_train.ord_1 = df_train.ord_1.astype(ord_1)
df_train.ord_2 = df_train.ord_2.astype(ord_2)
df_train.ord_3 = df_train.ord_3.astype(ord_3)
df_train.ord_4 = df_train.ord_4.astype(ord_4)

df_test.ord_1 = df_test.ord_1.astype(ord_1)
df_test.ord_2 = df_test.ord_2.astype(ord_2)
df_test.ord_3 = df_test.ord_3.astype(ord_3)
df_test.ord_4 = df_test.ord_4.astype(ord_4)
# print(df_train.ord_3.head())

df_train.ord_1 = df_train.ord_1.cat.codes
df_train.ord_2 = df_train.ord_2.cat.codes
df_train.ord_3 = df_train.ord_3.cat.codes
df_train.ord_4 = df_train.ord_4.cat.codes

df_test.ord_1 = df_test.ord_1.cat.codes
df_test.ord_2 = df_test.ord_2.cat.codes
df_test.ord_3 = df_test.ord_3.cat.codes
df_test.ord_4 = df_test.ord_4.cat.codes
# print(df_train[['ord_0', 'ord_1', 'ord_2', 'ord_3']].head())

# ■■■■■■■■■■■■■■■■    day, month  ■■■■■■■■■■■■■■■■
date_cols = ['day', 'month']

def date_cyc_enc(df, col, max_vals):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_vals)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_vals)
    return df

df_train = date_cyc_enc(df_train, 'day', 7)
df_test = date_cyc_enc(df_test, 'day', 7)

df_train = date_cyc_enc(df_train, 'month', 12)
dt_test = date_cyc_enc(df_test, 'month', 12)

# string.ascii_letters: 대소문자를 모두 포함하고 있는 문자열
import string
df_train['ord_5_oe_add'] = df_train['ord_5'].apply(
    lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))
df_test['ord_5_oe_add'] = df_test['ord_5'].apply(
    lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))

df_train['ord_5_oe_join'] = df_train['ord_5'].apply(
    lambda x:float(''.join(str(string.ascii_letters.find(letter)+1) for letter in x)))
df_test['ord_5_oe_join'] = df_test['ord_5'].apply(
    lambda x:float(''.join(str(string.ascii_letters.find(letter)+1) for letter in x)))

df_train['ord_5_oe1'] = df_train['ord_5'].apply(
    lambda x:(string.ascii_letters.find(x[0])+1))
df_test['ord_5_oe1'] = df_test['ord_5'].apply(
    lambda x:(string.ascii_letters.find(x[0])+1))

df_train['ord_5_oe2'] = df_train['ord_5'].apply(
    lambda x:(string.ascii_letters.find(x[0])+1))
df_test['ord_5_oe2'] = df_test['ord_5'].apply(
    lambda x:(string.ascii_letters.find(x[0])+1))


for col in ['ord_5_oe1', 'ord_5_oe2', 'ord_5_oe_add', 'ord_5_oe_join']:
    df_train[col] = df_train[col].astype('float64')
    df_test[col] = df_test[col].astype('float64')

# print(df_train[['ord_5', 'ord_5_oe_add', 'ord_5_oe_join', 'ord_5_oe1', 'ord_5_oe2']].head())
#   ord_5  ord_5_oe_add  ord_5_oe_join  ord_5_oe1  ord_5_oe2
# 0    kr          29.0         1118.0       11.0       11.0
# 1    bF          34.0          232.0        2.0        2.0
# 2    Jc          39.0          363.0       36.0       36.0
# 3    kW          60.0         1149.0       11.0       11.0
# 4    qP          59.0         1742.0       17.0       17.0

high_card_feats = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
# print(resumetable(df_train[high_card_feats]))
#     Name  dtypes  Missing  Uniques First Value Second Value Third Value  Entropy
# 0  nom_5  object        0      222   50f116bcf    b3b4d25d0   3263bdce5     7.49
# 1  nom_6  object        0      522   3ac1b8814    fbcb50fc1   0922e3cb8     8.74
# 2  nom_7  object        0     1220   68f6ad3e9    3b6dd5612   a6a36f527     9.97
# 3  nom_8  object        0     2215   c389000ab    4cd920251   de9c9f684    10.84
# 4  nom_9  object        0    11981   2f4cb3d51    f83c56c21   ae6800dd0    13.27

for col in high_card_feats:
    df_train[f'hash_{col}'] = df_train[col].apply(lambda x:hash(str(x)) % 5000)
    df_test[f'hash_{col}'] = df_test[col].apply(lambda x:hash(str(x)) % 5000)

for col in high_card_feats:
    enc_nom_1 = (df_train.groupby(col).size()) / len(df_train)
    df_train[f'freq_{col}'] = df_train[col].apply(lambda x:enc_nom_1[x])
    # df_test[f'freq_{col}'] = df_test[col].apply(lambda x:enc_nom_1[x])


# ■■■■■■■■■■■■■■■■    LabelEncoder    ■■■■■■■■■■■■■■■■
from sklearn.preprocessing import LabelEncoder
for f in ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']:
    if df_train[f].dtype == 'object' or df_test[f].dtype == 'object':
        label = LabelEncoder()
        label.fit(list(df_train[f].values) + list(df_test[f].values))
        df_train[f'le_{f}'] = label.transform(list(df_train[f].values))
        df_test[f'le_{f}'] = label.transform(list(df_test[f].values))
# ■■■■■■■■■■■■■■■■                    ■■■■■■■■■■■■■■■■

new_feat = ['hash_nom_5', 'hash_nom_6', 'hash_nom_7', 'hash_nom_8',
            'hash_nom_9',  'freq_nom_5', 'freq_nom_6', 'freq_nom_7', 
            'freq_nom_8', 'freq_nom_9', 'le_nom_5', 'le_nom_6',
            'le_nom_7', 'le_nom_8', 'le_nom_9']

# print(resumetable(df_train[high_card_feats + new_feat]))

# print(df_train[['nom_5', 'hash_nom_5', 'freq_nom_5', 'le_nom_5']].head())
#        nom_5  hash_nom_5  freq_nom_5  le_nom_5
# 0  50f116bcf        2810    0.008647        78
# 1  b3b4d25d0        2659    0.002640       159
# 2  3263bdce5        2127    0.008413        44
# 3  f12246592        1587    0.003250       209
# 4  5b0f5acd5        4693    0.006700        90

df_train.drop(['ord_5_ot', 'ord_5', 
                'hash_nom_6', 'hash_nom_7', 'hash_nom_8', 'hash_nom_9',
               #'le_nom_5', 'le_nom_6', 'le_nom_7', 'le_nom_8', 'le_nom_9',
                'freq_nom_6', 'freq_nom_7', 'freq_nom_8', 'freq_nom_9',
                'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'
              ], axis=1, inplace=True)

df_test.drop(['ord_5',
              'hash_nom_6', 'hash_nom_7', 'hash_nom_8', 'hash_nom_9', 
              #'le_nom_5', 'le_nom_6', 'le_nom_7', 'le_nom_8', 'le_nom_9',
              #'freq_nom_5', 'freq_nom_6', 'freq_nom_7', 'freq_nom_8', 'freq_nom_9',
              'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',
              ], axis=1, inplace=True)

x_train = df_train.drop(['id', 'target'], axis=1)
y_train = df_train['target']
y_train = y_train.astype(bool)
x_test = df_test.drop(['id'], axis=1)
print(x_train.shape, y_train.shape) # (300000, 47) (300000,)
print(x_test.shape) # (200000, 46)

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
from sklearn.metrics import make_scorer
import time
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
import gc
def objective(params):
    time1 = time.time()
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'subsample': "{:.2f}".format(params['subsample']),
        'reg_alpha': "{:.3f}".format(params['reg_alpha']),
        'reg_lambda': "{:.3f}".format(params['reg_lambda']),
        'learning_rate': "{:.3f}".format(params['learning_rate']),
        'num_leaves': '{:.3f}'.format(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'min_child_samples': '{:.3f}'.format(params['min_child_samples']),
        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),
        'bagging_fraction': '{:.3f}'.format(params['bagging_fraction'])
    }

    print("\n◎◎◎◎◎◎◎◎◎◎ New Run ◎◎◎◎◎◎◎◎◎◎")
    print(f"params = {params}")
    FOLDS = 3
    count = 1
    kf = KFold(n_splits=FOLDS, shuffle=False, random_state=66)

    # tss = TimeSeriesSplit(n_splits=FOLDS)
    y_predict = np.zeros(submission.shape[0])
    # y_oof = np.zeros(X_train.shape[0])
    score_mean = 0
    for tr_idx, val_idx in kf.split(x_train, y_train):
        clf = xgb.XGBClassifier(
            n_estimators=10, random_state=4, 
            verbose=True, 
            tree_method='gpu_hist', 
            **params
        )

        X_tr, X_vl = x_train.iloc[tr_idx, :], x_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        clf.fit(X_tr, y_tr)
        #y_pred_train = clf.predict_proba(X_vl)[:,1]
        #print(y_pred_train)
        score = make_scorer(roc_auc_score, needs_proba=True)(clf, X_vl, y_vl)
        # plt.show()
        score_mean += score
        print(f'{count} CV - score: {round(score, 4)}')
        count += 1
    time2 = time.time() - time1
    print(f"Total Time Run: {round(time2 / 60,2)}")
    gc.collect()
    print(f'Mean ROC_AUC: {score_mean / FOLDS}')
    del X_tr, X_vl, y_tr, y_vl, clf, score
    
    return -(score_mean / FOLDS)

space = {   'max_depth': hp.quniform('max_depth', 2, 8, 1),
            'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),
            'reg_lambda': hp.uniform('reg_lambda', 0.01, .4),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.15),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
            'gamma': hp.uniform('gamma', 0.01, .7),
            'num_leaves': hp.choice('num_leaves', list(range(20, 200, 5))),
            'min_child_samples': hp.choice('min_child_samples', list(range(100, 250, 10))),
            'subsample': hp.choice('subsample', [.5, 0.6, 0.7, .8]),
            'feature_fraction': hp.uniform('feature_fraction', 0.4, .8),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.4, .9)
        }

best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=45, 
            # trials=trials
           )

best_params = space_eval(space, best)
best_params['max_depth'] = int(best_params['max_depth'])
print(best_params)

clf = xgb.XGBClassifier(
    n_estimators=10,
    **best_params,
    tree_method='gpu_hist'
)

clf.fit(x_train, y_train)

y_predict = clf.predict_proba(x_test)[:,1]

feature_important = clf.get_booster().get_score(importance_type="gain")
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)


feature_important = clf.get_booster().get_score(importance_type="weight")
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

# Top 10 features
print(data.head(20))


submission['target'] = y_predict
submission.to_csv('submission.csv')
# kfold 3 n_estimators 100 100의 값:    0.7580380687384087
# kfold 5 n_estimators 300 300의 값: