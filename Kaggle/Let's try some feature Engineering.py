import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('./data/cat-in-the-dat/train.csv')
test = pd.read_csv('./data/cat-in-the-dat/test.csv')

print(train.shape)
print(test.shape)

# Subset
target = train['target']
train_id = train['id']
test_id = test['id']
train.drop(['target', 'id'], axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

print(train.shape)
print(test.shape)

# One Hot Encode
traintest = pd.concat([train, test])
traintest['ord_5.1'] = traintest['ord_5'].apply(lambda x: x[0])
traintest['ord_5.2'] = traintest['ord_5'].apply(lambda x: x[1])
traintest['ord_1.1'] = traintest['ord_1'].map(traintest.ord_1.value_counts().to_dict())

dummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)
train_ohe = dummies.iloc[:train.shape[0], :]
test_ohe = dummies.iloc[train.shape[0]:, :]

print(train_ohe.shape)
print(test_ohe.shape)

train_ohe = train_ohe.sparse.to_coo().tocsr()
test_ohe = test_ohe.sparse.to_coo().tocsr()

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression

# Model
def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):
    kf = KFold(n_splits=5)
    fold_splits = kf.split(train, target)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0]))
    i = 1
    for dev_index, val_index in fold_splits:
        print('Started ' + label + ' fold ' + str(i) + '/5')
        dev_X, val_X = train[dev_index], train[val_index]
        dev_y, val_y = target[dev_index], target[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            print(label + ' cv score {}: {}'.format(i, cv_score))
        i += 1
    print('{} cv scores : {}'.format(label, cv_scores))
    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std score : {}'.format(label, np.std(cv_scores)))
    pred_full_test = pred_full_test / 5.0
    results = {'label': label,
              'train': pred_train, 'test': pred_full_test,
              'cv': cv_scores}
    return results


def runLR(train_X, train_y, test_X, test_y, test_X2, params):
    print('Train LR')
    model = LogisticRegression(**params)
    model.fit(train_X, train_y)
    print('Predict 1/2')
    pred_test_y = model.predict_proba(test_X)[:, 1]
    print('Predict 2/2')
    pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    return pred_test_y, pred_test_y2


lr_params = {'solver': 'lbfgs', 'C': 0.1}
results = run_cv_model(train_ohe, test_ohe, target, runLR, lr_params, auc, 'lr')

submission = pd.DataFrame({'id': test_id, 'target': results['test']})
submission.to_csv('submission.csv', index=False)