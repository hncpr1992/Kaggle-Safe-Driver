import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from multiprocessing import * 

from time import gmtime, strftime

np.random.seed(1234)

# original data
print("Load original data...")
data = pd.read_csv("../input/train.csv")
col_drop = data.columns[data.columns.str.startswith("ps_calc")]
col_extra_drop = ['ps_ind_10_bin','ps_ind_11_bin','ps_ind_13_bin','ps_car_10_cat',"ps_ind_12_bin"]
data.drop(col_drop, axis=1, inplace=True)
data.drop(col_extra_drop, axis=1, inplace=True)
X = data.drop(["id","target"],axis=1)
y = data["target"]
del data

data_test = pd.read_csv("../input/test.csv")
data_test.drop(col_drop, axis=1, inplace=True)
data_test.drop(col_extra_drop, axis=1, inplace=True)
test_id = data_test["id"]
data_test.drop("id", axis=1, inplace=True)

# feature engineering data
print("Load feature engineered data...")
traget_ecd_train = pd.read_csv("../input/traget_ecd_train.csv")
traget_ecd_test = pd.read_csv("../input/traget_ecd_test.csv")

recon_reg3_train = pd.read_csv("../input/recon_reg3_train.csv")
recon_reg3_test = pd.read_csv("../input/recon_reg3_test.csv")

one_hot_train = pd.read_csv("../input/one_hot_train.csv")
one_hot_test = pd.read_csv("../input/one_hot_test.csv")

interaction_train = pd.read_csv("../input/interaction_train.csv")
interaction_test = pd.read_csv("../input/interaction_test.csv")

nan_ecd_train = pd.read_csv("../input/nan_ecd_train.csv")
nan_ecd_test = pd.read_csv("../input/nan_ecd_test.csv")


# combine
print("Combine data...")
X = pd.concat([X, traget_ecd_train, recon_reg3_train, one_hot_train, interaction_train, nan_ecd_train], axis=1)
data_test = pd.concat([data_test, traget_ecd_test, recon_reg3_test, one_hot_test, interaction_test, nan_ecd_test], axis=1)

del traget_ecd_train
del recon_reg3_train
del one_hot_train
del interaction_train

del traget_ecd_test
del recon_reg3_test
del one_hot_test
del interaction_test

assert X.shape[1] == data_test.shape[1]
assert all(X.columns == data_test.columns)
print("Shape of X is:",X.shape)

# create test data
dtest = xgb.DMatrix(data_test.as_matrix())

# gini function
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score

# parameters of xgboost
params = {'eta': 0.07, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          "min_child_weight":6,"gamma":10,"reg_alpha":8,"reg_lambda":1.3,"scale_pos_weight":1.6,
          'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}

# np.random.seed(123)
# kfolds = 5
# pred = np.zeros(data_test.shape[0])
# X_in = X.as_matrix()
# y_in = y.as_matrix()

# # stratefied
# skf = StratifiedKFold(n_splits=kfolds)

# for i, (train_index, valid_index) in enumerate(skf.split(X_in, y_in)):
#     print("The {} fold".format(i))
#     print("###################################################")
    
#     # train test split
#     X_train, X_valid = X_in[train_index], X_in[valid_index]
#     y_train, y_valid = y_in[train_index], y_in[valid_index]
#     dtrain = xgb.DMatrix(X_train, label=y_train.reshape(-1,1))
#     dvalid = xgb.DMatrix(X_valid, label=y_valid.reshape(-1,1))
#     evallist = [(dtrain, 'train'), (dvalid, 'valid')]
    
#     mdl = xgb.train(params, dtrain, 
#                 500, evallist, early_stopping_rounds=100, 
#                 feval=gini_xgb, maximize=True, verbose_eval=50)
#     pred += mdl.predict(dtest, ntree_limit=mdl.best_ntree_limit) / kfolds

# # export
# fe_submit = pd.DataFrame({"id":test_id,"target":pred})
# time_now = strftime("%Y-%m-%d_%H:%M", gmtime())
# print(time_now)
# fe_submit.to_csv("../output/submission_{}.csv".format(time_now), index=False)


#####################################################################################
# model stacking forked from 
# https://www.kaggle.com/zusmani/lgb-esemble-xgb-be-in-top-100-with-lb-0-285

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
#                y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
#                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
#                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:,1]                

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res

        
# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 650
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8   
lgb_params['min_child_samples'] = 500
lgb_params['random_state'] = 99

lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['random_state'] = 99

lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['random_state'] = 99

#incorporated one more layer of my defined lgb params 
lgb_params4 = {}
lgb_params4['n_estimators'] = 1450
lgb_params4['max_bin'] = 20
lgb_params4['max_depth'] = 6
lgb_params4['learning_rate'] = 0.25 # shrinkage_rate
lgb_params4['boosting_type'] = 'gbdt'
lgb_params4['objective'] = 'binary'
lgb_params4['min_data'] = 500         # min_data_in_leaf
lgb_params4['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
lgb_params4['verbose'] = 0

lgb_model = LGBMClassifier(**lgb_params)
lgb_model2 = LGBMClassifier(**lgb_params2)
lgb_model3 = LGBMClassifier(**lgb_params3)
lgb_model4 = LGBMClassifier(**lgb_params4)
log_model = LogisticRegression()

stack = Ensemble(n_splits=3,
        stacker = log_model,
        base_models = (lgb_model, lgb_model2, lgb_model3, lgb_model4))        
        
y_pred = stack.fit_predict(X, y, data_test)        

lgbsub = pd.DataFrame()
lgbsub['id'] = test_id
lgbsub['target'] = y_pred
lgbsub.to_csv('../output/lgb_esm_submission.csv', index=False)

df1 = pd.read_csv('../output/lgb_esm_submission.csv')
df2 = pd.read_csv('../output/fe_single_reconstruct_1_2_3_4_6_params2.csv') 
df2.columns = [x+'_' if x not in ['id'] else x for x in df2.columns]
blend = pd.merge(df1, df2, how='left', on='id')
for c in df1.columns:
    if c != 'id':
        blend[c] = (blend[c]*0.07)  + (blend[c+'_']*0.03)
blend = blend[df1.columns]
blend['target'] = (np.exp(blend['target'].values) - 1.0).clip(0,1)
blend.to_csv('../output/final_submission.csv', index=False, float_format='%.6f')










































