import numpy as np
import pandas as pd

data = pd.read_csv("../input/train.csv")
X = data.drop(["id","target"],axis=1)
y = data["target"]
data_test = pd.read_csv("../input/test.csv")

# create nan encoders
ps_ind_02_na_train = pd.Series([1 if x == -1 else 0 for x in X["ps_ind_02_cat"]])
ps_ind_04_na_train = pd.Series([1 if x == -1 else 0 for x in X["ps_ind_04_cat"]])
ps_ind_05_cat_na_train = pd.Series([1 if x == -1 else 0 for x in X["ps_ind_05_cat"]])

ps_ind_02_na_test = pd.Series([1 if x == -1 else 0 for x in data_test["ps_ind_02_cat"]])
ps_ind_04_na_test = pd.Series([1 if x == -1 else 0 for x in data_test["ps_ind_04_cat"]])
ps_ind_05_cat_na_test = pd.Series([1 if x == -1 else 0 for x in data_test["ps_ind_05_cat"]])

# export
export_data_train = pd.concat([ps_ind_02_na_train, ps_ind_04_na_train, ps_ind_05_cat_na_train], axis=1)
export_data_train.columns = ["ps_ind_02_na", "ps_ind_04_na", "ps_ind_05_cat_na"]

export_data_test = pd.concat([ps_ind_02_na_test, ps_ind_04_na_test, ps_ind_05_cat_na_test], axis=1)
export_data_test.columns = ["ps_ind_02_na", "ps_ind_04_na", "ps_ind_05_cat_na"]
 
# save
export_data_train.to_csv("../input/nan_ecd_train.csv", index=False)
export_data_test.to_csv("../input/nan_ecd_test.csv", index=False)







