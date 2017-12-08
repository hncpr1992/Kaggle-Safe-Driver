import numpy as np
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("../input/train.csv")
X = data.drop(["id","target"],axis=1)
y = data["target"]
data_test = pd.read_csv("../input/test.csv")

# encoder
lb = preprocessing.LabelBinarizer()

# transform
ps_ind_02_cat_ecd_train = pd.DataFrame(lb.fit_transform(X["ps_ind_02_cat"]), 
    columns=["ps_ind_02_cat_{}".format(i) for i in range(X["ps_ind_02_cat"].value_counts().shape[0])])
ps_ind_05_cat_ecd_train = pd.DataFrame(lb.fit_transform(X["ps_ind_05_cat"]), 
    columns=["ps_ind_05_cat_{}".format(i) for i in range(X["ps_ind_05_cat"].value_counts().shape[0])])
ps_car_01_cat_ecd_train = pd.DataFrame(lb.fit_transform(X["ps_car_01_cat"]), 
    columns=["ps_car_01_cat_{}".format(i) for i in range(X["ps_car_01_cat"].value_counts().shape[0])])
ps_car_04_cat_ecd_train = pd.DataFrame(lb.fit_transform(X["ps_car_04_cat"]), 
    columns=["ps_car_04_cat_{}".format(i) for i in range(X["ps_car_04_cat"].value_counts().shape[0])])
ps_car_06_cat_ecd_train = pd.DataFrame(lb.fit_transform(X["ps_car_06_cat"]), 
    columns=["ps_car_06_cat_{}".format(i) for i in range(X["ps_car_06_cat"].value_counts().shape[0])])
ps_car_09_cat_ecd_train = pd.DataFrame(lb.fit_transform(X["ps_car_09_cat"]), 
    columns=["ps_car_09_cat_{}".format(i) for i in range(X["ps_car_09_cat"].value_counts().shape[0])])

ps_ind_02_cat_ecd_test = pd.DataFrame(lb.fit_transform(data_test["ps_ind_02_cat"]), 
    columns=["ps_ind_02_cat_{}".format(i) for i in range(data_test["ps_ind_02_cat"].value_counts().shape[0])])
ps_ind_05_cat_ecd_test = pd.DataFrame(lb.fit_transform(data_test["ps_ind_05_cat"]), 
    columns=["ps_ind_05_cat_{}".format(i) for i in range(data_test["ps_ind_05_cat"].value_counts().shape[0])])
ps_car_01_cat_ecd_test = pd.DataFrame(lb.fit_transform(data_test["ps_car_01_cat"]), 
    columns=["ps_car_01_cat_{}".format(i) for i in range(data_test["ps_car_01_cat"].value_counts().shape[0])])
ps_car_04_cat_ecd_test = pd.DataFrame(lb.fit_transform(data_test["ps_car_04_cat"]), 
    columns=["ps_car_04_cat_{}".format(i) for i in range(data_test["ps_car_04_cat"].value_counts().shape[0])])
ps_car_06_cat_ecd_test = pd.DataFrame(lb.fit_transform(data_test["ps_car_06_cat"]), 
    columns=["ps_car_06_cat_{}".format(i) for i in range(data_test["ps_car_06_cat"].value_counts().shape[0])])
ps_car_09_cat_ecd_test = pd.DataFrame(lb.fit_transform(data_test["ps_car_09_cat"]), 
    columns=["ps_car_09_cat_{}".format(i) for i in range(data_test["ps_car_09_cat"].value_counts().shape[0])])

# export
export_data_train = pd.concat([ps_ind_02_cat_ecd_train,ps_ind_05_cat_ecd_train,ps_car_01_cat_ecd_train,
              ps_car_04_cat_ecd_train,ps_car_06_cat_ecd_train,ps_car_09_cat_ecd_train], axis=1)

export_data_test = pd.concat([ps_ind_02_cat_ecd_test,ps_ind_05_cat_ecd_test,ps_car_01_cat_ecd_test,
              ps_car_04_cat_ecd_test,ps_car_06_cat_ecd_test,ps_car_09_cat_ecd_test], axis=1)

# save
export_data_train.to_csv("../input/one_hot_train.csv", index=False)
export_data_test.to_csv("../input/one_hot_test.csv", index=False)



