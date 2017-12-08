# target encoding 
import numpy as np
import pandas as pd

data = pd.read_csv("../input/train.csv")
X = data.drop(["id","target"],axis=1)
y = data["target"]
data_test = pd.read_csv("../input/test.csv")

# TRAIN 
# target encoding
ps_car_11_cat_0_1 = X.ps_car_11_cat.groupby(y).value_counts().unstack().apply(lambda x: x/x.sum(), axis=0).T.sort_values(ascending=False,by=1)
ps_car_11_cat_0_1["new_feat"] = ps_car_11_cat_0_1.ix[:,1].round(2)
ps_car_11_cat_map = ps_car_11_cat_0_1["new_feat"]
ps_car_11_cat_map_train = X["ps_car_11_cat"].map(ps_car_11_cat_map)

ps_car_04_cat_0_1 = X.ps_car_04_cat.groupby(y).value_counts().unstack().apply(lambda x: x/x.sum(), axis=0).T.sort_values(ascending=False,by=1)
ps_car_04_cat_0_1["new_feat"] = ps_car_04_cat_0_1.ix[:,1].round(2)
ps_car_04_cat_map = ps_car_04_cat_0_1["new_feat"]
ps_car_04_cat_map_train = X["ps_car_11_cat"].map(ps_car_04_cat_map)

ps_car_06_cat_0_1 = X.ps_car_06_cat.groupby(y).value_counts().unstack().apply(lambda x: x/x.sum(), axis=0).T.sort_values(ascending=False,by=1)
ps_car_06_cat_0_1["new_feat"] = ps_car_06_cat_0_1.ix[:,1].round(3)
ps_car_06_cat_map = ps_car_06_cat_0_1["new_feat"]
ps_car_06_cat_map_train = X["ps_car_11_cat"].map(ps_car_06_cat_map)

ps_car_09_cat_0_1 = X.ps_car_09_cat.groupby(y).value_counts().unstack().apply(lambda x: x/x.sum(), axis=0).T.sort_values(ascending=False,by=1)
ps_car_09_cat_0_1["new_feat"] = ps_car_09_cat_0_1.ix[:,1].round(3)
ps_car_09_cat_map = ps_car_09_cat_0_1["new_feat"]
ps_car_09_cat_map_train = X["ps_car_11_cat"].map(ps_car_09_cat_map)

ps_reg_01_0_1 = X.ps_reg_01.groupby(y).value_counts().unstack().apply(lambda x: x/x.sum(), axis=0).T.sort_values(ascending=False,by=1)
ps_reg_01_0_1["new_feat"] = ps_reg_01_0_1.ix[:,1].round(3)
ps_reg_01_map = ps_reg_01_0_1["new_feat"]
ps_reg_01_map_train = X["ps_car_11_cat"].map(ps_reg_01_map)

ps_reg_02_0_1 = X.ps_reg_02.groupby(y).value_counts().unstack().apply(lambda x: x/x.sum(), axis=0).T.sort_values(ascending=False,by=1)
ps_reg_02_0_1["new_feat"] = (ps_reg_02_0_1.ix[:,1].round(3))
ps_reg_02_map = ps_reg_02_0_1["new_feat"]
ps_reg_02_map_train = X["ps_car_11_cat"].map(ps_reg_02_map)

# export data for train
export_data_train = pd.concat([ps_car_11_cat_map_train,ps_car_04_cat_map_train,ps_car_06_cat_map_train,
	ps_car_09_cat_map_train,ps_reg_01_map_train,ps_reg_02_map_train], axis=1)
export_data_train.columns = ["ps_car_11_cat_map","ps_car_04_cat_map","ps_car_06_cat_map",
	"ps_car_09_cat_map","ps_reg_01_map","ps_reg_02_map"]

# TEST
ps_car_11_cat_map_test = data_test["ps_car_11_cat"].map(ps_car_11_cat_map)
ps_car_04_cat_map_test = data_test["ps_car_04_cat"].map(ps_car_04_cat_map)
ps_car_06_cat_map_test = data_test["ps_car_06_cat"].map(ps_car_06_cat_map)
ps_car_09_cat_map_test = data_test["ps_car_09_cat"].map(ps_car_09_cat_map)
ps_reg_01_map_test = data_test["ps_reg_01"].map(ps_reg_01_map)
ps_reg_02_map_test = data_test["ps_reg_02"].map(ps_reg_02_map)

# export data for test
export_data_test = pd.concat([ps_car_11_cat_map_test,ps_car_04_cat_map_test,ps_car_06_cat_map_test,
	ps_car_09_cat_map_test,ps_reg_01_map_test,ps_reg_02_map_test], axis=1)
export_data_test.columns = ["ps_car_11_cat_map","ps_car_04_cat_map","ps_car_06_cat_map",
	"ps_car_09_cat_map","ps_reg_01_map","ps_reg_02_map"]

# save
export_data_train.to_csv("../input/traget_ecd_train.csv", index=False)
export_data_test.to_csv("../input/traget_ecd_test.csv", index=False)















