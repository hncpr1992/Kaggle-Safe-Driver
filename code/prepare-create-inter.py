import numpy as np
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("../input/train.csv")
X = data.drop(["id","target"],axis=1)
y = data["target"]
data_test = pd.read_csv("../input/test.csv")

def create_interactions(data, feat_list):
    lb = preprocessing.LabelBinarizer()
    container = data.ix[:,0]*0
    
    for comb in feat_list:
        feat = comb[0] + "_" + comb[1]
        new_feat = data[comb[0]].astype(str) + "+" + data[comb[1]].astype(str)
        new_feat = pd.DataFrame(lb.fit_transform(new_feat), 
                                columns=[(feat+"_{}").format(i+1) 
                                for i in range(new_feat.value_counts().shape[0])])
        container = pd.concat([container, new_feat], axis=1)
    return container.ix[:,1:]

feat_list = [
             ['ps_reg_01', 'ps_car_02_cat'],  
             ['ps_reg_01', 'ps_car_04_cat']
]

# transform
export_data_train = create_interactions(X, feat_list)
export_data_test = create_interactions(data_test, feat_list)

# save
export_data_train.to_csv("../input/interaction_train.csv", index=False)
export_data_test.to_csv("../input/interaction_test.csv", index=False)







