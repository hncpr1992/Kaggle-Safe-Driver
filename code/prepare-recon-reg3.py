# reconstruct reg 03
import numpy as np
import pandas as pd

data = pd.read_csv("../input/train.csv")
X = data.drop(["id","target"],axis=1)
y = data["target"]
data_test = pd.read_csv("../input/test.csv")

def recon(reg):
    integer = int(np.round((40*reg)**2)) 
    for a in range(32):
        if (integer - a) % 31 == 0:
            A = a
    M = (integer - A)//31
    return A, M


ps_reg_A_train = X['ps_reg_03'].apply(lambda x: recon(x)[0])
ps_reg_M_train = X['ps_reg_03'].apply(lambda x: recon(x)[1])
ps_reg_A_train.replace(19,-1, inplace=True)
ps_reg_M_train.replace(51,-1, inplace=True)

ps_reg_A_test = data_test['ps_reg_03'].apply(lambda x: recon(x)[0])
ps_reg_M_test = data_test['ps_reg_03'].apply(lambda x: recon(x)[1])
ps_reg_A_test.replace(19,-1, inplace=True)
ps_reg_M_test.replace(51,-1, inplace=True)

# export data for train
export_data_train = pd.concat([ps_reg_A_train, ps_reg_M_train], axis=1)
export_data_train.columns = ["ps_reg_A","ps_reg_M"]

export_data_test = pd.concat([ps_reg_A_test, ps_reg_M_test], axis=1)
export_data_test.columns = ["ps_reg_A","ps_reg_M"]

# save
export_data_train.to_csv("../input/recon_reg3_train.csv", index=False)
export_data_test.to_csv("../input/recon_reg3_test.csv", index=False)








