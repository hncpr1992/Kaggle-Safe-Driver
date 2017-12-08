import numpy as np
import pandas as pd

data = pd.read_csv("../input/train.csv")
X = data.drop(["id","target"],axis=1)
y = data["target"]
data_test = pd.read_csv("../input/test.csv")








