import numpy as np 
import pandas as pd


data = pd.read_excel("./train.xlsx")
data['name'] = data['Image_Path'].apply(lambda x: x.split("/")[-1].split(".")[0])

data.to_excel("./train_set.xlsx")