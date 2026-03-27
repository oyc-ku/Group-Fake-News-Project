import pandas as pd
data=pd.read_csv("../data/data_stemmed.csv", index_col=0)
data=data.sample(frac=1,random_state=0)
trainset=data.iloc[:int(len(data)*0.80)]
validation=data.iloc[int(len(data)*0.80):int(len(data)*0.90)]
test=data.iloc[int(len(data)*0.90):]

trainset.to_csv("995,000_trainset_80.csv", index=False)
validation.to_csv("995,000_vail_10_set.csv", index=False)
test.to_csv("995,000_test_10_set.csv", index=False)
