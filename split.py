from sklearn.model_selection import train_test_split
import pandas as pd

import chardet
with open('data/skill_builder_data.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

df = pd.read_csv('data/skill_builder_data.csv',encoding = "ISO-8859-1", low_memory=False)
train, test = train_test_split(df, test_size=0.2)
train.to_csv('data/skill_builder_train.csv')
test.to_csv('data/skill_builder_test.csv')