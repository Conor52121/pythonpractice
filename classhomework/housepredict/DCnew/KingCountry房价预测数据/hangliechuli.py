import pandas as pd
columns_train = ['date', 'price', 'bedroom']
df=pd.read_csv('train2.csv',header=0)
#df=pd.read_csv('train2.csv',names=columns_train)
#df=pd.read_csv('train2.csv')
df=df.drop(columns='ID',axis=1)
# print(df)
df=df.drop([0],axis=0)
# print(df)
#df=df.drop([0],axis=1)
print(df)