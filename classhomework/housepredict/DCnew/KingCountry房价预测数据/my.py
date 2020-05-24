import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score

train_file_path = 'train.csv'
train=pd.read_csv(train_file_path, header=1)
print(train)
#pred_file_path='kc_pred.csv'
#pred=pd.read_csv(pred_file_path,header=0)
#print(pred)
#pred.to_csv('./pred.csv', index=1, header=0)
# train.info()
# print( train.isnull().sum())
# print('==============================')
# print( train.duplicated())
# print('==============================')
# print(train['price'].describe())
# plt.subplots(figsize=(12,9))
# sns.distplot(train['price'])
# train1 =pd.read_csv(r'train.csv', header=0, )
# test1 = pd.read_csv(r'test_noLabel.csv', header=0, )
# # print(train1,test1)
# print("Skewness: %f" %train['price'].skew()) #偏度
# print("Kurtosis: %f" %train['price'].kurt()) #峰度
# #
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False
# corrmat = train.corr()
# plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat, vmax=0.9, square=True,center=2.0)
# Corr = train.corr()
# Corr[Corr['price']>0.1]
# sns.set(font='SimHei')
# cols = ['sale_date','price','num_bedroom','num_bathroom','area_house','area_parking','floor','rating','floorage','area_basement','year_built','year_repair','latitude','longitude']
# sns.set(font='SimHei')
# cols = ['sale_date','price','num_bedroom','num_bathroom','area_house','area_parking','floor','rating','floorage','area_basement','year_built','year_repair','latitude','longitude']
# sns.pairplot(train[cols],size = 2.5)
# plt.show()

# def Data(IO):
#     X=pd.read_excel(IO)
#     Y = X['销售价格']
#     X= X.drop(['销售价格'],axis = 1)
#     X_train, X_test, y_train, y_test = \
#     cross_validation.train_test_split( X, Y, test_size=0.3, random_state=0)
#     return (X_train, X_test, y_train, y_test)
#
# def RF(X_train, X_test, y_train, y_test): #随机森林
#     from sklearn.ensemble import RandomForestClassifier
#     model= RandomForestClassifier(n_estimators=100)
#     model.fit(X_train, y_train)
#     predicted= model.predict(X_test)
#     score = accuracy_score(y_test, predicted)
#     return (score)
#
# import pandas as pd
# train =pd.read_csv(r'E:/kc_train.csv', header=None, names=['销售日期', '销售价格', '卧室数', '浴室数', '房屋面积', '停车面积', '楼层数', '房屋评分', '建筑面积', '地下室面积', '建筑年份','修复年份', '纬度', '经度'])
# test = pd.read_csv(r'E:/kc_test.csv', header=None, names=['销售日期', '卧室数', '浴室数', '房屋面积', '停车面积', '楼层数', '房屋评分', '建筑面积', '地下室面积', '建筑年份','修复年份', '纬度', '经度'])
# print(train,test)
#
# selected_features = ['销售日期', '卧室数', '浴室数', '房屋面积', '停车面积', '楼层数', '房屋评分', '建筑面积', '地下室面积', '建筑年份','修复年份', '纬度', '经度']
# X_train = train[selected_features]
# X_test = test[selected_features]
# y_train = train['销售价格']
#
# from sklearn.feature_extraction import DictVectorizer
# dict_vec = DictVectorizer(sparse=False)
# X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
# X_test = dict_vec.transform(X_test.to_dict(orient='record'))
# e)
# from sklearn.ensemble import RandomForestRegressor
# #from sklearn.ensemble import GradientBoostingRegressor
# rfr = RandomForestRegressor()
# #rfr = GradientBoostingRegressor()
# rfr.fit(X_train, y_train)
# rfr_y_predict = rfr.predict(X_test)
#
# rfr_submission = pd.DataFrame({'Id': test['销售日期'], 'SalePrice': rfr_y_predict})
# rfr_submission.to_csv('E:/submission.csv', index=Fals