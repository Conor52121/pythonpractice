from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error

'''一、读取文件'''
# csv文件路径，放在当前py文件统一路径(存放路径自己选择)
# 测试集和训练集的存放路径
train_file_path = 'train.csv'
test_file_path = 'test_noLabel.csv'
# pandas库读取训练集csv文件（训练集1460行，测试集1459行，评分的时候要一致，所以训练集要删掉一行）
# train_data = pd.read_csv(train_file_path).drop(0)
train_data = pd.read_csv(train_file_path)
# pandas库读取测试集csv文件
test_data = pd.read_csv(test_file_path)

'''二、确认预测特征变量和选择要训练的特征'''
# 确定要预测的特征变量（标签）
train_y = train_data.price
# print(train_y)
# 要训练的特征列表,LotArea:占地面积;OverallQual:整体的材料和成品质量;YearBuilt:最初施工日期;TotRmsAbvGrd:房间的总数(不包含浴室)
predictor_cols = ['rating','area_house','area_parking','floor','rating','floorage']
# 要训练的真正的数据
train_X = train_data[predictor_cols]
# 删除有缺失的行(这里选取的特征列表都没有缺失)
train_X = train_X.dropna(axis=0)
#print(train_X)
#print(train_y)


'''三、创建模型和训练'''
# 创建随机森林模型
my_model = RandomForestRegressor()
# 把要训练的数据丢进去，进行模型训练
my_model.fit(train_X,train_y)


'''四、用测试集预测房价'''
test_X = test_data[predictor_cols]
predicted_prices = my_model.predict(test_X)
predicted_prices_01 = my_model.predict(train_X)
# print(predicted_prices)

print(predicted_prices.shape)
print(train_y.shape)
'''五、使用(RMSE)均方对数误差是做评价指标'''
print(mean_squared_error(predicted_prices_01, train_y))

'''六、把预测的值按照格式保存为csv文件'''
my_submission = pd.DataFrame({'ID': test_data.ID, 'price': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
