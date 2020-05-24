from kc_data_import import read_data
from kc_data_preprocessing import preprocessing
from kc_data_prediction import predict


from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error
 
def main():
    # 读取数据
    test = read_data('test_noLabel.csv')
    train = read_data('train.csv')
    # 数据预处理
    train_data, test_data = preprocessing(train, test)
 

    '''二、确认预测特征变量和选择要训练的特征'''
    # 确定要预测的特征变量（标签）
    train_y = train_data.price
    # print(train_y)
    # num_bedroom消去提高到45
    #   'num_bathroom',消去提高到46
    # 'area_parking'消去提高到了44.1
    # 'floor'消去提到了44.7
    # 'year_repair'提到了46
    # ()
    predictor_cols = ['area_parking','area_house','rating','floorage','area_basement','building_age','repair_age','b_b_ratio','f_c_ratio','f_c_ratio','radius','angle','year_2014','year_2015','is_repair_0.0','is_repair_1.0','have_basement_0.0','have_basement_1.0','floor_1.0','floor_1.5','floor_2.0','floor_2.5','floor_3.0','floor_3.5','num_bedroom_0','num_bedroom_1','num_bedroom_2','num_bedroom_3','num_bedroom_4','num_bedroom_5','num_bedroom_6','num_bedroom_7','num_bedroom_8','num_bedroom_9','num_bedroom_10']
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
    # my_submission = pd.DataFrame({'ID': test_data.ID, 'price': predicted_prices})
    my_submission = pd.DataFrame({'ID': test_data.ID, 'price': predicted_prices})
    # you could use any filename. We choose submission here
    my_submission.to_csv('submission.csv', index=False)


 
 
if __name__ == '__main__':
    main()



