from kc_data_import import read_data
from kc_data_preprocessing import preprocessing
from kc_data_prediction import predict
 
 
def main():
    # 读取数据
    columns_test = ['date', 'bedroom', 'bathroom', 'floor space', 'parking space', 'floor', 'grade',
                     'covered area', 'basement area', 'build year', 'repair year', 'longitude', 'latitude']
    columns_train = ['date', 'price', 'bedroom', 'bathroom', 'floor space', 'parking space', 'floor', 'grade',
                     'covered area', 'basement area', 'build year', 'repair year', 'longitude', 'latitude']
    test = read_data('kc_test.csv', columns_test)
    train = read_data('kc_train.csv', columns_train)
 
    # 数据预处理
    train_data, test_data = preprocessing(train, test)
 
    # 预测模型搭建
    pred_y = predict(train_data, test_data, is_shuffle=False)
 
    # 输出预测结果
    pred_y.to_csv('./kc_pred.csv', index=False, header=['price'])
 
 
if __name__ == '__main__':
    main()
