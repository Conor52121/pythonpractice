from kc_data_import import read_data
from kc_data_preprocessing import preprocessing
from kc_data_prediction import predict
 
 
def main():
    # 读取数据
    test = read_data('test_noLabel.csv')
    train = read_data('train.csv')
    # 数据预处理
    train_data, test_data = preprocessing(train, test)
 
    # 预测模型搭建
    pred_y = predict(train_data, test_data, is_shuffle=False)

    # 输出预测结果
    pred_y.to_csv('./pred3.csv',index=1, header=['price'])
 
 
if __name__ == '__main__':
    main()
