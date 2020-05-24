import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
 
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
# pd.set_option('display.max_rows', None)
 
 
# 极坐标转换
def polar_coordinates(x, y, x_min, y_min):
    # 极坐标半径
    radius =  np.sqrt((x - x_min) ** 2 + (y - y_min) ** 2)
    # radius = np.sqrt((x ** 2+y ** 2))
 
    # 极坐标角度
    angle = np.arctan((y - y_min) / (x - x_min)) * 180 / np.pi
    # angle = np.arctan(y / x * 180 / np.pi)
 
    return radius, angle
 
 
# 极坐标地址
def get_radius_angle(loc_x, loc_y):
 
    x_min, y_min = loc_x.min(), loc_y.min()
    radius, angle = [], []
 
    for x, y in zip(loc_x, loc_y):
        radius.append(polar_coordinates(x, y, x_min, y_min)[0])
        angle.append(polar_coordinates(x, y, x_min, y_min)[1])
 
    radius = np.array(radius)
    angle = np.array(angle)
 
    return radius, angle
 
 
def preprocessing(train, test):
 
    # 目标售房价格
    temp_target = pd.DataFrame()
    temp_target['price'] = train.pop('price')
 
    # 合并训练集 测试集
    data_all = pd.concat([train, test])
    data_all.reset_index(inplace=True)
 
    # 
    temp_all = pd.DataFrame()
    columns = ['ID','num_bedroom', 'num_bathroom', 'floor', 'rating',
               'area_house', 'area_parking', 'floorage', 'area_basement',
               ]
    for col in columns:
        temp_all[col] = data_all[col]
 
    # 年份 季度 月份
    temp_all['year'] = data_all['sale_date'].apply(lambda x: x.year)
    temp_all['quarter'] = data_all['sale_date'].apply(lambda x: x.quarter)
    temp_all['month'] = data_all['sale_date'].apply(lambda x: x.month)
 
    # 房屋是否修复
    temp_all['is_repair'] = np.zeros((temp_all.shape[0], 1))
    for i in range(len(temp_all['is_repair'])):
        if data_all['year_repair'][i] > 0:
            temp_all['is_repair'][i] = 1
 
    # 房屋有无地下室
    temp_all['have_basement'] = np.zeros((temp_all.shape[0], 1))
    for i in range(len(temp_all['have_basement'])):
        if data_all['area_basement'][i] == 0:
            temp_all['have_basement'][i] = 1
 
    # 房龄
    temp_all['building_age'] = temp_all['year'] - data_all['year_built']
 
    # 上次修复后年数
    temp_all['repair_age'] = temp_all['year'] - data_all['year_repair']
    for i in range(len(temp_all['repair_age'])):
        if temp_all['repair_age'][i] == 2014 or temp_all['repair_age'][i] == 2015:
            temp_all['repair_age'][i] = temp_all['building_age'][i]
 
    # 卧室数/浴室数 比率
    data_all['num_bedroom'].replace(0, 1, inplace=True)
    data_all['num_bathroom'].replace(0, 1, inplace=True)
    temp_all['b_b_ratio'] = data_all['num_bedroom'] / data_all['num_bathroom']
 
    # 房屋面积/建筑面积 比率
    temp_all['f_c_ratio'] = temp_all['area_house'] / temp_all['floorage']
 
    # 房屋面积/停车面积 比率
    temp_all['f_p_ratio'] = temp_all['area_house'] / temp_all['floorage']
 
    # 经纬度 转换极坐标
    loc_x = data_all['longitude'].values
    loc_y = data_all['latitude'].values
    radius, angle = get_radius_angle(loc_x, loc_y)
    temp_all['radius'] = radius.round(decimals=8)
    temp_all['angle'] = angle.round(decimals=8)
    
    # 使用get_dummies进行one-hot编码
    temp_all = pd.get_dummies(temp_all, columns=['year', 'quarter', 'month',
                                                 'num_bedroom', 'num_bathroom', 'floor',
                                                 'is_repair', 'have_basement'
                                                 ])
 
    # 训练集  测试集划分
    temp_train = temp_all[temp_all.index < 10000]
    temp_test = temp_all[temp_all.index >= 10000]
    temp_train['price'] = temp_target['price']
 
    return temp_train, temp_test
