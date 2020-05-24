import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.feature_extraction import DictVectorizer
import warnings
warnings.filterwarnings('ignore')
 
 
def feat_standard(data):
    st_scaler = StandardScaler()
    st_scaler.fit(data)
    data = st_scaler.transform(data)
    return data
 
 
def feat_normalizer(data):
    no_scaler = Normalizer()
    no_scaler.fit(data)
    data = no_scaler.transform(data)
    return data
 
 
def feat_encoder(data, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(data[c].values))
        data[c] = lbl.transform(list(data[c].values))
 
    return data
 
 
def feat_dictvectorizer(train_x, valid_x):
    dict_vec = DictVectorizer(sparse=False)
    train_x = dict_vec.fit_transform(train_x.to_dict(orient='record'))
    valid_x = dict_vec.transform(valid_x.to_dict(orient='record'))
 
    return train_x, valid_x
 
 
def mse_func(y_true, y_predict):
    assert isinstance(y_true, list), 'y_true must be type of list'
    assert isinstance(y_predict, list), 'y_true must be type of list'
 
    m = len(y_true)
    squared_error = 0
    for i in range(m):
        error = y_true[i] - y_predict[i]
        squared_error = squared_error + error ** 2
    mse = squared_error / (10000 * m)
    return mse
 
 
def predict(train_, valid_, is_shuffle=True):
    print(f'data shape:\ntrain--{train_.shape}\nvalid--{valid_.shape}')
    folds = KFold(n_splits=5, shuffle=is_shuffle, random_state=1024)
    pred = [k for k in train_.columns if k not in ['price']]
    sub_preds = np.zeros((valid_.shape[0], folds.n_splits))
    print(f'Use {len(pred)} features ...')
    res_e = []
 
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_, train_['price']), start=1):
        print(f'the {n_fold} training start ...')
        train_x, train_y = train_[pred].iloc[train_idx], train_['price'].iloc[train_idx]
        valid_x, valid_y = train_[pred].iloc[valid_idx], train_['price'].iloc[valid_idx]
 
        print('数据标准化...')
        # feat_st_cols = ['area_house', 'area_parking', 'floorage', 'building_age']
        # train_x[feat_st_cols] = feat_standard(train_x[feat_st_cols])
        # valid_x[feat_st_cols] = feat_standard(valid_x[feat_st_cols])
 
        train_x, valid_x = feat_dictvectorizer(train_x, valid_x)
 
        dt_stump = DecisionTreeRegressor(max_depth=30,
                                         min_samples_split=15,
                                         min_samples_leaf=10,
                                         max_features=50,
                                         random_state=11,
                                         max_leaf_nodes=350)
 
        reg = AdaBoostRegressor(base_estimator=dt_stump, n_estimators=100)
 
        reg.fit(train_x, train_y)
 
        train_pred = reg.predict(valid_x)
        tmp_score = mse_func(list(valid_y), list(train_pred))
        res_e.append(tmp_score)
 
        sub_preds[:, n_fold - 1] = reg.predict(valid_[pred])
 
    print('5 folds 均值：', np.mean(res_e))
    valid_['price'] = np.mean(sub_preds, axis=1)
    return valid_['price']
