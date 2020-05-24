import os
import pandas as pd
 
 
def assert_msg(condition, msg):
    if not condition:
        raise Exception(msg)
 
 
def read_data(filename):
    # 获取数据路径
    file_path = os.path.join(os.path.dirname(__file__), filename)
 
    # 判定文件是否存在
    assert_msg(file_path, '文件不存在')
 
    # 返回CSV文件
    return pd.read_csv(file_path,
                       header=0,
                       parse_dates=[1],  # 20150101 转换成日期值 2015-01-01
                       infer_datetime_format=True
                       )
