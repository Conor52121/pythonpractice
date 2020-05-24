# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# 引入必要库
import pyecharts
from pyecharts.charts import Radar
from pyecharts import options as opts
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号(X轴)


class item:
    def __init__(self):
        self.date = list()  # 日期
        self.max_temp = list()  # 最高温
        self.min_temp = list()  # 最低温
        self.weather = list()  # 天气
        self.wind_direction = list()  # 风向


Data_Box = item()  # 数据盒子


# 函数默认返回北京市2018年1月到12月的url
def get_url(city='beijing'):
    '''
    city为城市拼写的字符串，year为年份+月份
    '''
    for time in range(201901, 201913):
        url = "http://lishi.tianqi.com/{}/{}.html".format(city, time)
        yield url


def get_datas():
    urls = get_url()
    cookie = {
        "cityPy": "UM_distinctid=171f2280ef23fb-02a4939f3c1bd4-335e4e71-144000-171f2280ef3dab; Hm_lvt_ab6a683aa97a52202eab5b3a9042a8d2=1588905651; CNZZDATA1275796416=871124600-1588903268-%7C1588990372; Hm_lpvt_ab6a683aa97a52202eab5b3a9042a8d2=1588994046"}
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3756.400 QQBrowser/10.5.4039.400"}
    for url in urls:
        html = requests.get(url=url, headers=header, cookies=cookie)
        soup = BeautifulSoup(html.text, 'html.parser')
        ul = soup.find_all("ul", class_='lishitable_content clearfix')[0]
        lis = ul.find_all("li")[:-1]
        for li in lis:
            # 最后一个li标签不是天气数据
            div = li.find_all("div")
            Data_Box.date.append(div[0].text)
            Data_Box.max_temp.append(div[1].text)
            Data_Box.min_temp.append(div[2].text)
            Data_Box.weather.append(div[3].text)
            Data_Box.wind_direction.append(div[4].text.split(" ")[0])
    return "数据获取完毕"


# 获取dataframe
def get_result():
    get_datas()
    result = pd.DataFrame(
        {"日期": Data_Box.date, "最高温度": Data_Box.max_temp, "最低温度": Data_Box.min_temp, "天气状况": Data_Box.weather,
         "风向": Data_Box.wind_direction})
    return result


# 执行方法，获取数据
result = get_result()

# 是否存在非空数据
print('空数据有', result.isnull().any().sum())

# 简单查看下爬取到的数据
print(result.head(20))
# 由于提取的默认是字符串，所以这里更改一下数据类型
result['日期'] = pd.to_datetime(result['日期'])
result["最高温度"] = pd.to_numeric(result['最高温度'])
result["最低温度"] = pd.to_numeric(result['最低温度'])
result["平均温度"] = (result['最高温度'] + result['最低温度']) / 2
# 看一下更改后的数据状况
print(result.dtypes)
# 温度的分布
sns.distplot(result['平均温度'])
# 天气状况分布
# sns.countplot(result['天气状况'])
df = result.groupby(['天气状况'])['日期'].count()
df_bar = pd.DataFrame(df)
df_bar.plot.bar()
# # 按月份统计降雨和没有降雨的天气数量


result['是否降水'] = result['天气状况'].apply(lambda x: '未降水' if x in ['晴', '多云', '阴', '雾', '浮尘', '霾', '扬沙'] else '降水')
rain = result.groupby([result['日期'].apply(lambda x: x.month), '是否降水'])['是否降水'].count()

month = [str(i) + "月份" for i in range(1, 13)]
# 每月下雨天数
is_rain = [rain[i]['降水'] if '降水' in rain[i].index else 0 for i in range(1, 13)]
# 每月不下雨天数
no_rain = [rain[i]['未降水'] if '未降水' in rain[i].index else 0 for i in range(1, 13)]

line = pd.DataFrame({'降水天数': is_rain, '未降水天数': no_rain}, index=[x for x in range(1, 13)])
line.plot()

# 按照月份查看最高、最低、平均温度的走势
result.groupby(result['日期'].apply(lambda x: x.month)).mean().plot(kind='line')

# 风向雷达图
directions = ['北风', '西北风', '西风', '西南风', '南风', '东南风', '东风', '东北风']
labels = np.array(directions)  # 标签
dataLenth = 8  # 数据长度
v = []
days = result['风向'].value_counts()
for d in directions:
    v.append(days[d])
data_radar = np.array(v)  # 数据
angles = np.linspace(0, 2 * np.pi, dataLenth, endpoint=False)  # 分割圆周长
data_radar = np.concatenate((data_radar, [data_radar[0]]))  # 闭合
angles = np.concatenate((angles, [angles[0]]))  # 闭合
plt.polar(angles, data_radar, 'bo-', linewidth=1)  # 做极坐标系
plt.thetagrids(angles * 180 / np.pi, labels)  # 做标签
plt.fill(angles, data_radar, facecolor='r', alpha=0.25)  # 填充
plt.ylim(0, 100)  # 设置最大、最小值
plt.title('wind')
plt.savefig("wind.png")
plt.show()