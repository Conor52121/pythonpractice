# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# 引入必要库
import pyecharts
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
plt.rcParams['font.sans-serif']=['SimHei']
#% matplotlib inline
 # 函数默认返回北京市2018年1月到12月的url
def get_url(city='beijing'):
    '''
    city为城市拼写的字符串，year为年份+月份
    '''
    for time in range(201801,201806):
        print('循环获取url')
        url = "http://lishi.tianqi.com/{}/{}.html".format(city,time)
        yield url
def get_datas(urls = get_url()):
    print('get data')
    cookie = {"cityPy":"UM_distinctid=171f2280ef23fb-02a4939f3c1bd4-335e4e71-144000-171f2280ef3dab; Hm_lvt_ab6a683aa97a52202eab5b3a9042a8d2=1588905651; CNZZDATA1275796416=871124600-1588903268-%7C1588990372; Hm_lpvt_ab6a683aa97a52202eab5b3a9042a8d2=1588994046"}
    header = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3756.400 QQBrowser/10.5.4039.400"}
    for url in urls:
        html = requests.get(url = url, headers = header, cookies=cookie)
        soup = BeautifulSoup(html.content, 'html.parser')
        date = soup.select('.lishitable_content clearfix li div')
        print(date)
        max_temp = soup.select('.lishitable_content clearfix li div')
        print(max_temp)
        min_temp = soup.select("#tool_site > div.lishtiable > ul > li:nth-of-type(3)")
        weather = soup.select("#tool_site > div.lishtiable > ul > li:nth-of-type(4)")
        wind_direction = soup.select("#tool_site > div.tqtongji2 > ul > li:nth-of-type(5)")
        date = [x.text for x in date]
        max_temp = [x.text for x in max_temp[1:]]
        min_temp = [x.text for x in min_temp[1:]]
        weather = [x.text for x in weather[1:]]
        wind_direction = [x.text for x in wind_direction[1:]]
        yield pd.DataFrame([date,max_temp,min_temp,weather,wind_direction]).T

# 获取数据方法
def get_result():
    print('get result')
    result = pd.DataFrame()
    for data in get_datas():
        print('循环获取结果')
        result = result.append(data)
    return result
# 执行方法，获取数据
print('开始')
result = get_result()

# 是否存在非空数据
print('空数据有',result.isnull().any().sum())

# 简单查看下爬取到的数据
result.head(5) 
# 改一下列名
result.columns = ["日期","最高温度","最低温度","天气状况","风向"]
# 由于提取的默认是字符串，所以这里更改一下数据类型
result['日期'] = pd.to_datetime(result['日期'])
result["最高温度"] = pd.to_numeric(result['最高温度'])
result["最低温度"] = pd.to_numeric(result['最低温度'])
result["平均温度"] = (result['最高温度'] + result['最低温度'])/2
# 看一下更改后的数据状况
result.info()
print(result)
# 温度的分布
sns.distplot(result['平均温度'])
# 天气状况分布
sns.countplot(result['天气状况'])
# 按月份统计降雨和没有降雨的天气数量


result['是否降水'] = result['天气状况'].apply(lambda x:'未降水' if x in ['晴','多云','阴','雾','浮尘','霾','扬沙'] else '降水')
rain = result.groupby([result['日期'].apply(lambda x:x.month),'是否降水'])['是否降水'].count()

month = [str(i)+"月份" for i in range(1,13)]
is_rain = [rain[i]['降水'] if '降水' in rain[i].index else 0 for i in range(1,13)]
no_rain = [rain[i]['未降水'] if '未降水' in rain[i].index else 0  for i in range(1,13)]

line = pyecharts.Line("各月降水天数统计")

line.add(
    "降水天数",
    month,
    is_rain,
    is_fill=True,
    area_opacity=0.7,
    is_stack = True)

line.add(
    "未降水天数",
    month,
    no_rain,
    is_fill=True,
    area_opacity=0.7,
    is_stack = True)
line

# 按照月份查看最高、最低、平均温度的走势
result.groupby(result['日期'].apply(lambda x:x.month)).mean().plot(kind='line')

directions = ['北风', '西北风', '西风', '西南风', '南风', '东南风', '东风', '东北风']
schema = []
v = []
days = result['风向'].value_counts()
for d in directions:
    schema.append((d,100))
    v.append(days[d])
v = [v]
radar = pyecharts.Radar()
radar.config(schema)
radar.add("风向统计", v, is_axisline_show=True)
radar.render('wind.html')

plt.show()
