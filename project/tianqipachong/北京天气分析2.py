# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:55:43 2020

@author: LEGION
"""

# 引入必要库
import pyecharts
import matplotlib.pyplot as plt
from pyecharts.charts import Radar
from pyecharts.charts import Line
from pyecharts import options as opts
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as plt
import numpy as np
import time
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False    # 用来正常显示负号(X轴)

class item:
    def __init__(self):
        self.date=list()#日期
        self.max_temp = list()#最高温
        self.min_temp=list()#最低温
        self.weather=list()#天气
        self.wind_direction=list()#风向      
Data_Box=item()#数据盒子


 # 函数默认返回北京市5年内：2014年1月到2018年12月的url
def get_url(city='beijing'):
    '''
    city为城市拼写的字符串，year为年份+月份
    '''
    # 时间为5年,用date列表遍历
    years = [201401,201501,201601,201701,201801]
    for year in years:
        for time in range(year,year+12):
            url = "http://lishi.tianqi.com/{}/{}.html".format(city,time)
            yield url

def get_datas():
    urls = get_url()
    header = { 'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Encoding':'gzip, deflate',
        'Accept-Language':'zh-CN,zh;q=0.9',
        'Cache-Control':'max-age=0',
        'Connection':'keep-alive',
        'Host':'lishi.tianqi.com', 
        'Cookie':'UM_distinctid=1721c874da5e3-027b2ecf9054b1-b383f66-100200-1721c874da6fb; CNZZDATA1275796416=1935459176-1589614548-%7C1589625348; Hm_lvt_ab6a683aa97a52202eab5b3a9042a8d2=1589627577,1589627579,1589627784,1589627812; Hm_lpvt_ab6a683aa97a52202eab5b3a9042a8d2=1589627931',
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3756.400 QQBrowser/10.5.4039.400"}
    proxies = {"http:":"http://121.232.146.184","https:":"https://144.255.48.197", }
    for url in urls:
        html = requests.get(url = url, headers = header, proxies =proxies)
        soup = BeautifulSoup(html.text, 'lxml')
        try:
            # 网站正常显示时
            ul = soup.find_all("ul",class_='lishitable_content clearfix')[0]
            # 最后一个li标签不是天气数据
            lis = ul.find_all("li")[:-1]
            for li in lis:
                div = li.find_all("div")
                Data_Box.date.append(div[0].text)
                Data_Box.max_temp.append(div[1].text)
                Data_Box.min_temp.append(div[2].text)
                Data_Box.weather.append(div[3].text)
                Data_Box.wind_direction.append(div[4].text.split(" ")[0])
        except:
            # 网站不正常显示时
            ul = soup.find_all("ul",class_='thrui')[0]
            lis = ul.find_all("li")
            for li in lis:
                div = li.find_all("div")
                Data_Box.date.append(div[0].text.split("  ")[0])
                Data_Box.max_temp.append(div[1].text.replace("℃",""))
                Data_Box.min_temp.append(div[2].text.replace("℃",""))
                Data_Box.weather.append(div[3].text)
                Data_Box.wind_direction.append(div[4].text.split(" ")[0])
#         print(url)
        time.sleep(2)
    return "数据获取完毕"

# 获取dataframe
def get_result():
    get_datas()
    result = pd.DataFrame({"日期":Data_Box.date,"最高温度":Data_Box.max_temp,"最低温度":Data_Box.min_temp,"天气状况":Data_Box.weather,"风向":Data_Box.wind_direction})
    return result


# 执行方法，获取数据
result = get_result()


# 是否存在非空数据
print('空数据有',result.isnull().any().sum())

# 简单查看下爬取到的数据
result.head(20) 
# 由于提取的默认是字符串，所以这里更改一下数据类型
result['日期'] = pd.to_datetime(result['日期'])
result["最高温度"] = pd.to_numeric(result['最高温度'])
result["最低温度"] = pd.to_numeric(result['最低温度'])
result["平均温度"] = (result['最高温度'] + result['最低温度'])/2
# 看一下更改后的数据状况
result.dtypes
# 温度的分布图
sns.distplot(result['平均温度'])

# 按月份统计降雨和没有降雨的天气数量
result['是否降水'] = result['天气状况'].apply(lambda x:'未降水' if x in ['晴','多云','阴','雾','浮尘','霾','扬沙'] else '降水')
rain = result.groupby([result['日期'].apply(lambda x:x.month),'是否降水'])['是否降水'].count()

month = [str(i)+"月份" for i in range(1,13)]
# 每月下雨天数
is_rain = [rain[i]['降水'] if '降水' in rain[i].index else 0 for i in range(1,13)]
# 每月不下雨天数
no_rain = [rain[i]['未降水'] if '未降水' in rain[i].index else 0  for i in range(1,13)]
# 作柱形图,每月降雨与不降雨天数对比
line = pd.DataFrame({'降水天数':is_rain,'未降水天数':no_rain}, index=[x for x in range(1,13)]) 
line.plot.bar()
result['year'] = result['日期'].dt.year  # 增加年份列
result['month'] = result['日期'].dt.month   # 增加月份列

# 按照年月查看最高、最低温度、平均温度的走势
max_t = result['最高温度'].groupby([result['year'],result['month']]).mean()
mix_t = result['最低温度'].groupby([result['year'],result['month']]).mean()
avr_t = result['平均温度'].groupby([result['year'],result['month']]).mean()
yeardf= pd.DataFrame(result['year'])
years = yeardf.drop_duplicates()['year'].tolist()
# 最高温折线图，保存为Line_maxT.html
line_max = (
        Line()
        .add_xaxis(month)
        .add_yaxis(str(years[0]), max_t[years[0]].tolist())
        .add_yaxis(str(years[1]), max_t[years[1]].tolist())
        .add_yaxis(str(years[2]), max_t[years[2]].tolist())
        .add_yaxis(str(years[3]), max_t[years[3]].tolist())
        .add_yaxis(str(years[4]), max_t[years[4]].tolist())
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title="Line-最高温度"),
        yaxis_opts=opts.AxisOpts(name='最高温度情况(℃)',splitline_opts=opts.SplitLineOpts(is_show=True)),
        xaxis_opts=opts.AxisOpts(name='月份'))
        .render("Line_maxT.html")
)
# 最低温折线图,保存为Line_mixT.html
line_mix = (
        Line()
        .add_xaxis(month)
        .add_yaxis(str(years[0]), mix_t[years[0]].tolist())
        .add_yaxis(str(years[1]), mix_t[years[1]].tolist())
        .add_yaxis(str(years[2]), mix_t[years[2]].tolist())
        .add_yaxis(str(years[3]), mix_t[years[3]].tolist())
        .add_yaxis(str(years[4]), mix_t[years[4]].tolist())
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title="Line-最低温度"),
        yaxis_opts=opts.AxisOpts(name='最低温度情况(℃)',splitline_opts=opts.SplitLineOpts(is_show=True)),
        xaxis_opts=opts.AxisOpts(name='月份'))
        .render("Line_mixT.html")
)
# 平均温度折线图,保存为Line_avrT.html
line_avr = (
        Line()
        .add_xaxis(month)
        .add_yaxis(str(years[0]), avr_t[years[0]].tolist())
        .add_yaxis(str(years[1]), avr_t[years[1]].tolist())
        .add_yaxis(str(years[2]), avr_t[years[2]].tolist())
        .add_yaxis(str(years[3]), avr_t[years[3]].tolist())
        .add_yaxis(str(years[4]), avr_t[years[4]].tolist())
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title="Line-平均温度"),
        yaxis_opts=opts.AxisOpts(name='平均温度情况(℃)',splitline_opts=opts.SplitLineOpts(is_show=True)),
        xaxis_opts=opts.AxisOpts(name='月份'))
        .render("Line_avrT.html")
)


# 风向雷达图
directions = ['北风', '西北风', '西风', '西南风', '南风', '东南风', '东风', '东北风']
v1 = [[int(result[(result["year"]==years[0])&(result['风向'].str.contains(i))]['日期'].count()) for i in directions]]
v2 = [[int(result[(result["year"]==years[1])&(result['风向'].str.contains(i))]['日期'].count()) for i in directions]]
v3 = [[int(result[(result["year"]==years[2])&(result['风向'].str.contains(i))]['日期'].count()) for i in directions]]
v4 = [[int(result[(result["year"]==years[3])&(result['风向'].str.contains(i))]['日期'].count()) for i in directions]]
v5 = [[int(result[(result["year"]==years[4])&(result['风向'].str.contains(i))]['日期'].count()) for i in directions]]
c_schema = [
    {"name": directions[0], "max": 300, "min": 0},
    {"name": directions[1], "max": 300, "min": 0},
    {"name": directions[2], "max": 300, "min": 0},
    {"name": directions[3], "max": 300, "min": 0},
    {"name": directions[4], "max": 300, "min": 0},
    {"name": directions[5], "max": 300, "min": 0},
    {"name": directions[6], "max": 300, "min": 0},
    {"name": directions[7], "max": 300, "min": 0},
]
c = (
    Radar()
    .add_schema(schema=c_schema, shape="circle")
    .add(str(years[0]), v1, color="#b3e4a1")
    .add(str(years[1]), v2, color="#CCCCCC")
    .add(str(years[2]), v3, color="#f9713c")
    .add(str(years[3]), v4, color="#5CACEE")
    .add(str(years[4]), v5, color="#CD0000")
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(title_opts=opts.TitleOpts(title="Radar-5年风向统计"))
    .render("radar_wind.html")
)

# 堆叠图
condions = ['多云','雨','晴','阴','雾','雪']
condiondf = pd.DataFrame([
    [result[(result["year"]==years[0])&(result['天气状况'].str.contains(i))]['日期'].count() for i in condions],
    [result[(result["year"]==years[1])&(result['天气状况'].str.contains(i))]['日期'].count() for i in condions],
    [result[(result["year"]==years[2])&(result['天气状况'].str.contains(i))]['日期'].count() for i in condions],
    [result[(result["year"]==years[3])&(result['天气状况'].str.contains(i))]['日期'].count() for i in condions],
    [result[(result["year"]==years[4])&(result['天气状况'].str.contains(i))]['日期'].count() for i in condions]
],
    index=[str(years[0]),str(years[1]),str(years[2]),str(years[3]),str(years[4])],columns=pd.Index([condions[0],condions[1],condions[2],condions[3],condions[4],condions[5]],name="天气状况")) 
condiondf.plot(kind='barh',stacked=True)