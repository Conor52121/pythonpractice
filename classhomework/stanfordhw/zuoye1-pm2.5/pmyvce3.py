'''
利用线性回归Linear Regression模型预测 PM2.5

特征工程中的特征选择与数据可视化的直观分析
通过选择的特征进一步建立回归模型
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''数据读取与预处理'''
# DataFrame类型
train_data = pd.read_csv("./Dataset/train.csv")
train_data.drop(['Date', 'stations', 'observation'], axis=1, inplace=True)

ItemNum=18
#训练样本features集合
X_Train=[]
#训练样本目标PM2.5集合
Y_Train=[]

for i in range(int(len(train_data)/ItemNum)):
    observation_data = train_data[i*ItemNum:(i+1)*ItemNum] #一天的观测数据
    for j in range(15):
        x = observation_data.iloc[:, j:j + 9]
        y = int(observation_data.iloc[9,j+9])
        # 将样本分别存入X_Train、Y_Train中
        X_Train.append(x)
        Y_Train.append(y)
print(X_Train)
print(Y_Train)

'''绘制散点图'''
x_AMB=[]
x_CH4=[]
x_CO=[]
x_NMHC=[]

x_NO=[]
x_NO2=[]
x_NOX=[]
x_O3=[]

x_PM10=[]
x_PM2Dot5=[]
x_RAINFALL=[]
x_RH=[]

x_SO2=[]
x_THC=[]
x_WD_HR=[]
x_WIND_DIREC=[]

x_WIND_SPEED=[]
x_WS_HR=[]

y=[]

for i in range(len(Y_Train)):
    y.append(Y_Train[i])
    x=X_Train[i]
    # print(type(x.iloc[0,0]))
    # 求各测项的平均值
    x_WIND_SPEED_sum = 0
    x_WS_HR_sum = 0
    for j in range(9):
        x_WIND_SPEED_sum = x_WIND_SPEED_sum + float(x.iloc[0, j])
        x_WS_HR_sum = x_WS_HR_sum + float(x.iloc[1, j])
    x_WIND_SPEED.append(x_WIND_SPEED_sum / 9)
    x_WS_HR.append(x_WS_HR_sum / 9)
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title('WIND_SPEED')
plt.scatter(x_WIND_SPEED, y)
plt.subplot(1, 2, 2)
plt.title('WS_HR')
plt.scatter(x_WS_HR, y)
plt.show()
x_SO2_sum = 0
x_THC_sum = 0
x_WD_HR_sum = 0
x_WIND_DIREC_sum = 0
for j in range(9):
    x_SO2_sum = x_SO2_sum + float(x.iloc[0, j])
    x_THC_sum = x_THC_sum + float(x.iloc[1, j])
    x_WD_HR_sum = x_WD_HR_sum + float(x.iloc[2, j])
    x_WIND_DIREC_sum = x_WIND_DIREC_sum + float(x.iloc[3, j])
x_SO2.append(x_SO2_sum / 9)
x_THC.append(x_THC_sum / 9)
x_WD_HR.append(x_WD_HR_sum / 9)
x_WIND_DIREC.append(x_WIND_DIREC_sum / 9)
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.title('SO2')
plt.scatter(x_SO2, y)
plt.subplot(2, 2, 2)
plt.title('THC')
plt.scatter(x_THC, y)
plt.subplot(2, 2, 3)
plt.title('WD_HR')
plt.scatter(x_WD_HR, y)
plt.subplot(2, 2, 4)
plt.title('WIND_DIREC')
plt.scatter(x_WIND_DIREC, y)
plt.show()
x_PM10_sum = 0
x_PM2Dot5_sum = 0
x_RAINFALL_sum = 0
x_RH_sum = 0
for j in range(9):
    x_PM10_sum = x_PM10_sum + float(x.iloc[0, j])
    x_PM2Dot5_sum = x_PM2Dot5_sum + float(x.iloc[1, j])
    x_RAINFALL_sum = x_RAINFALL_sum + float(x.iloc[2, j])
    x_RH_sum = x_RH_sum + float(x.iloc[3, j])
x_PM10.append(x_PM10_sum / 9)
x_PM2Dot5.append(x_PM2Dot5_sum / 9)
x_RAINFALL.append(x_RAINFALL_sum / 9)
x_RH.append(x_RH_sum / 9)
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.title('PM10')
plt.scatter(x_PM10, y)
plt.subplot(2, 2, 2)
plt.title('PM2.5')
plt.scatter(x_PM2Dot5, y)
plt.subplot(2, 2, 3)
plt.title('RAINFALL')
plt.scatter(x_RAINFALL, y)
plt.subplot(2, 2, 4)
plt.title('RH')
plt.scatter(x_RH, y)
plt.show()
x_AMB_sum=0
x_CH4_sum=0
x_CO_sum=0
x_NMHC_sum=0
for j in range(9):
    x_AMB_sum = x_AMB_sum + float(x.iloc[0,j])
    x_CH4_sum = x_CH4_sum + float(x.iloc[1, j])
    x_CO_sum = x_CO_sum + float(x.iloc[2, j])
    x_NMHC_sum = x_NMHC_sum + float(x.iloc[3, j])
x_AMB.append(x_AMB_sum / 9)
x_CH4.append(x_CH4_sum / 9)
x_CO.append(x_CO_sum / 9)
x_NMHC.append(x_NMHC_sum / 9)
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.title('AMB')
plt.scatter(x_AMB, y)
plt.subplot(2, 2, 2)
plt.title('CH4')
plt.scatter(x_CH4, y)
plt.subplot(2, 2, 3)
plt.title('CO')
plt.scatter(x_CO, y)
plt.subplot(2, 2, 4)
plt.title('NMHC')
plt.scatter(x_NMHC, y)
plt.show()