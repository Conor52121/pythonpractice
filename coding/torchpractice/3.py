#------------------------作者信息-----------------------------------------
# -*- coding: utf-8 -*-
# @Time: 2020/5/23 20:04
# @Author: wangshengkang
# @Version: 1.0
# @Filename: 2.py
# @Software: PyCharm
#--------------------------作者信息--------------------------------------
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable

data=np.load('boston_housing.npz')
train_x=data['x'][:404]
train_y=data['y'][:404]
valid_x=data['x'][404:]
valid_y=data['y'][404:]

train_x_pd=pd.DataFrame(train_x)
train_y_pd=pd.DataFrame(train_y)
valid_x_pd=pd.DataFrame(valid_x)
valid_y_pd=pd.DataFrame(valid_y)

print(train_x_pd.head(5))
print(train_y_pd.head(5))
print(train_x_pd.shape)
print(train_x_pd.shape[1])

min_max_scale=MinMaxScaler()
min_max_scale.fit(train_x_pd)
train_x_sc=min_max_scale.transform(train_x_pd)
#my_array1 = np.array(train_x_sc)
#my_tensor1 = torch.tensor(my_array1).float()
#x=Variable(my_tensor1,requires_grad=True)

#x=torch.from_numpy(train_x_sc).float()
x = torch.autograd.Variable(torch.from_numpy(train_x_sc))
x=x.float()

min_max_scale.fit(train_y_pd)
train_y_sc=min_max_scale.transform(train_y_pd)
#my_array2 = np.array(train_y_sc)
#my_tensor2 = torch.tensor(my_array2).float()
#y=Variable(my_tensor2,requires_grad=True)

#y=torch.from_numpy(train_y_sc).float()
y = torch.autograd.Variable(torch.from_numpy(train_y_sc))
y=y.float()



min_max_scale.fit(valid_x_pd)
valid_x_sc=min_max_scale.transform(valid_x_pd)
x2 = torch.autograd.Variable(torch.from_numpy(valid_x_sc))
x2=x2.float()

min_max_scale.fit(valid_y_pd)
valid_y_sc=min_max_scale.transform(valid_y_pd)
y2 = torch.autograd.Variable(torch.from_numpy(valid_y_sc))
y2=y2.float()


class house(nn.Module):
    def __init__(self):
        super(house,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(13,10),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(10,15),
            nn.ReLU(),

            nn.Linear(15,1),
        )

    def forward(self,x):
        out = self.fc(x)
        return out

model=house()
loss=nn.MSELoss(reduction='sum')
#loss=nn.MSELoss()
#loss=nn.MSELoss(reduce=True, size_average=True)
optimizer= torch.optim.Adam(model.parameters(),lr=0.001)
epochs=200

loss_total = []
iteration = []
for epoch in range(epochs):


    train_loss=0.0
    train_acc=0.0
    model.train()
    #for i in enumerate(x,y,0)
    train_pre=model(x)
    batch_loss=loss(y,train_pre)

    iteration.append(epoch)  # i是你的iter
    loss_total.append(batch_loss)  # total_loss.item()是你每一次inter输出的loss


    #train_acc += np.sum(np.argmax(train_pre.detach().numpy()) == x.detach().numpy())
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()
    #print('epoch %3d , loss %3d ,acc %3d' % (epoch,batch_loss,train_acc))
    print('epoch %3d , loss %3d' % (epoch, batch_loss))


print(iteration)
print(loss_total)
#plt.figure()
plt.plot(iteration, loss_total, label="loss")
plt.title('torch loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train'],loc='upper left')
#plt.draw()
plt.show()




#torch.save(model,'housetorch.pth')

torch.save(model.state_dict(), "housetorch.pth")

#buffer = io.BytesIO()
#torch.save(model, buffer)

#model=house()
model.load_state_dict(torch.load('housetorch.pth'))

# torch.save(model,'housetorch.pt')
# model=torch.load(model,'housetorch.pt')



model.eval()
valid_pre=model(x2)

#RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
valid_pre_numpy=valid_pre.detach().numpy()
min_max_scale.fit(valid_y_pd)
valid_pre_fg=min_max_scale.inverse_transform(valid_pre_numpy)
print(valid_y_pd.head(10))
print(valid_pre_fg)






