# BasicPytorch

```
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset

from sklearn.model_selection import train_test_split

import os
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'

class Net_1(nn.Module):
    def __init__(self, n_intput, neural_num, d_prob=0.5):
        super(Net_1, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(n_intput, neural_num),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(neural_num, neural_num),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(neural_num, neural_num),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(neural_num, 1),
        )

    def forward(self, x):
        return self.linears(x)
    
    
class Net_2(nn.Module):
    def __init__(self, n_intput, neural_num, d_prob=0.5):
        super(Net_2, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(n_intput, neural_num),
            nn.ReLU(inplace=False),

            nn.Dropout(0.5),
            nn.Linear(neural_num, neural_num),
            nn.ReLU(inplace=False),

            nn.Dropout(0.5),
            nn.Linear(neural_num, neural_num),
            nn.ReLU(inplace=False),

            nn.Dropout(0.5),
            nn.Linear(neural_num, 1),
        )

    def forward(self, x):
        return self.linears(x)        
    
    
class Net_3(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net_3,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,input):
        out = self.hidden1(input)
        out = F.relu(out, inplace=False)
        
        out = self.dropout(out)
        out = self.hidden2(out)
        out = F.relu(out, inplace=False)

        out = self.dropout(out)
        out = self.hidden2(out)
        out = F.relu(out, inplace=False)

        out = self.dropout(out)
        out =self.predict(out)

        return out
    
    
class Net_4(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net_4,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,input):
        out = self.hidden1(input)
        out = F.relu(out, inplace=True)
        
        out = self.dropout(out)
        out = self.hidden2(out)
        out = F.relu(out, inplace=True)

        out = self.dropout(out)
        out = self.hidden2(out)
        out = F.relu(out, inplace=True)

        out = self.dropout(out)
        out =self.predict(out)

        return out


dfx = pd.DataFrame()
dfy = pd.DataFrame()

dfx['x'] = pd.Series(np.random.rand(1000)).astype(np.float32)
dfx['y'] = pd.Series(np.random.uniform(40,80,1000)).astype(np.float32)

dfy['z'] = dfx.apply(lambda x: pow(x['x'],2) + 3*x['y'] + random.random(), axis=1).astype(np.float32)

dfx = torch.tensor(dfx.astype(np.float32).values)
dfy = torch.tensor(dfy.astype(np.float32).values)

x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size = 0.15)

if len(y_train.shape) == 1:
    y_train = y_train.reshape(y_train.shape[0],1)
    
if len(y_test.shape) == 1:
    y_test = y_test.reshape(y_test.shape[0],1)    
    
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
#定义迭代器
train_data = DataLoader(dataset = train_set, batch_size = 64, shuffle=True)
test_data  = DataLoader(dataset = test_set, batch_size = 64, shuffle=False)    
   
```

然后开始执行

```
net1 = Net_1(2,1024,1)
print(net1)

# optimizer = torch.optim.SGD(net.parameters(),lr = 0.0001)
optimizer = torch.optim.Adam(net1.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()

for t in range(1000):
    #前向传播
    prediction = net1(x_train)
    #记录单批次一次batch的loss
    loss = loss_func(prediction,y_train)
    #反向传播
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if t%100 ==0:
        print('Loss = %.4f' % loss.data)
```
Net_1(  
  (linears): Sequential(  
    (0): Linear(in_features=2, out_features=1024, bias=True)  
    (1): ReLU(inplace=True)  
    (2): Dropout(p=0.5, inplace=False)  
    (3): Linear(in_features=1024, out_features=1024, bias=True)  
    (4): ReLU(inplace=True)  
    (5): Dropout(p=0.5, inplace=False)  
    (6): Linear(in_features=1024, out_features=1024, bias=True)  
    (7): ReLU(inplace=True)  
    (8): Dropout(p=0.5, inplace=False)  
    (9): Linear(in_features=1024, out_features=1, bias=True)  
  )  
)  
Loss = 33906.8633  
Loss = 334.8231  
Loss = 269.7854  
Loss = 248.1467  
Loss = 290.9594  
Loss = 238.1143  
Loss = 214.1551  
Loss = 216.3815  
Loss = 196.5995  
Loss = 267.9281  

```
net2 = Net_2(2,1024,1)
print(net2)

# optimizer = torch.optim.SGD(net.parameters(),lr = 0.0001)
optimizer = torch.optim.Adam(net2.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()

for t in range(1000):
    #前向传播
    prediction = net2(x_train)
    #记录单批次一次batch的loss
    loss = loss_func(prediction,y_train)
    #反向传播
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if t%100 ==0:
        print('Loss = %.4f' % loss.data)
```
Net_2(
  (linears): Sequential(
    (0): Linear(in_features=2, out_features=1024, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=1024, out_features=1024, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=1024, out_features=1024, bias=True)
    (7): ReLU()
    (8): Dropout(p=0.5, inplace=False)
    (9): Linear(in_features=1024, out_features=1, bias=True)
  )
)
Loss = 35718.5039
Loss = 329.2430
Loss = 270.2094
Loss = 282.4565
Loss = 245.1587
Loss = 220.8517
Loss = 239.9501
Loss = 263.8072
Loss = 210.4690
Loss = 209.2976


```
net3 = Net_3(2,1024,1)
print(net3)

# optimizer = torch.optim.SGD(net.parameters(),lr = 0.0001)
optimizer = torch.optim.Adam(net3.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()

for t in range(1000):
    #前向传播
    prediction = net3(x_train)
    #记录单批次一次batch的loss
    loss = loss_func(prediction,y_train)
    #反向传播
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
        
    if t%100 ==0:
        print('Loss = %.4f' % loss.data)
```
Net_3(
  (hidden1): Linear(in_features=2, out_features=1024, bias=True)
  (hidden2): Linear(in_features=1024, out_features=1024, bias=True)
  (predict): Linear(in_features=1024, out_features=1, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
)
Loss = 36181.4453
Loss = 452.1683
Loss = 349.2935
Loss = 287.5428
Loss = 269.7836
Loss = 243.0369
Loss = 308.7162
Loss = 269.6352
Loss = 308.8691
Loss = 277.1499


```
net4 = Net_4(2,1024,1)
print(net4)

optimizer = torch.optim.Adam(net4.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()

for t in range(1000):
    #前向传播
    prediction = net4(x_train)
    #记录单批次一次batch的loss
    loss = loss_func(prediction,y_train)
    #反向传播
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if t%100 ==0:
        print('Loss = %.4f' % loss.data)
```
Net_4(
  (hidden1): Linear(in_features=2, out_features=1024, bias=True)
  (hidden2): Linear(in_features=1024, out_features=1024, bias=True)
  (predict): Linear(in_features=1024, out_features=1, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
)
Loss = 34595.2148
Loss = 321.6986
Loss = 263.2131
Loss = 265.3299
Loss = 266.4159
Loss = 260.8026
Loss = 243.9910
Loss = 207.9348
Loss = 227.1938
Loss = 231.1912
