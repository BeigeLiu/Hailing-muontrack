# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 17:10:15 2021

        Sparse resnet        recent err:

"""
import pandas as pd
import torch
from torch.utils.data import  Dataset,DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms,models
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import os
import math
import pickle as pkl
import numpy as np
import torch.utils.data as data
from torch.autograd import  Variable
import sparseconvnet as scn
from HailingDataLoader import collate_fn_hailing
import Hailing_plot
import time
#调用GPU

def block(a, b, dimension=3, residual_blocks=True, leakiness=0):  # default using residual_block
       m = scn.Sequential()
       if residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
             ).add(scn.AddTable())
       else:               #VGG style blocks
            m.add(scn.Sequential()
                 .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))
       return(m)
class Sparsenet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.spatial_size= torch.LongTensor([10,10,40])
        self.inputLayer = scn.Sequential().add(scn.InputLayer(3,self.spatial_size,3))
        self.module = scn.Sequential().add(scn.SubmanifoldConvolution(3,3,32,3,False)).add(
             block(32,32,3)).add(block(32,32,3)).add(scn.BatchNormLeakyReLU(32)).add(scn.Convolution(3,32,64,[2,2,2],[2,2,2],False))
        self.module.add(block(64,64,3)).add(#5*5*20
            scn.BatchNormLeakyReLU(64)).add(scn.Convolution(3,64,256,[2,2,4],[1,1,2],False))#4*4*9
        self.module.add(block(256,256,3)).add(
            scn.BatchNormLeakyReLU(256)).add(scn.Convolution(3,256,512,[2,2,3],[2,2,3],False))#2*2*2
        self.module.add(block(512,512,3)).add(
            scn.BatchNormLeakyReLU(512)).add(scn.Convolution(3,512,1024,[2,2,3],[2,2,3],False))#1*1*1
        self.outputLayer = scn.SparseToDense(3,1024)
        self.classifier = nn.Sequential(
             nn.Linear(1024,1024),
             nn.BatchNorm1d(1024),
             nn.LeakyReLU(),
             nn.Dropout(0.1),
             nn.Linear(1024,1024),
             nn.BatchNorm1d(1024),
             nn.LeakyReLU(),
             nn.Dropout(0.1),
             nn.Linear(1024,2))
    def forward(self,input):
        point_cloud = input
        coords = point_cloud[:,0:4].float()
        features = point_cloud[:,4:].float()
        out = self.inputLayer((coords, features))
        out = self.module(out)
        out = self.outputLayer(out)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)
        return(out)
    
class err_function(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,target,prediction):
     angel = torch.zeros(size = (len(target),1),dtype = torch.float,requires_grad= True)
     for i in range(len(target)):
        fi = target[i][0]
        theta = target[i][1]
        x = torch.FloatTensor([math.sin(theta)*math.cos(fi),math.sin(theta)*math.sin(fi),math.cos(theta)])
        fi = prediction[i][0]
        theta = prediction[i][1]
        y = torch.FloatTensor([math.sin(theta)*math.cos(fi),math.sin(theta)*math.sin(fi),math.cos(theta)])
        angel[i] = torch.acos(x.dot(y)/torch.sqrt(x.dot(x)*y.dot(y)))
     print(torch.mean(angel))
     return torch.mean(angel)
    
def err_func(target,prediction):
    for i in range(len(target)):
        fi = target[i][0]
        theta = target[i][1]
        x = torch.FloatTensor([math.sin(theta)*math.cos(fi),math.sin(theta)*math.sin(fi),math.cos(theta)])
        fi = prediction[i][0]
        theta = prediction[i][1]
        y = torch.FloatTensor([math.sin(theta)*math.cos(fi),math.sin(theta)*math.sin(fi),math.cos(theta)])
        angel = math.acos(x.dot(y)/math.sqrt(x.dot(x)*y.dot(y)))
    angel = 60*180*angel/len(target)/3.14159
    return angel

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model =Sparsenet().to(device)#实例化
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.SmoothL1Loss().to(device)
#exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                           patience=1, verbose=True, 
                                           threshold=0.00005, 
                                           threshold_mode='abs', cooldown=0, 
                                           min_lr=0, eps=1e-08)
list_epoch = []
test_loss = []
train_loss = []
pre_loss=[]
pre_count = []


def train(train_loader):
    model.train()
    running_loss = 0.0
    ep_loss = 0
    running_err = 0
    err = []
    for batch_idx, (data, target) in enumerate(train_loader):
            # 读入数据
            img = data.type(torch.FloatTensor).to(device)
            label = target.type(torch.FloatTensor).to(device)
            # 计算模型预测结果和损失
            output = model(img)
            loss = criterion(output, label)
            optimizer.zero_grad() # 计算图梯度清零
            loss.backward() # 损失反向传播
            optimizer.step()# 然后更新参数
            running_loss+=loss.item()
            running_err += err_func(label,output)
    print("Train err:",running_err/len(train_loader))
    err.append(running_err/len(train_loader))
    train_loss.append(math.log(running_loss/(len(train_loader))))
    ep_loss = running_loss/(len(train_loader))
    print('Train loss: ',ep_loss )

def test(data_loader):
    model.eval()
    running_loss = 0
    err = []
    running_err = 0
    with torch.no_grad():
     for i, (data,target) in enumerate(data_loader):
        img = data.type(torch.FloatTensor).to(device)
        label = target.type(torch.FloatTensor).to(device)
        output = model(img)
        loss = criterion(output, label)
        running_loss += loss.item()
        running_err += err_func(label,output)
     print("Test err:",running_err/len(data_loader))
     err.append(running_err/len(data_loader))
     test_loss.append(math.log(running_loss/len(data_loader)))
    if math.log(running_loss/len(data_loader)) <= min(test_loss):
        time.localtime()
        torch.save(model.state_dict(),"/home/dachuang/lbg/model_saved/Sparsemodel"+str(time.asctime()))
    print("Test loss" ,running_loss/len(data_loader))


def prediction(data_loader):
    model.load_state_dict(torch.load("/home/dachuang/lbg/model_saved/Sparsemodel"+str(time.asctime())))
    model.eval()
    with torch.no_grad():
     for i, (data,target) in enumerate(data_loader):
        img = data.type(torch.FloatTensor).unsqueeze(1).to(device)
        label = target.type(torch.FloatTensor).to(device)
        output = model(img)
        loss = criterion(output, label)
        labels = label.cpu()
        prediction = output.cpu()
        pre_loss.append(loss.item())
        if i%500 == 0:
            print("real (x',y'',z'):     ",(labels[0][0].item(),labels[0][1].item(),labels[0][2].item()))
            print("predicted (x',y'，z''):",(prediction[0][0].item(),prediction[0][1].item(),prediction[0][2].item()))
            print('loss:             ',loss.item())

class Mydataset(data.Dataset):

    def __init__(self, data):
        self.data = data
        self.image = []
        self.label = []
        for item in data:
            self.image.append(item[0])
            self.label.append(item[1])
        pass

    def __getitem__(self, index):
        input = self.image[index]
        target = self.label[index]
        return input, target

    def __len__(self):
        return len(self.label)


def main():
    batch_size_train = 64
    batch_size_test = 64
    epoch = 50
    train_path='/home/omnisky/WorkSpace/DaChuang/raw_data/HailingMuonTrack/1TeV/train_norm.pkl'
    test_path ='/home/omnisky/WorkSpace/DaChuang/raw_data/HailingMuonTrack/1TeV/validate_norm.pkl'
    train_dataset=Mydataset(pkl.load(open(train_path,'rb')))
    test_dataset = Mydataset(pkl.load(open(test_path,'rb')))
    train_loader=data.DataLoader(train_dataset, batch_size=16,collate_fn = collate_fn_hailing,shuffle = True)
    test_loader = data.DataLoader(test_dataset,batch_size =16,collate_fn = collate_fn_hailing,shuffle = True)
    pre_loader = data.DataLoader(test_dataset,batch_size =1)
    for i in range(epoch):
        print("epoch",i+1)
        if i >= 1: scheduler.step(test_loss[-1])
        train(train_loader)
        test(test_loader)
        list_epoch.append(i+1)
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(121)
    ax.plot(list_epoch, train_loss, color='blue',linewidth = 2.0)
    ax.plot(list_epoch, test_loss, color='red',linewidth = 1.0)
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.title('Train & Test Loss')
    plt.xlabel('epoch')
    plt.ylabel('log MSEloss')
    time.localtime()
    plt.savefig("/home/dachuang/lbg/figs/loss"+str(time.asctime()))
main()
