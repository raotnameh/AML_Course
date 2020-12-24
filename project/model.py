from config import *
import torch
import torch.nn as nn
from utils import *
from torchvision import models


# Loading VGG16 with batch norm as mentioned in the paper
vggModel=models.vgg16_bn(pretrained=True)
net1, net2 = [], []
for i in vggModel.children():
    for r, i in enumerate(i.children()):
        if r <=23: net1.append(i)
        elif r<= 32: net2.append(i)
    break
net1, net2  = nn.Sequential(*net1), nn.Sequential(*net2)

# Feature block using vgg16 layer
class features_(nn.Module):
    def __init__(self, net1, net2):
        super(features_,self).__init__()
        
        self.net1 = net1
        self.net2 = net2
        self.gavgp = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(768, len_code*n_book)
        
    def forward(self,x):
        x = self.net1(x) # shape: torch.Size([32, 3, 32,32])>torch.Size([32, 3, 4, 4])
        x_branch = self.gavgp(x)
        x = self.net2(x) # shape: torch.Size([32, 3, 4, 4])> torch.Size([32, 3, 4, 4])
        x = self.gavgp(x)
        
        x = torch.cat((x,x_branch),1)
        
        return self.linear(x.view(-1,768))
    
# Soft assignment to convert features F into quantaized form q
class softassignment_(nn.Module):
    def __init__(self,len_code, n_book, intn_word):
        super(softassignment_,self).__init__()
        self.Z = nn.Linear(len_code * n_book,intn_word, bias=False)

    def forward(self,features,n_book,alpha,):
        z_ = intranorm(self.Z.state_dict()['weight'], n_book).split(n_book,1)
        x_ = features.split(n_book,1)

        for i in range(n_book):
            size_z = z_[i].shape[0] # number of codewords
            size_x = x_[i].shape[0] # batch size
            xx = x_[i].unsqueeze(-1)
            xx = xx.repeat(1,1,size_z)
            zz = z_[i].unsqueeze(-1)
            zz = zz.repeat(1,1,size_x).T

            diff = 1 - torch.sum(torch.mul(xx,zz), 1) # 32,16
            softmax_diff = F.softmax(diff*(-alpha),1) #32,16
            soft_des_temp = torch.matmul(softmax_diff,z_[i]) # 32,12
            if i == 0: descriptor = soft_des_temp
            else: descriptor = torch.cat((descriptor,soft_des_temp),1)

        return intranorm(descriptor,n_book) # 32,144

# Simple classifier
class classifier_(nn.Module):
    def __init__(self,n_CLASSES, len_code, n_book,):
        super(classifier_,self).__init__()
        self.prototypes = nn.Linear(len_code * n_book, n_CLASSES, bias=False)
        self.n_book = n_book
        
    def forward(self, x):
        x_ = x.split(self.n_book,1)
        c_ = (intranorm(self.prototypes.state_dict()['weight'], n_book)*beta).T.split(self.n_book,0)
        for i in range(self.n_book):
            sub_res = torch.matmul(x_[i], c_[i]).unsqueeze(-1)
            if i == 0: res = sub_res
            else: res = torch.cat((res,sub_res),2)
        
        return torch.sum(res, 2)


class flipGradient_(nn.Module):
    def forward(self,x,l=1.0):
        positivePath=(x*2).clone().detach().requires_grad_(False)
        negativePath=(-x).requires_grad_(True)
        return positivePath+negativePath
