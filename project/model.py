# -*- coding: utf-8 -*-
"""AML-Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wEub0Lz0qr04Jmsr9n913fcSZ0bqV17Z
"""


from config import *
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import os

#Setting the variables
numClasses=10
numCodeWords=16
numCodeBooks=12
lenCodeWord=12
softAssgnAlpha=20.0  # Soft assignment Scaling Factor
beta = 4
batchSize=500
totalEpochs=300
lam_1=0.1
lam_2=0.1
T=0.5

if torch.cuda.is_available():
  print("CUDA available")
  device=torch.device('cuda')
else:
  device=torch.device('cpu')

def flipGradient(x,l=1.0):
    #positivePath=torch.tensor(x*torch.tensor(l+1).type(torch.FloatTensor),requires_grad=False)
    #negativePath=torch.tensor(-x*torch.tensor(l).type(torch.FloatTensor),requires_grad=True)
    positivePath=(x*torch.tensor(l+1)).clone().detach().requires_grad_(False)
    negativePath=(-x*torch.tensor(l)).clone().detach().requires_grad_(True)
    return positivePath+negativePath

def IntraNorm(featureDescriptor,numCodeBooks):
    x=torch.split(featureDescriptor,numCodeBooks,dim=1)

    for codeBookIndex in range(numCodeBooks):
        l2NormValues=F.normalize(x[codeBookIndex],dim=1,p=2)
        #print("Shape : ",l2NormValues.shape)
        if codeBookIndex==0:
            inNorm=l2NormValues
        else:
            inNorm=torch.cat((inNorm,l2NormValues),1)
    
    return inNorm

def softAssignment(zValue, xValue, numCodeBooks,alpha):
    x=torch.split(xValue,numCodeBooks,dim=1)
    y=torch.split(zValue,numCodeBooks,dim=1)
    #print(len(x))
    #print(len(y))

    for codeBookIndex in range(numCodeBooks):
        sizeX=x[codeBookIndex].shape[0]
        sizeY=y[codeBookIndex].shape[0]
        #print("SizeX :",sizeX,"  SizeY : ",sizeY)

        firstDim,secondDim=x[codeBookIndex].shape
        #print("firstDim : ",firstDim,"  secondDim : ",secondDim)
        xx=torch.reshape(x[codeBookIndex].unsqueeze(0),(firstDim,secondDim,1))
        #print("Between xx : ",xx.shape)
        xx=xx.repeat(1,1,sizeY)
        #print("End xx : ",xx.shape)

        firstDim,secondDim=y[codeBookIndex].shape
        #print("firstDim : ",firstDim,"  secondDim : ",secondDim)
        yy=torch.reshape(y[codeBookIndex].unsqueeze(0),(firstDim,secondDim,1))
        #print("Between yy : ",yy.shape)
        yy=yy.repeat(1,1,sizeX).permute(2,1,0).to(device)
        #print("End yy : ",yy.shape)
        diff=1-torch.sum(torch.mul(xx.to(device),yy),dim=1)
        #print("Diff shape : ",diff.shape)
        softmaxFunction=nn.Softmax(dim=1)
        softmaxOutput=softmaxFunction(diff*(-alpha))
        #print("Softmax out shape : ",softmaxOutput.shape)
        multipliedValue=torch.matmul(softmaxOutput,y[codeBookIndex])
        #print(softDesTemp.shape)

        if codeBookIndex==0:
            featureDescriptor=multipliedValue
        else:
            featureDescriptor=torch.cat((featureDescriptor,multipliedValue),1)

    return IntraNorm(featureDescriptor,numCodeBooks)

"""
x=torch.randn(1,3,224,224)
z_=gpqModel.C
x_=gpqModel.Z
alpha=softAssgnAlpha
print(SoftAssignment(z_,x_,numCodeBooks,alpha).shape)
"""

def NPQLoss(labelsSimilarity,embeddingX,embeddingQ,numCodeBooks,regLambda=0.002):
    
    regAnchor=torch.mean(torch.sum(torch.square(embeddingX),dim=1))
    regPositive=torch.mean(torch.sum(torch.square(embeddingQ),dim=1))
    l2Loss=torch.mul(0.25*regLambda,regAnchor+regPositive)
    #print("l2Loss : ",l2Loss)
    
    embeddingX=F.normalize(embeddingX,dim=1,p=2)
    embeddingQ=F.normalize(embeddingQ,dim=1,p=2)
    #print(type(embeddingX)," , ",type(embeddingQ))
    #print("Shape embeddingsX : ",embeddingX.shape," , embeddingQ : ",embeddingQ.shape)
    logits=torch.matmul(embeddingX,torch.transpose(embeddingQ,0,1))
    #print("Logits shape : ",logits.shape)
    #print("Similarity : ",labelsSimilarity.shape)
    #print("Log softmax : ",F.log_softmax(logits,-1).shape)

    lossValue=torch.sum(-labelsSimilarity * F.log_softmax(logits,-1),-1)
    #meanLoss =  torch.nn.functional.cross_entropy(logits, labelsSimilarity.long())
    #print("Loss value : ",lossValue)
    meanLoss=lossValue.mean()
    #print("Mean loss : ",meanLoss)

    return meanLoss+l2Loss
 
def CLSLoss(label,logits):
    lossValue=torch.sum(-label * F.log_softmax(logits,-1),-1)
    meanLoss=lossValue.mean()
    return meanLoss

def SMELoss(features,centroids,numSegments):
    
    #print("features shape : ",features.shape,"  centroids shape : ",centroids.shape)
    x=torch.split(features,numSegments,dim=1)
    y=torch.split(centroids,numSegments,dim=1)
    #print("x : ",x[0].shape," y : ",y[0].shape)
    
    for segmentIndex in range(numSegments):
        multipliedOutput=torch.matmul(x[segmentIndex],torch.transpose(y[segmentIndex],0,1))
        firstDim,secondDim=multipliedOutput.shape
        currentLogits=torch.reshape(multipliedOutput.unsqueeze(0),(firstDim,secondDim,1))
        
        if segmentIndex==0:
            logits=currentLogits
        else:
            logits=torch.cat((logits,currentLogits),2)
    
    logits=F.softmax(torch.mean(logits,2),dim=1)
    lossValue=torch.sum(logits*(torch.log(logits+1e-5)),1)
    return torch.mean(lossValue)


def SMELossModified(features,centroids,numSegments):
    #print("features shape : ",features.shape,"  centroids shape : ",centroids.shape)
    numAugmentations=len(features)
    y=torch.split(centroids,numSegments,dim=1)
    
    for augIndex in range(numAugmentations):
        x=torch.split(features[augIndex]*beta,numSegments,dim=1)
        
        for segmentIndex in range(numSegments):
            multipliedOutput=torch.matmul(x[segmentIndex],torch.transpose(y[segmentIndex],0,1))
            firstDim,secondDim=multipliedOutput.shape
            currentLogits=torch.reshape(multipliedOutput.unsqueeze(0),(firstDim,secondDim,1))
            
            if segmentIndex==0:
                logits=currentLogits
            else:
                logits=torch.cat((logits,currentLogits),2)
    
        logits=F.softmax(torch.mean(logits,2),dim=1)
        if augIndex==0:
            avgLogits=logits
        else:
            avgLogits=avgLogits+logits
            
    #Computing average predictions obtained
    avgLogits=avgLogits/numAugmentations
    
    #Applying temperature sharpening procedure
    sharpenedPredictions=avgLogits**(1/T)    
    logits=sharpenedPredictions/sharpenedPredictions.sum(dim=1,keepdim=True)
    lossValue=torch.sum(logits*(torch.log(logits+1e-5)),1)
    return torch.mean(lossValue)
