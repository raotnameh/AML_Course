# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 01:06:15 2020

@author: gowth
"""

from model import *
from data.data_loader import *
from config import *
from scipy.sparse import csr_matrix
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder

projectPath='models/mix/'
useSavedModel=False
numAugmentations=3
modelName='GPQModelImprovement1'

print("Loading the dataset")
Source_x, Source_y, Target_x = prepare_Data(data_dir, True)
Gallery_x, Query_x = prepare_Data(data_dir, False)
label_Similarity = csr_matrix(scipy.io.loadmat("data/cifar10/cifar10_Similarity.mat")['label_Similarity']).todense()
#label_Similarity=np.transpose(np.asarray(label_Similarity),(1,0))
print("Data loading finished")

print("Loading the models")
device = torch.device("cuda:0")
Net = GPQModel().to(device)
Prototypes = IntraNorm(Net.C, numCodeBooks).to(device)
Z = softAssignment(Prototypes,Net.Z,torch.tensor(numCodeBooks).to(device),torch.tensor(softAssgnAlpha).to(device))
optimizer = optim.Adam(Net.parameters(),lr=0.0002,betas=(0.5,0.999))

# Net = GPQModel()
# Prototypes = IntraNorm(Net.C, numCodeBooks)
# Z = softAssignment(Prototypes,Net.Z,numCodeBooks,softAssgnAlpha)
# optimizer = optim.Adam(Net.parameters(),lr=0.0002,betas=(0.5,0.999))

numLabelledSamples=Source_x.shape[0]
numUnlabelledSamples=Target_x.shape[0]
numIterations=int(numLabelledSamples/batchSize)
startEpoch=0
improvement1Loss=[]

if useSavedModel:
    modelFile=projectPath+"saved/"+modelName+".pth.tar"
    checkpoint=torch.load(modelFile)
    startEpoch=checkpoint['startEpoch']+1
    Net.load_state_dict(checkpoint['modelStateDict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    Z=checkpoint['Z']
    improvement1Loss=checkpoint['improvement1Loss']
    Prototypes=checkpoint['Prototypes']
    

for epoch in tqdm(range(startEpoch,totalEpochs,1)):
#for epoch in range(1):
    epochLoss=0
    print("Epoch : ",epoch," numIterations : ",numIterations)
    for iterationIndex in range(numIterations):
    #for iterationIndex in range(1):
        labelledIndices=np.random.choice(numLabelledSamples,size=batchSize,replace=False)
        unlabelledIndices=np.random.choice(numUnlabelledSamples,size=batchSize,replace=False)
        
        XLabelled=Source_x[labelledIndices]
        YLabelled=Source_y[labelledIndices]
        XUnlabelled=Target_x[unlabelledIndices]
        XLabelled=np.asarray(data_augmentation(XLabelled))
        XLabelled=torch.from_numpy(XLabelled).to(device)
        allAugmentationResults=[]
        for augIndex in range(numAugmentations):
            curSample=torch.from_numpy(np.asarray(data_augmentation(XUnlabelled))).to(device)
            featureU=flipGradient(Net.featureExtractor(curSample.reshape(batchSize,3,32,32).clone().detach()))
            featureU=IntraNorm(featureU,numCodeBooks)
            allAugmentationResults.append(featureU)
            
        YLabelled=np.eye(numClasses)[YLabelled]
        YLabelledMat=np.matmul(YLabelled,YLabelled.transpose())
        YLabelledMat/=np.sum(YLabelledMat,axis=1,keepdims=True)
        
        feature_S=Net.featureExtractor(XLabelled.reshape(batchSize,3,32,32))
        feature_S=IntraNorm(feature_S,numCodeBooks)
        
        
        descriptor_S=softAssignment(Z,feature_S,numCodeBooks,softAssgnAlpha)
        logits_S=Net.classifier(feature_S*beta,Prototypes*beta)
        hash_loss = NPQLoss(torch.from_numpy(YLabelledMat).to(device),feature_S, descriptor_S,numCodeBooks)
        
        cls_loss = CLSLoss(torch.from_numpy(YLabelled).to(device),logits_S)
        entropy_loss = SMELossModified(allAugmentationResults, Prototypes * beta, numCodeBooks)
        final_loss = hash_loss + lam_1*entropy_loss + lam_2*cls_loss 
        
        optimizer.zero_grad()
        final_loss.backward(retain_graph=True)
        optimizer.step()
        improvement1Loss.append(final_loss.item())
        epochLoss+=final_loss.item()
        
        if iterationIndex==numIterations-1:
            print("Final loss : ",final_loss," of epoch : ",epoch)
    
    epochLoss=epochLoss/numIterations
    improvement1Loss.append(epochLoss)
    
    stateToBeSaved={
      'startEpoch': epoch,
      'modelStateDict': Net.state_dict(),
      'optimizer' : optimizer.state_dict(),
      'Z':Z,
      'Prototypes':Prototypes,
      'improvement1Loss':improvement1Loss
    }
    
    if epoch%10==0:
      checkPointFile=projectPath+modelName+str(epoch)+".pth.tar"
      torch.save(stateToBeSaved,checkPointFile)
