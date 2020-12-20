from model import *
from data.data_loader import *
from config import *
from scipy.sparse import csr_matrix
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder

projectPath='models/dummy/'
useSavedModel=False

print("Loading the dataset")
Source_x, Source_y, Target_x = prepare_Data(data_dir, True)
# Gallery_x, Query_x = prepare_Data(data_dir, False)
label_Similarity = csr_matrix(scipy.io.loadmat("data/cifar10/cifar10_Similarity.mat")['label_Similarity']).todense()
print("Data loading finished")

source = torch.utils.data.DataLoader([(Source_x[i], Source_y[i]) for i in range(len(Source_x))],batch_size=batchSize)
target = torch.utils.data.DataLoader(Target_x,batch_size=batchSize)
a = torch.utils.data.DataLoader(Source_x,batch_size=batchSize)
b = torch.utils.data.DataLoader(Source_y,batch_size=batchSize)
c = torch.utils.data.DataLoader(Target_x,batch_size=batchSize)

print("Loading the models")
Net = GPQModel()
Prototypes = IntraNorm(Net.C, numCodeBooks)
Z = softAssignment(Prototypes,Net.Z,numCodeBooks,softAssgnAlpha)
optimizer = optim.Adam(Net.parameters(),lr=0.0002,betas=(0.5,0.999))

numLabelledSamples=Source_x.shape[0]
numUnlabelledSamples=Target_x.shape[0]
numIterations=int(numLabelledSamples/batchSize)
startEpoch=0
allEpochLoss=[]

if useSavedModel:
    modelFile=projectPath+"saved/GPQModel.pth.tar"
    checkpoint=torch.load(modelFile)
    startEpoch=checkpoint['startEpoch']+1
    Net.load_state_dict(checkpoint['modelStateDict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    Z=checkpoint['Z']
    allEpochLoss=checkpoint['allEpochLoss']
    Prototypes=checkpoint['Prototypes']

for epoch in range(startEpoch,totalEpochs,1):
    print("Epoch : ",epoch," numIterations : ",numIterations)
    epochLoss=0
    for iterationIndex in range(numIterations):
    #for iterationIndex in range(1):
        labelledIndices=np.random.choice(numLabelledSamples,size=batchSize,replace=False)
        unlabelledIndices=np.random.choice(numUnlabelledSamples,size=batchSize,replace=False)
        
        XLabelled=Source_x[labelledIndices]
        YLabelled=Source_y[labelledIndices]
        XUnlabelled=Target_x[unlabelledIndices]
        XLabelled=np.asarray(data_augmentation(XLabelled))
        XLabelled=torch.from_numpy(XLabelled)
        XUnlabelled=np.asarray(data_augmentation(XUnlabelled))
        XUnlabelled=torch.from_numpy(XUnlabelled)
        
        YLabelled=np.eye(numClasses)[YLabelled]
        YLabelledMat=np.matmul(YLabelled,YLabelled.transpose())
        YLabelledMat/=np.sum(YLabelledMat,axis=1,keepdims=True)
        
        feature_S=Net.featureExtractor(XLabelled.reshape(batchSize,3,32,32))
        feature_T=flipGradient(Net.featureExtractor(XUnlabelled.reshape(batchSize,3,32,32).clone().detach()))
        
        feature_S=IntraNorm(feature_S,numCodeBooks)
        feature_T=IntraNorm(feature_T,numCodeBooks)
        
        descriptor_S=softAssignment(Z,feature_S,numCodeBooks,softAssgnAlpha)
        logits_S=Net.classifier(feature_S*beta,Prototypes*beta)
        hash_loss = NPQLoss(torch.from_numpy(YLabelledMat),feature_S, descriptor_S,numCodeBooks)
        
        cls_loss = CLSLoss(torch.from_numpy(YLabelled),logits_S)
        entropy_loss = SMELoss(feature_T * beta, Prototypes * beta, numCodeBooks)
        final_loss = hash_loss + lam_1*entropy_loss + lam_2*cls_loss 
        
        optimizer.zero_grad()
        final_loss.backward(retain_graph=True)
        optimizer.step()
        
        epochLoss+=final_loss.item()
        if iterationIndex==numIterations-1:
            print("Final loss : ",final_loss," of epoch : ",epoch)
    
    epochLoss=epochLoss/numIterations
    allEpochLoss.append(epochLoss)
    
    stateToBeSaved={
      'startEpoch': epoch,
      'modelStateDict': Net.state_dict(),
      'optimizer' : optimizer.state_dict(),
      'Z':Z,
      'Prototypes':Prototypes,
      'allEpochLoss':allEpochLoss
    }
    
    if epoch%10==0:
        checkPointFile=projectPath+"saved/GPQModel"+str(epoch)+".pth.tar"
        torch.save(stateToBeSaved,checkPointFile)
