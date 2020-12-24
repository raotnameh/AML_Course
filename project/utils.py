import torch.nn.functional as F
from config import *
import torch
import torch.nn as nn

def test_(similarity,model, classifier, softassignment, gallery, query,Indexing_, pqDist, cat_apcal):
    with torch.no_grad():
        model.eval()
        classifier.eval()
        softassignment.eval()

        for r, i in tqdm(enumerate(gallery)):
            dfg = intranorm(model(i.to(device).permute(0,3,1,2)),n_book)
            if r == 0: 
                dummy = dfg
            else: 
                dummy = torch.cat((dummy, dfg), 0)

        for r, i in tqdm(enumerate(query)):
            dfg = intranorm(model(i.to(device).permute(0,3,1,2)),n_book)
            if r == 0: 
                query_x = dfg
            else: 
                query_x = torch.cat((query_x, dfg), 0)

        dummy = Indexing_(softassignment.Z.state_dict()['weight'].cpu(), dummy.cpu(), n_book)
        gallery_x = dummy.numpy().astype(int)
        quantizedDist = pqDist(intranorm(softassignment.Z.state_dict()['weight'].cpu(), n_book), n_book,gallery_x, query_x.cpu().numpy()).T
        Rank = np.argsort(quantizedDist, axis=0)
        mean_average_precision=cat_apcal(similarity,Rank,54000)

        return mean_average_precision

def NPQLoss(labelsSimilarity,embeddingX,embeddingQ,numCodeBooks,regLambda=0.002):
    
    regAnchor=torch.mean(torch.sum(torch.square(embeddingX),dim=1))
    regPositive=torch.mean(torch.sum(torch.square(embeddingQ),dim=1))
    l2Loss=torch.mul(0.25*regLambda,regAnchor+regPositive)
    
    embeddingX=F.normalize(embeddingX,dim=1,p=2)
    embeddingQ=F.normalize(embeddingQ,dim=1,p=2)
    logits=torch.matmul(embeddingX,torch.transpose(embeddingQ,0,1))
   
    lossValue=torch.sum(-labelsSimilarity * F.log_softmax(logits,-1),-1)
    meanLoss=lossValue.mean()
    return meanLoss+l2Loss
 
def SMELoss(features,centroids,numSegments):
    
    x=torch.split(features,numSegments,dim=1)
    y=torch.split(centroids,numSegments,dim=1)
    
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


def intranorm(features,n_book):
    x = features.split(n_book,1)
    
    for b in range(n_book):
        if b==0: dummy = F.normalize(x[b],1)
        else:
            dummy = torch.cat((dummy,F.normalize(x[b],1)),1)
    return dummy


# Find the closest codeword index          
def Indexing_(Z,des,numSeg):
        Z = intranorm(Z,numSeg)
        x = torch.split(des,numSeg,1)
        y = torch.split(Z,numSeg,1)
        for i in range(numSeg):
            size_x = x[i].shape[0]
            size_y = y[i].shape[0]
            xx = x[i].unsqueeze(-1)

            dummy = torch.tensor(1)
            xx = xx.tile([1,1,size_y])
            yy = y[i].unsqueeze(-1)
            yy = yy.tile([1,1,size_x])
            yy = yy.permute(2,1,0)
            diff = torch.sum(torch.multiply(xx,yy),1)

            arg = torch.argmax(diff,1)
            max_idx = arg.reshape(-1,1)

            if i == 0: quant_idx = max_idx
            else: quant_idx = torch.cat((quant_idx,max_idx),1)

        return quant_idx

# Average Precision (AP) Calculation
def cat_apcal(label_Similarity, IX, top_N):

    [_, numtest] = IX.shape

    apall = np.zeros(numtest)

    for i in range(numtest):
        y = IX[:, i]
        x = 0
        p = 0

        for j in range(top_N):
            if label_Similarity[i, y[j]] == 1:
                x = x + 1
                p = p + float(x) / (j + 1)
        if p == 0:
            apall[i] = 0
        else:
            apall[i] = p / x

    mAP = np.mean(apall)

    return mAP

# Compute distances and build look-up-table
def pqDist(Z, numSeg, g_x, q_x):
    n1 = q_x.shape[0]
    n2 = g_x.shape[0]
    l1, l2 = Z.shape
    D_Z = np.zeros((l1, numSeg), dtype=np.float32)
    q_x_split = np.split(q_x, numSeg, 1)

    g_x_split = np.split(g_x, numSeg, 1)
    Z_split = np.split(Z, numSeg, 1)
    D_Z_split = np.split(D_Z, numSeg, 1)



    Dpq = np.zeros((n1, n2), dtype=np.float32)
    for i in range(n1):
        for j in range(numSeg):
            for k in range(l1):
                D_Z_split[j][k] =1-np.dot(q_x_split[j][i],Z_split[j][k])
            if j == 0:
                y = D_Z_split[j][g_x_split[j]]
            else:
                y = np.add(y, D_Z_split[j][g_x_split[j]])
        Dpq[i, :] = np.squeeze(y)
    return Dpq
