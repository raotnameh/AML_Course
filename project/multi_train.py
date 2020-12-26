from config import *
import torch
import horovod.torch as hvd
from data.data_loader import *
from model import *
from utils import *
import torch.optim as optim

hvd.init()
torch.cuda.set_device(hvd.local_rank())
if __name__ == '__main__':
    #Dataloader
    Source_x, Source_y, Target_x = prepare_Data(data_dir, True)
    train_sampler = torch.utils.data.distributed.DistributedSampler([(Source_x[i], Source_y[i]) for i in range(len(Source_x))], num_replicas=hvd.size(), rank=hvd.rank())
    source = torch.utils.data.DataLoader([(Source_x[i], Source_y[i]) for i in range(len(Source_x))],batch_size=batchSize, sampler=train_sampler)
    target_sampler = torch.utils.data.distributed.DistributedSampler(Target_x, num_replicas=hvd.size(), rank=hvd.rank())
    target = torch.utils.data.DataLoader(Target_x,batch_size=batchSize,sampler=train_sampler)
    
    # Gallery_x, Query_x = prepare_Data(data_dir, False)
    # gallery = torch.utils.data.DataLoader(Gallery_x,batch_size=2*batchSize)
    # query = torch.utils.data.DataLoader(Query_x,batch_size=2*batchSize)
    # similarity = csr_matrix(scipy.io.loadmat("data/cifar10/cifar10_Similarity.mat")['label_Similarity']).todense()

    #models
    model = features_(net1, net2).to(device)
    classifier = classifier_(n_CLASSES, len_code, n_book).to(device)
    softassignment = softassignment_(len_code, n_book, intn_word).to(device)
    flipGradient = flipGradient_()

    # optimizer
    class_optim = optim.Adam(classifier.parameters(),lr=0.002,weight_decay=0.00001,amsgrad=True)
    model_optim = optim.Adam(model.parameters(),lr=0.0002,weight_decay=0.00001,amsgrad=True)
    soft_optim = optim.Adam(softassignment.parameters(),lr=0.002,weight_decay=0.00001,amsgrad=True)

    class_optim = hvd.DistributedOptimizer(class_optim, named_parameters=classifier.named_parameters())
    model_optim = hvd.DistributedOptimizer(model_optim, model.named_parameters())
    soft_optim = hvd.DistributedOptimizer(soft_optim ,softassignment.named_parameters())

    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    if weights_path:
        print("Loading weights")
        weights = torch.load(weights_path)
        
        model.load_state_dict(weights['modelStateDict'][0])
        classifier.load_state_dict(weights['modelStateDict'][1])
        softassignment.load_state_dict(weights['modelStateDict'][2])

        model_optim.load_state_dict(weights['modelOptimDict'][0])
        class_optim.load_state_dict(weights['modelOptimDict'][1])
        soft_optim.load_state_dict(weights['modelOptimDict'][2])

    class_optim.zero_grad()
    model_optim.zero_grad()
    soft_optim.zero_grad()


    target_ = iter(target)
    score = 0
    save = ""
    for epoch in range(total_epochs):
        m_,n,o,p = 0,0,0,0
        model.train()
        classifier.train()
        softassignment.train()

        for df, batch in enumerate(source):
            x, y = batch
            x = torch.tensor(data_augmentation(x)).to(device)
            try: 
                xu = next(target_)
                xu = torch.tensor(data_augmentation(xu)).to(device)
            except:
                target_ = iter(target)
                xu = next(target_)
                xu = torch.tensor(data_augmentation(xu)).to(device)

            features = intranorm(model(x.permute(0,3,1,2)), n_book)
            featuresu = flipGradient(intranorm(model(xu.permute(0,3,1,2)), n_book))
            quanta = softassignment(features,n_book,alpha)
            logits = classifier(features)

            y = y.to(device)
            cls_loss = torch.nn.functional.cross_entropy(logits,y)

            y = torch.eye(n_CLASSES)[y].to(device)
            entropy_loss = SMELoss(featuresu ,intranorm(classifier.prototypes.state_dict()['weight'], n_book) , n_book)

            y_ = torch.matmul(y,y.T)
            y_ /= torch.sum(y_, axis=1, keepdims=True)
            hash_loss = NPQLoss(y_,features, quanta,n_book)   

            final_loss = hash_loss + lam_1*cls_loss  + lam_2*entropy_loss 

            o += cls_loss.item()
            m_ += final_loss.item()
            n += hash_loss.item()
            p += entropy_loss.item()

            final_loss.backward()

            model_optim.step()
            soft_optim.step()
            class_optim.step()

            class_optim.zero_grad()
            model_optim.zero_grad()
            soft_optim.zero_grad()

        save += f"Total_loss: {m_/10}\t Hash_loss: {n/10}\t Classsifier_loss: {o/10}\t Entropy_loss: {p/10}"

        if epoch % test_term == 0 and do_val == True: 
            mean_average_precision = test_(similarity, model, classifier, softassignment, gallery, query,Indexing_, pqDist, cat_apcal)
            save += f"\tmAP_score: {mean_average_precision}"
            print(f"Total_loss: {m_/10}\t Hash_loss: {n/10}\t Classsifier_loss: {o/10}\t Entropy_loss: {p/10}\tmAP_score: {mean_average_precision}")
            if mean_average_precision > score:
                print("Found better validated model")
                score = mean_average_precision
                stateToBeSaved={
                    'modelStateDict': [model.state_dict(),classifier.state_dict(), softassignment.state_dict()],
                    'modelOptimDict': [model_optim.state_dict(),class_optim.state_dict(), soft_optim.state_dict()],
                    'score': mean_average_precision,
                    'epoch': epoch+1}
                torch.save(stateToBeSaved,f"{model_save_path}/GPQ.pth")
        else:
            print(f"Total_loss: {m_/10}\t Hash_loss: {n/10}\t Classsifier_loss: {o/10}\t Entropy_loss: {p/10}")
        save += '\n'
        with open("loss.txt", "w") as f:
            f.write(save)
