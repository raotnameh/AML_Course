from config import *
from data.data_loader import *
from model import *
from utils import *
    
#Dataloader
Gallery_x, Query_x = prepare_Data(data_dir, False)
gallery = torch.utils.data.DataLoader(Gallery_x,batch_size=2*batchSize)
query = torch.utils.data.DataLoader(Query_x,batch_size=2*batchSize)
similarity = csr_matrix(scipy.io.loadmat("data/cifar10/cifar10_Similarity.mat")['label_Similarity']).todense()

#models
model = features_(net1, net2).to(device)
classifier = classifier_(n_CLASSES, len_code, n_book).to(device)
softassignment = softassignment_(len_code, n_book, intn_word).to(device)
flipGradient = flipGradient_()

if weights_path:
    print("Loading weights")
    weights = torch.load(weights_path)
    
    model.load_state_dict(weights['modelStateDict'][0])
    classifier.load_state_dict(weights['modelStateDict'][1])
    softassignment.load_state_dict(weights['modelStateDict'][2])


mean_average_precision = test_(similarity, model, classifier, softassignment, gallery, query,Indexing_, pqDist, cat_apcal)
print(f"mAP score is: {mean_average_precision}")