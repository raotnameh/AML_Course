from model import *
from data.data_loader import *



print("Loading the dataset")
#Source_x, Source_y, Target_x = prepare_Data(data_dir, True)
#Gallery_x, Query_x = ci10.prepare_data(data_dir, False)
print("Data loading finished")

Net = GPQModel()
Prototypes = IntraNorm(Net.C, numCodeBooks)
Z = softAssignment(Prototypes, Net.Z, numCodeBooks, softAssgnAlpha)
feature_S = IntraNorm(feature_S, numCodeBooks)
feature_T = IntraNorm(feature_T, numCodeBooks)

"""
descriptor_S = softAssignment(Z, feature_S, numCodeBooks, softAssgnAlpha)

logits_S = Net.Classifier(feature_S * beta, tf.transpose(Prototypes) * beta)

hash_loss = N_PQ_loss(labels_Similarity=label_Mat, embeddings_x=feature_S, embeddings_q=descriptor_S, n_book)
cls_loss = CLS_loss(label, logits_S)
entropy_loss = SME_loss(feature_T * beta, tf.transpose(Prototypes) * beta, numCodeBooks)
"""
