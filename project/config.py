import os
import numpy as np
from tqdm.auto import tqdm
import pickle as pk
import scipy.io
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import torch

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda")
# Dataset path
# Source: 5,000, Target: 54,000
# Target: 54,000 Query: 1,000
data_dir = 'data/cifar10'


# For training
model_save_path = 'models/'
do_val = False

# For evaluation
weights_path = None #'models/GPQ.pth'
cifar10_label_sim_path = 'cifar10/cifar10_Similarity.mat'


n_CLASSES = 10
image_size = 32
img_channels = 3
n_DB = 54000

'Hyperparameters for training'
# Training epochs, 1 epoch represents training all the source data once
total_epochs = 10000
batchSize = 1

# test and save model for every test_term-th epoch.
test_term = 20

# length of codeword
len_code = 12

# Number of codebooks
n_book = 12

# Number of codewords=(2^bn_word)
bn_word = 4
intn_word = pow(2, bn_word)

# Number of bits for retrieval
n_bits = n_book * bn_word

# Soft assignment input scaling factor
alpha = 20.0

# Classification input scaling factor
beta = 4

# lam1, 2: loss function balancing parameters
lam_1 = 0.5
lam_2 = 0.1
