import os
import tensorflow as tf
import numpy as np
import pickle as pk
import scipy.io
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Dataset path
# Source: 5,000, Target: 54,000
# Target: 54,000 Query: 1,000
data_dir = 'data/cifar10'


# For training
ImagNet_pretrained_path = 'models/ImageNet_pretrained'
model_save_path = 'models/'

# For evaluation
model_load_path = 'models/48bits_example.ckpt'
cifar10_label_sim_path = 'cifar10/cifar10_Similarity.mat'


n_CLASSES = 10
image_size = 32
img_channels = 3
n_DB = 54000

'Hyperparameters for training'
# Training epochs, 1 epoch represents training all the source data once
total_epochs = 300
batch_size = 500
# save model for every save_term-th epoch
save_term = 20

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
lam_1 = 0.1
lam_2 = 0.1


