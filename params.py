import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# special tokens
PAD = 0
UNK = 1
SOS = 2
EOS = 3
SEP = 4

# dataset max seq_len
X_seq_len = 60
Y_seq_len = 61
K_seq_len = 44

# training hyperparameters
n_pre_epoch = 5
n_epoch = 15
n_batch = 16
lr = 1e-5

# transformer hyperparameters
n_vocab = 19682
dim_model = 512
n_head = 8
n_encode_layer = 2
n_decode_layer = 2
dropout = 0.1