import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# special tokens
PAD = 0
UNK = 1
SOS = 2
EOS = 3

# dataset max seq_len
X_seq_len = 60
Y_seq_len = 61
K_seq_len = 43

# training hyperparameters
n_pre_epoch = 5
n_epoch = 15
n_batch = 16
lr = 1e-4

# transformer hyperparameters
n_vocab = 19682
dim_model = 512
n_head = 8
n_encode_layer = 4
n_decode_layer = 4
dropout = 0.1