from unittest import result
from data import load_data, PersonaDataset
from model import Knowledge_Encoder, Knowledge_Manager, create_pad_mask, get_seq_mask, Transformer_Encoder, Transformer_Decoder
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import FloatTensor, long, optim
import params
import os
import tokenizer
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
knowledge_encoder = Knowledge_Encoder(
    num_tokens=params.n_vocab, dim_model=params.dim_model, num_heads=params.n_head, \
    num_encoder_layers=params.n_encode_layer, dropout_p=params.dropout
).to(device)

encoder = Transformer_Encoder(
    num_tokens=params.n_vocab, dim_model=params.dim_model, num_heads=params.n_head, \
    num_encoder_layers=params.n_encode_layer, dropout_p=params.dropout
).to(device)

knowledge_manager = Knowledge_Manager(
    X_seq_len=60,
    Y_seq_len=params.Y_seq_len,
    K_seq_len=43,
    dim_model=params.dim_model
).to(device)

ut = torch.zeros( params.n_batch, 43, dtype=long ).to(device)
ut_pad = create_pad_mask(ut, params.PAD)

K = torch.ones( params.n_batch,3,43, dtype=long ).to(device)
encoder_hidden = torch.ones( 60, params.n_batch, params.dim_model ).to(device)
K_pad = create_pad_mask( K, params.PAD ).to(device)
K_hidden = knowledge_encoder( K, K_pad )
manager_output = knowledge_manager( encoder_hidden, K_hidden )


# pool of square window of size=3, stride=2
m = nn.AvgPool2d((5,2), stride=1)
# pool of non-square window
# encoded_K: (N, sequence length, n_batch, dim_model)
input1 = torch.Tensor([[[[1,2,3,4,5]]]])
input2 = torch.Tensor([[[[4,5,6,7,8]]]])
input3 = torch.Tensor([[[[9,10,11,12,13]]]])
input4 = torch.Tensor([[[[9,10,11,12,13]]]])
input5 = torch.Tensor([[[[9,10,11,12,13]]]])

input = torch.cat((input1,input2,input3,input4,input5), dim=2)
print(input)
print(input.shape)
output = m(input)
print(output.shape)

test = torch.rand(16,3)
output = torch.argmax(test, dim=1)
out = torch.zeros_like(test).scatter_(1, output.unsqueeze(1), 1.)
print(out.shape)

device = 'cuda'
knowledge_encoder = Knowledge_Encoder(
    num_tokens=params.n_vocab, dim_model=params.dim_model, num_heads=params.n_head, \
    num_encoder_layers=params.n_encode_layer, dropout_p=params.dropout
).to(device)

knowledge_encoder.load_state_dict(torch.load("./checkpoint/k_encoder_10.pt"))

knowledge_encoder.eval()

k = torch.ones(1,3,43, dtype=long).to(device)
k = torch.add(k,1200)
k_mask = create_pad_mask(k,params.PAD).to(device)
k_embed = knowledge_encoder( k,k_mask )
print(k_embed)

device = 'cuda'

encoder = Transformer_Encoder(
    num_tokens=params.n_vocab, dim_model=params.dim_model, num_heads=params.n_head, \
    num_encoder_layers=params.n_encode_layer, dropout_p=params.dropout
).to(device)

encoder.load_state_dict(torch.load("./checkpoint/encoder_10.pt"))
encoder.eval()

x = torch.zeros(1,5, dtype=long).to(device)
one = torch.ones(1,2, dtype=long).to(device)
one[0][1]=2
x = torch.cat((one,x),dim=1)
#x = torch.add(x,67)
x_mask = create_pad_mask(x,params.PAD)
x_embed = encoder( x,x_mask )
print(x_embed)


x_embed = x_embed.permute(1,0,2)
mean_X = nn.AvgPool2d((2,1), stride=1)
mean_x = mean_X(x_embed).squeeze(1)  


device = 'cuda'

knowledge_manager = Knowledge_Manager(
    X_seq_len=params.X_seq_len,
    Y_seq_len=params.Y_seq_len,
    K_seq_len=params.K_seq_len,
    dim_model=params.dim_model
).to(device)

knowledge_manager.load_state_dict(torch.load("./checkpoint/k_manager_10.pt"))
#knowledge_manager.eval()
knowledge_manager.train()

k_hidden = torch.rand( 3, params.K_seq_len, 1, params.dim_model )
for i in range(20):
    x_hidden = torch.rand( params.X_seq_len, 1, params.dim_model )
    K_index, selected_K = knowledge_manager(x_hidden,None,k_hidden)
    print(K_index)

mean_x = nn.AvgPool2d((4,1), stride=1)

# (batch, seq, dim)
x_hidden = torch.ones(1,4,5).type(FloatTensor)

# (batch, seq)
x_pad_mask = torch.tensor([[0,0,1,1]]).type(FloatTensor)

b_mask = torch.rand(1,x_pad_mask.shape[1],x_pad_mask.shape[1])
for b in range(1) :
    mask = torch.eye(x_pad_mask.shape[1])
    mask = mask * x_pad_mask[b]
    b_mask[b] = mask

print(b_mask)
result = torch.bmm(x_hidden.transpose(1,2),b_mask).transpose(1,2)
print(result)



# mask = torch.tril(torch.ones(x_pad_mask.shape[1], x_pad_mask.shape[1]) == 1) # Lower triangular matrix
# print(mask)  

# result = torch.bmm(x_hidden.transpose(1,2),b_mask).transpose(1,2)
# print(result)
"""

"""
x_hidden = torch.ones(2,4,4).type(FloatTensor)
x_pad_mask = torch.tensor([[False,False,False,True],[True,False,False,False]])


n_batch = x_hidden.shape[0]
seq_len = x_hidden.shape[1]

mask = torch.eye(seq_len).unsqueeze(0).repeat(n_batch,1,1)
print(x_pad_mask.shape)
result_mask = torch.einsum('ijk,ij -> ijk', mask, x_pad_mask)
print(result_mask)
embed = torch.bmm(x_hidden.transpose(1,2),result_mask).transpose(1,2)
print(embed)
"""

"""
from model import make_pad_embed_zero
x_hidden = torch.rand(3,4,4).type(FloatTensor)
print(x_hidden)
x_pad_mask = torch.tensor([[False,False,False,True],[True,False,False,False],[True,True,True,True]])

result = make_pad_embed_zero(x_hidden,x_pad_mask)
print(result)
"""

from model import make_pad_embed_zero
x_hidden = torch.rand(2,3,4,4).type(FloatTensor)
print(x_hidden)
x_pad_mask = torch.tensor([[[False,False,False,False],[False,False,False,True],[True,True,True,True]],
                           [[True,False,False,False],[True,False,False,True],[True,True,True,False]]
                          ])

result = make_pad_embed_zero(x_hidden,x_pad_mask)
print(result)