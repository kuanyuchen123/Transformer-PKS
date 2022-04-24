import torch
import params
from tokenizers import Tokenizer
from model import create_pad_mask, get_tgt_mask, Transformer_Encoder, Transformer_Decoder
from torch.utils.data import DataLoader
from data import load_data, PersonaDataset
import torch.nn as nn

def test(encoder, decoder, data_loader):
    total_loss = 0
    NLLLoss = nn.NLLLoss(ignore_index=params.PAD)
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        for step, (src_X, src_Y, tgt_Y, src_K) in enumerate(data_loader):
            src_X = src_X.to(device)
            src_Y = src_Y.to(device)
            tgt_Y = tgt_Y.to(device)
            src_K = src_K.to(device)

            sequence_length = src_Y.size(1)
            src_pad_mask = create_pad_mask(src_X, params.PAD).to(device)
            tgt_pad_mask = create_pad_mask(src_Y, params.PAD).to(device)
            tgt_mask = get_tgt_mask(sequence_length).to(device)
            encoder_hidden = encoder(src_X, src_pad_mask)
            hidden, outputs = decoder(encoder_hidden, src_Y, tgt_mask, tgt_pad_mask, src_pad_mask)
            nll_loss = NLLLoss(outputs.permute(1,2,0), tgt_Y)
            total_loss += nll_loss.detach().item()
        
    return total_loss / len(data_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Transformer_Encoder(
    num_tokens=params.n_vocab, dim_model=params.model_d, num_heads=params.n_head, \
    num_encoder_layers=params.n_encode_layer, dropout_p=params.dropout
).to(device)

decoder = Transformer_Decoder(
    num_tokens=params.n_vocab, dim_model=params.model_d, num_heads=params.n_head, \
    num_decoder_layers=params.n_decode_layer, dropout_p=params.dropout
).to(device)


tokenizer = Tokenizer.from_file("./vocab.json")
print("loading data...")
X, Y, K = load_data("./data/test.txt")
dataset = PersonaDataset(X, Y, K)
data_loader = DataLoader(dataset=dataset, batch_size=params.n_batch, shuffle=False)

for i in range(30):
    encoder.load_state_dict(torch.load("./checkpoint/encoder_{}.pt".format(i)))
    decoder.load_state_dict(torch.load("./checkpoint/decoder_{}.pt".format(i)))
    loss = test(encoder, decoder, data_loader)
    print("epoch{}: {}".format(i, loss/50))