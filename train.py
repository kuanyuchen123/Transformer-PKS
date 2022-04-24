from data import load_data, PersonaDataset
from model import create_pad_mask, get_seq_mask, Transformer_Encoder, Transformer_Decoder, Knowledge_Encoder, Knowledge_Manager
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import params
import tokenizer
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(model, optimizer, data_loader):
    encoder, decoder, knowledge_encoder, knowledge_manager = [*model]
    encoder.train()
    decoder.train()
    knowledge_encoder.train()
    knowledge_manager.train()
    NLLLoss = nn.NLLLoss(reduction="mean",ignore_index=params.PAD)
    KLDLoss = nn.KLDivLoss(reduction="batchmean")
    avg_loss = {'nll_loss': 0, 'kld_loss': 0}

    for epoch in range(params.n_epoch):
        for step, (src_X, src_Y, tgt_Y, src_K) in enumerate(data_loader):
            src_X = src_X.to(params.device)
            src_Y = src_Y.to(params.device)
            tgt_Y = tgt_Y.to(params.device)
            src_K = src_K.to(params.device)

            sequence_length = src_Y.size(1)
            src_X_pad_mask = create_pad_mask(src_X, params.PAD).to(params.device)
            src_Y_pad_mask = create_pad_mask(src_Y, params.PAD).to(params.device)
            src_K_pad_mask = create_pad_mask(src_K, params.PAD).to(params.device)
            src_Y_seq_mask = get_seq_mask(sequence_length).to(params.device)

            optimizer.zero_grad()
            X_hidden = encoder(src_X, src_X_pad_mask)
            Y_hidden = knowledge_encoder(src_Y, src_Y_pad_mask)
            K_hidden = knowledge_encoder(src_K, src_K_pad_mask)

            prior, posterior, K_index, selected_K, vocab_selected_K = knowledge_manager(X_hidden,Y_hidden,K_hidden,src_X_pad_mask,src_Y_pad_mask,src_K_pad_mask)
            selected_K_pad_mask = torch.einsum('abc, ab -> ac', src_K_pad_mask.type(torch.FloatTensor), K_index.type(torch.FloatTensor))
            selected_K_pad_mask = selected_K_pad_mask.type(torch.BoolTensor).to(params.device)
            vocab_selected_K = (
                vocab_selected_K.repeat(sequence_length-1, 1, 1)
                .transpose(0, 1)
                .contiguous()
                .view(-1, params.n_vocab)
            )

            hidden, outputs = decoder(X_hidden, src_Y, selected_K, src_X_pad_mask, src_Y_pad_mask, selected_K_pad_mask, src_Y_seq_mask)

            nll_loss = NLLLoss(outputs.permute(1,2,0), tgt_Y)
            # bow_loss = NLLLoss(vocab_selected_K, src_Y[:, 1:].contiguous().view(-1))
            kld_loss = KLDLoss(prior, posterior.detach())
            loss = nll_loss + kld_loss
            loss.backward()
            optimizer.step()

            avg_loss['nll_loss'] += nll_loss
            avg_loss['kld_loss'] += kld_loss
            if (step + 1) % 50 == 0:
                avg_loss['nll_loss'] /= 50
                avg_loss['kld_loss'] /= 50
                print(
                    "Epoch [%.2d/%.2d] Step [%.4d/%.4d]: nll_loss=%.4f, kld_loss=%.5f"
                    % (
                        epoch + 1,
                        params.n_epoch,
                        step + 1,
                        len(data_loader),
                        avg_loss['nll_loss'],
                        avg_loss['kld_loss'],
                    )
                )

                avg_loss['nll_loss'] = 0
                avg_loss['kld_loss'] = 0

        torch.save(encoder.state_dict(), "./checkpoint/encoder_{}.pt".format(epoch))
        torch.save(decoder.state_dict(), "./checkpoint/decoder_{}.pt".format(epoch))
        torch.save(knowledge_encoder.state_dict(), "./checkpoint/k_encoder_{}.pt".format(epoch))
        torch.save(knowledge_manager.state_dict(), "./checkpoint/k_manager_{}.pt".format(epoch))

if __name__ == "__main__":
    print( "Set device to {}".format( params.device ) )

    if os.path.exists("vocab.json"):
        print("vocab dictionary built...")
    else :
        print("building vocab dictionary...")
        tokenizer.tokenize()

    print("loading model...")
    encoder = Transformer_Encoder(
        num_tokens=params.n_vocab, dim_model=params.dim_model, num_heads=params.n_head, \
        num_encoder_layers=params.n_encode_layer, dropout_p=params.dropout
    ).to(params.device)

    decoder = Transformer_Decoder(
        num_tokens=params.n_vocab, dim_model=params.dim_model, num_heads=params.n_head, 
        num_decoder_layers=params.n_decode_layer, dropout_p=params.dropout
    ).to(params.device)

    knowledge_encoder = Knowledge_Encoder(
        num_tokens=params.n_vocab, dim_model=params.dim_model, num_heads=params.n_head, \
        num_encoder_layers=params.n_encode_layer, dropout_p=params.dropout
    ).to(params.device)

    knowledge_manager = Knowledge_Manager(
        X_seq_len=params.X_seq_len,
        Y_seq_len=params.Y_seq_len,
        K_seq_len=params.K_seq_len,
        dim_model=params.dim_model
    ).to(params.device)

    model = [encoder, decoder, knowledge_encoder, knowledge_manager]

    parameters = (
        list(encoder.parameters())
        + list(knowledge_encoder.parameters())
        + list(knowledge_manager.parameters())
        + list(decoder.parameters())
    )

    print("loading data...")
    X, Y, K = load_data("./data/train.txt")
    dataset = PersonaDataset(X, Y, K)
    data_loader = DataLoader(dataset=dataset, batch_size=params.n_batch, shuffle=True)
    optimizer = optim.Adam(parameters, lr=params.lr)

    print("start training...")
    train(model, optimizer, data_loader)