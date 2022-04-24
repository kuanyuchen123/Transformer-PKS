import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import params

def create_pad_mask(matrix: torch.tensor, pad_token: int) -> torch.tensor:
    return (matrix == pad_token)

def get_seq_mask(size) -> torch.tensor:
    mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

    return mask

def make_pad_embed_zero( embed, pad_mask ) :
    """
    utterence:
        embed: (n_batch, sequence length, dim_model)
        pad_mask: (n_batch, sequence length)
    knowledge:
        embed: (n_batch, N, sequence length, dim_model)
        pad_mask: (n_batch, N, sequence length)
    """
    if len(embed.shape) == 3 :
        n_batch = embed.shape[0]
        seq_len = embed.shape[1]

        # eye: (n_batch, sequence length, sequence length)
        eye = torch.eye(seq_len).unsqueeze(0).repeat(n_batch,1,1).to(params.device)
        mask = torch.einsum('ijk,ij -> ijk', eye, ~pad_mask)
        embed = torch.bmm(embed.transpose(1,2),mask).transpose(1,2)
        return embed
    else :
        n_batch = embed.shape[0]
        N = embed.shape[1]
        seq_len = embed.shape[2]

        # eye: (n_batch, N, sequence length, sequence length)
        eye = torch.eye(seq_len).unsqueeze(0).unsqueeze(1).repeat(n_batch,N,1,1).to((params.device))
        mask = torch.einsum('ijkl,ijk -> ijkl', eye, ~pad_mask)
        embed = torch.einsum('ijkl,ijkm -> ijkl', embed, mask)
        return embed


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)
        
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        if len(token_embedding.shape) == 4 :
            pos_encoding = self.pos_encoding.unsqueeze(1).repeat(1,3,1,1)
            return self.dropout(token_embedding + pos_encoding[:token_embedding.size(0), :, :])
        else :
            return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class Transformer_Encoder(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        dropout_p
    ):
        super().__init__()
        self.dim_model = dim_model
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src, src_pad_mask=None):
        """
        src: (n_batch, sequence length)
        src_embeded + pos: (n_batch, sequence length, dim_model)
        encoder_hidden: (sequence length, n_batch, dim_model)
        """

        src_embeded = self.embedding(src) * math.sqrt(self.dim_model)
        src_embeded = self.positional_encoder(src_embeded)
        
        # src_embeded: (sequence length, n_batch, dim_model)
        src_embeded = src_embeded.permute(1,0,2)
        encoder_hidden = self.encoder(src_embeded, src_key_padding_mask=src_pad_mask)

        return encoder_hidden

class Transformer_Decoder(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_decoder_layers,
        dropout_p
        ):

        super().__init__()
        self.dim_model = dim_model
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.output_layer = nn.Linear(dim_model, num_tokens)

    def forward(self, encoder_hidden, tgt, selected_k_hidden, src_pad_mask=None, tgt_pad_mask=None, selected_K_pad_mask=None, tgt_mask=None):
        """
        tgt: (n_batch, sequence length)
        tgt_embeded + pos: (n_batch, sequence length, dim_model)
        decoder_hidden: (sequence length, n_batch, dim_model)
        output: (sequence, n_batch, num_tokens)
        """

        tgt_embeded = self.embedding(tgt) * math.sqrt(self.dim_model)
        tgt_embeded = self.positional_encoder(tgt_embeded)
        # tgt_embeded: (sequence length, n_batch, dim_model)
        tgt_embeded = tgt_embeded.permute(1,0,2)

        # concat the selected knowledge into encoder hidden
        # concat the selected knowledge pad mask into encoder pad mask
        encoder_hidden = torch.cat((encoder_hidden,selected_k_hidden.transpose(0,1)), dim=0)
        src_pad_mask = torch.cat((src_pad_mask, selected_K_pad_mask), dim=1)

        decoder_hidden = self.decoder(tgt_embeded, encoder_hidden, tgt_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.output_layer(decoder_hidden)
        output = F.log_softmax(output, dim=2)

        return decoder_hidden, output

class Knowledge_Encoder(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        dropout_p
    ):
        super().__init__()
        self.dim_model = dim_model
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

    def forward(self, K, K_pad_mask=None):
        """
        Knowledge:
            K: (n_batch, N, sequence length)
            K_embeded: (n_batch, N, sequence length, dim_model)
            knowledge_hidden: (N, sequence length, n_batch, dim_model)
        Response:
            K: (n_batch, sequence length)
            K_embeded: (n_batch, sequence length, dim_model)
            knowledge_hidden: (sequence length, n_batch, dim_model)
        """

        # encode knowledge
        if len(K.shape) == 3 :
            n_batch = K.size(0)
            N = K.size(1)
            seq_len = K.size(2)

            K_embeded = self.embedding(K) * math.sqrt(self.dim_model)
            K_embeded = self.positional_encoder(K_embeded)
            # K_embeded: (N, n_batch, sequence length, dim_model)
            K_embeded = K_embeded.transpose(0,1)
            K_pad_mask = K_pad_mask.transpose(0,1)
            knowledge_hidden = torch.zeros(N, seq_len, n_batch, self.dim_model).to(params.device)
            for i in range(N) :
                # k: (n_batch, sequence length, dim_model) -> (sequence length, n_batch, dim_model)
                k = K_embeded[i].permute(1,0,2)
                k_pad_mask = K_pad_mask[i]
                k_hidden = self.encoder(k, src_key_padding_mask=k_pad_mask)
                knowledge_hidden[i] = k_hidden

            return knowledge_hidden
        # encode response
        else :
            K_embeded = self.embedding(K) * math.sqrt(self.dim_model)
            K_embeded = self.positional_encoder(K_embeded)
            # K_embeded: (sequence length, n_batch, dim_model)
            K_embeded = K_embeded.permute(1,0,2)
            knowledge_hidden = self.encoder(K_embeded, src_key_padding_mask=K_pad_mask)

            return knowledge_hidden

class Knowledge_Manager(nn.Module):
    def __init__(
        self,
        X_seq_len,
        Y_seq_len,
        K_seq_len,
        dim_model
    ):
        super().__init__()
        self.mean_X = nn.AvgPool2d((X_seq_len,1), stride=1)
        self.mean_Y = nn.AvgPool2d((Y_seq_len,1), stride=1)
        self.mean_K = nn.AvgPool2d((K_seq_len,1), stride=1)
        self.linear1 = nn.Linear(2*dim_model,dim_model)
        self.linear2 = nn.Linear(dim_model,params.n_vocab)

    def forward(self, 
                utterence_hidden, response_hidden, knowledge_hidden,
                utterence_pad_mask, response_pad_mask, knowledge_pad_mask
                ) :
        """
        utterence_hidden: (sequence length, n_batch, dim_model)
        response_hidden: (sequence length, n_batch, dim_model)
        knowledge_hidden: (N, sequence length, n_batch, dim_model)
        mean_X: (n_batch, dim_model)
        mean_Y: (n_batch, dim_model)
        mean_K: (n_batch, N, dim_model)
        """

        if response_hidden != None :
            # utterence_hidden: (n_batch, sequence length, dim_model)
            # knowledge_hidden: (n_batch, N, sequence length, dim_model)
            utterence_hidden = make_pad_embed_zero(utterence_hidden.transpose(0,1), utterence_pad_mask)
            response_hidden = make_pad_embed_zero(response_hidden.transpose(0,1), response_pad_mask)
            knowledge_hidden = make_pad_embed_zero(knowledge_hidden.permute(2,0,1,3), knowledge_pad_mask)

            mean_X = self.mean_X(utterence_hidden).squeeze(1)                             
            mean_Y = self.mean_Y(response_hidden).squeeze(1)
            mean_K = self.mean_K(knowledge_hidden).squeeze(2)

            # prior: (n_batch, N)
            prior = F.log_softmax(
                torch.bmm(mean_X.unsqueeze(1), mean_K.transpose(1, 2)), dim=2
            ).squeeze(1)

            # posterior: (n_batch, N)
            X_cat_Y = self.linear1(torch.cat((mean_X,mean_Y), dim=1))
            posterior_logits = torch.bmm(X_cat_Y.unsqueeze(1), mean_K.transpose(1, 2)).squeeze(1)
            posterior = F.softmax(posterior_logits, dim=1)
            # K_index: (n_batch, N[one_hot])
            K_index = F.gumbel_softmax(posterior_logits,0.0001)
            # selected_K: (n_batch, sequence length, dim_model)
            selected_K = torch.einsum('abcd, ab -> acd', knowledge_hidden, K_index)

            # vocab_selected_K: (n_batch, n_vocab)
            vocab_selected_K = torch.einsum('abc, ab -> ac', mean_K, K_index)
            vocab_selected_K = F.log_softmax(self.linear2(vocab_selected_K), dim=1)

            return prior, posterior, K_index, selected_K, vocab_selected_K

        else :
            # utterence_hidden: (n_batch, sequence length, dim_model)
            # knowledge_hidden: (n_batch, N, sequence length, dim_model)
            utterence_hidden = make_pad_embed_zero(utterence_hidden.transpose(0,1), utterence_pad_mask)
            knowledge_hidden = make_pad_embed_zero(knowledge_hidden.permute(2,0,1,3), knowledge_pad_mask)

            mean_X = self.mean_X(utterence_hidden).squeeze(1)                             
            mean_K = self.mean_K(knowledge_hidden).squeeze(2)

            # prior: (n_batch, N)
            prior = F.log_softmax(
                torch.bmm(mean_X.unsqueeze(1), mean_K.transpose(1, 2)), dim=2
            ).squeeze(1)

            max_index = torch.argmax(prior, dim=1)
            K_index = torch.zeros_like(prior).scatter_(1, max_index.unsqueeze(1), 1.)
            selected_K = torch.einsum('abcd, ab -> acd', knowledge_hidden, K_index)
            return K_index, selected_K
