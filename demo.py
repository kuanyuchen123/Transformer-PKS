import torch
import params
from tokenizers import Tokenizer
from model import create_pad_mask, get_seq_mask, Transformer_Encoder, Transformer_Decoder, Knowledge_Encoder, Knowledge_Manager
from beam import beam_decode

print( "Set device to {}".format(params.device))

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

encoder.load_state_dict(torch.load("./checkpoint/encoder_0.pt"))
decoder.load_state_dict(torch.load("./checkpoint/decoder_0.pt"))
knowledge_encoder.load_state_dict(torch.load("./checkpoint/k_encoder_0.pt"))
knowledge_manager.load_state_dict(torch.load("./checkpoint/k_manager_0.pt"))

encoder.eval()
decoder.eval()
knowledge_encoder.eval()
knowledge_manager.eval()

tokenizer = Tokenizer.from_file("./vocab.json")
tokenizer.enable_padding(pad_id=params.PAD, length=params.K_seq_len)

print("Enter 3 personas of bot: ")
K = []
K.append(input("Type first Knowledge: ".lower())
K.append(input("Type second Knowledge: ".lower())
K.append(input("Type third Knowledge: ".lower())

tokenized_K = torch.tensor([tokenizer.encode_batch(K)[i].ids for i in range(3)]).unsqueeze(0).to(params.device)
K_pad_mask = create_pad_mask(tokenized_K, params.PAD).to(params.device)
K_hidden = knowledge_encoder(tokenized_K, K_pad_mask)

print( "Start chatting with our chatbot!" )
while True :
    X = input("you: ")
    if X == "exit" : break 
    tokenizer.enable_padding(pad_id=params.PAD, length=params.X_seq_len)
    tokenized_X = torch.tensor(tokenizer.encode(X.lower()).ids).unsqueeze(0).to(params.device)
    X_pad_mask = create_pad_mask(tokenized_X, params.PAD).to(params.device)
    X_hidden = encoder(tokenized_X, X_pad_mask)

    K_index, selected_K = knowledge_manager(X_hidden,None,K_hidden,X_pad_mask,None,K_pad_mask)
    print(K_index)
    selected_K_pad_mask = torch.einsum('abc, ab -> ac', K_pad_mask.type(torch.FloatTensor), K_index.type(torch.FloatTensor))
    selected_K_pad_mask = selected_K_pad_mask.type(torch.BoolTensor).to(params.device)
    with torch.no_grad():
        result = beam_decode(decoder, X_hidden, X_pad_mask, selected_K, selected_K_pad_mask, params.device)

    result = tokenizer.decode(result[0])
    print("bot: {}".format(result))
