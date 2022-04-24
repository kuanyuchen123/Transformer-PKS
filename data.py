import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
import params

def load_data(path):
    with open(path, errors="ignore") as file:
        X = []
        K = []
        Y = []
        k = []

        for line in file:
            dialog_id = line.split()[0]
            if dialog_id == "1":
                k = []

            if "your persona:" in line:
                if len(k) == 3:
                    continue
                k_line = line.split("persona:")[1].strip("\n").lower()
                k.append(k_line)

            elif "__SILENCE__" not in line:
                K.append(k)
                X_line = " ".join(line.split("\t")[0].split()[1:]).lower()
                y_line = line.split("\t")[1].strip("\n").lower()
                X.append(X_line)
                Y.append(y_line)

    X_ind = []
    Y_ind = []
    K_ind = []
    tokenizer = Tokenizer.from_file("./vocab.json")

    for line in X:
        tokens = tokenizer.encode(line).ids
        X_ind.append(tokens)

    for line in Y:
        tokens = tokenizer.encode(line).ids
        Y_ind.append(tokens)

    
    for lines in K:
        k_temp = []
        for line in lines :
            tokens = tokenizer.encode(line).ids
            k_temp.append(tokens)

        K_ind.append(k_temp)
        
    return X_ind, Y_ind, K_ind

class PersonaDataset(Dataset):
    def __init__(self, X, y, K):
        X_len = max([len(line) for line in X])
        y_len = max([len(line) for line in y])
        k_len = 0
        for lines in K:
            for line in lines:
                if k_len < len(line):
                    k_len = len(line)
        
        src_X = list()
        src_Y = list()
        tgt_Y = list()
        src_K = list()

        for line in X:
            line.extend([0] * (X_len - len(line)))
            src_X.append(line)

        for line in y:
            src_line = line[:]
            tgt_line = line[:]
            src_line.insert(0, params.SOS) 
            tgt_line.append(params.EOS) 
            src_line.extend([params.PAD] * (y_len - len(src_line) + 1))
            tgt_line.extend([params.PAD] * (y_len - len(tgt_line) + 1))
            src_Y.append(src_line)
            tgt_Y.append(tgt_line)

        for lines in K:
            src_k = list()
            for line in lines:
                line.extend([params.PAD] * (k_len - len(line)))
                src_k.append(line)

            src_K.append(src_k)

        self.src_X = torch.LongTensor(src_X)
        self.src_Y = torch.LongTensor(src_Y)
        self.tgt_Y = torch.LongTensor(tgt_Y)
        self.src_K = torch.LongTensor(src_K)
        self.dataset_size = len(self.src_X)

    def __getitem__(self, index):
        src_X = self.src_X[index]
        src_Y = self.src_Y[index]
        tgt_Y = self.tgt_Y[index]
        src_K = self.src_K[index]
        return src_X, src_Y, tgt_Y, src_K

    def __len__(self):
        return self.dataset_size