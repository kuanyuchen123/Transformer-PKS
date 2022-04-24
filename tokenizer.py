from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def tokenize():
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"])
    tokenizer.pre_tokenizer = Whitespace()

    data = ["./data/train.txt","./data/test.txt"]
    tokenizer.train(data, trainer)
    tokenizer.save("./vocab.json")