import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu, save_checkpoints, load_checkpoints
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

spacy_ger = spacy.load("de")
spacy_eng = spacy.load("en")

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize = tokenize_ger, lower = True, init
= "<sos>", eos_token = "<eos>")

english = Field(tokenize = tokenize_eng, lower = True, init
= "<sos>", eos_token = "<eos>")

train_data, valid_data, test_data = Multi30k.splits(exts = (".de", ".en"), fields = (german, english))

german.build_vocab(train_data, max_size = 10000, min_freq = 2)
english.build_vocab(train_data, max_size = 10000, min_freq = 2)

class Transformer(nn.Module):
    def __init__(self,
                embedding_size,
                src_vocab_size,
                trg_vocab_size,
                src_pad_idx,
                num_heads,
                num_encoder_layers,
                num_decoder_layers,
                forward_expansion,
                dropout,
                max_len,
                device
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)
        self.device = device

        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expression,
            dropout,
        )

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        # src shape : (src_len, N)
        # but torch implementation of transformers takes it input as : (N, src_len)
        src_mask = src.transpose(0,1) == self.src_pad_idx
        return src_mask

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (torch.arange(0,src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device))
        trg_positions = (torch.arange(0,trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(self.device))

        embed_src = self.dropout((self.src_word_embedding(src) + self.src_position_embedding(src_positions)))
        #this makes sure that networl is aware of position of words
        embed_trg = self.dropout((self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)))
        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequence_mask



    