import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu, save_checkpoints, load_checkpoints
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
# from torchtext.data import Field, BucketIterator
from torchtext.legacy.data import BucketIterator


spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

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
            forward_expansion,
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
        #this makes sure that network is aware of position of words
        embed_trg = self.dropout((self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)))
        src_padding_mask = self.make_src_mask(src)
        #generate_sq_subsequence_mask is a function from transformers library just to create trg_mask
        trg_mask = self.transformer.generate_square_subsequence_mask(trg_seq_length).to(self.device)

        out = self.transformer(embed_src, embed_trg, src_key_padding_mask = src_padding_mask,
        tgt_mask = trg_mask)
        out = self.fc_out(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

# Training hyperparameters
num_epochs = 10
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 32
src_pad_idx = english.vocab.stoi["<pad>"]

# Tensorboard for nice plots
writer = SummaryWriter("runs/loss_plot")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits( (train_data, 
                                valid_data, 
                                test_data), 
                                batch_size = batch_size, 
                                sort_within_batch = True, 
                                sort_key = lambda x: len(x.src), 
                                device = device)


model = Transformer(embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout, max_len, device).to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)
#pad_idx is irrelavent to compute loss and cost so we need to set it to ignore_index

if load_model:
    load_checkpoints(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

for epoch in range(epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")
    
    if save_model:
        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
        }
        save_checkpoints(checkpoint)
    model.eval()
    translated_sentence = translate_sentence(model, sentence, german, english, device, max_length = 50)
    print(f"Translated example sentence: \n {translated_sentence}")
    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target[:-1])
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)  #to prevent exploding gradient
        optimizer.step() #update weights

        writer.add_scalar("Training loss", loss, global_step = step)  #for tensorboard
        step += 1

score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score*100:.2f}")   # 2f is for 2 decimal places






    