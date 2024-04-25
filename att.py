import torch
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads = embed_size), "Embed size needs to be divisible by heads"
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bais = False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        #split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        # einsum is used for matrix multiplication of multiple dimentions
        energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))


        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)
        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        #attention shape : (N, heads, query_len, key_len)
        # values_shape : (N, value_len, heads, head_dim)
        # after matrix_mul : (N, heads, query_len, head_dim)

        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.Relu(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self,value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(self,
                src_vocab_size, 
                embed_size, 
                num_layers, 
                heads, 
                device, 
                forward_expansion, 
                dropout, 
                max_length):
        
        super(Encoder, self).__init___()
        self.embed_size = embed_size
        self.heads = heads
        self.dropout = dropout
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)


        self.layers = nn.ModuleLlist(
            [
            TransformerBlock(
                embed_size, 
                heads,
                dropout = dropout,
                forward_expansion = forward_expansion,
            )
            ]
        )
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, x, mask):
        # N = num of examples
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x)+self.position_embedding(positions))
        
        for layer in self.layers:
            #we are in eocoder, value, key, query are gonna be same : out
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size,heads,forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.Layers(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
    

# X is input, x,x,x goes to attention, then norm, then to transformer which is same part in both encoder and encoder.
# it takes input Q,V,K and output after passing it from attention layer, norm layer



class Decoder(nn.Module):
    def __init__(self, 
                 trg_vocab_size,
                 num_layers,
                 embed_size, 
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.embedding








