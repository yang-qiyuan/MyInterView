import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Moudle):
    def __init__(self, n_embed, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.n_embed = n_embed
        self.head_dim = n_embed // n_heads

        # Q, K, V
        self.q = nn.Linear(n_embed, n_embed, bias=False)
        self.k = nn.Linear(n_embed, n_embed, bias=False)
        self.v = nn.linear(n_embed, n_embed, bias=False)
        # out projection
        self.o = nn.Linear(n_embed, n_embed, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, C = x.size() # B: batch size, L: sequence length, C: n_embed

        # calculate query key values for all heads in batch 
        q, k, v = self.q(x), self.k(x), self.v(x)
        # size: B, n_heads, L, n_embed
        k = k.view(B, L, self.n_heads, C//self.n_heads).transpose(1, 2)
        q = q.view(B, L, self.n_heads, C//self.n_heads).transpose(1, 2)
        v = v.view(B, L, self.n_heads, C//self.n_heads).transpose(1, 2)
        # scaled dot-product attention
        # size: B, n_heads, L, L
        att = (q @ k.transpose(-2, -1)*(1.0 / math.sqrt(k.size(-1))))
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # (B, n_heads, L, L) @ (B, n_heads, L, C//n_heads) -> (B, n_heads, L, C//n_heads)
        y = att @ v 
        # concatenate all heads
        # contiguous makes the tensor stored in a contiguous chunk of memory
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = self.resid_dropout(self.o(y))
        return y

# Transformer Block 
class Block(nn.Moudle):
    def __init__(self, n_embed, n_heads, dropout):
        super(Block, self).__init__()
        self.attn = MultiHeadAttention(n_embed, n_heads, dropout)
        self.mlp = nn.sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.GELU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )
        # layer norm
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        # residual connection
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, )
        
# Byte Pair Encoding
"""
Byte Pair Encoding (BPE) is a simple data compression technique that is used to convert a list of words into a list of subwords.
The algorithm works by iteratively merging the most frequent pair of consecutive bytes in a sequence.
Collected from the original paper: https://arxiv.org/abs/1508.07909
"""
class BPE():
    def get_stats(self, vocab):
        """
        Given a list of integers, return a dictionary of counts of consecutive pairs
        Example: {"l o w": 5, "l o w e r": 2} -> {"lo w": 5, "lo w e r": 2} for one merge
        Optionally allows to update an existing dictionary of counts
        """
        pairs = defaultdict(int)
        for word, freq in vocabs.items():
            symbols = word.split()
            # calculate the frequency of alphabet pairs
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge(self, pair, vocab_in):
        """
        Merge frequency pairs of characters in a word
        pair: tuple of two characters
        vocab_in: dictionary of word frequencies
        """
        v_out = {}
        bigram = " ".join(pair)
        for word in vocab_in:
            # merge the most frequent pair
            w_out = word.replace(bigram, "".join(pair))
            v_out[w_out] = vocab_in[word]
        return v_out
    
    def train(self, n_merges):
        """
        Train the BPE model
        n_merges: number of iterations to merge the most frequent pair of characters
        """
        for _ in range(n_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge(best, vocab)
        return vocab











