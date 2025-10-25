# practice of self attention
import torch
class multiheadattention(nn.Module):
    def __init__(self, n_embed, n_head, dropout):
        super.init()
        self.n_head = n_head
        self.n_embed = n_embed
        self.n_head_dim = n_embed//n_head

        # flash attention
        self.flash = False
        # attention module
        self.q = torch.nn.Linear(n_embed, n_embed, bias=False)
        self.k = torch.nn.Linear(n_embed, n_embed, bias=False)
        self.v = torch.nn.Linear(n_embed, n_embed, bias=False)
        self.o = torch.nn.Linear(n_embed, n_embed, bias=False)

        # dropout
        self.attn_dropout = nn.dropout(dropout)
        self.mlp_dropout = nn.dropout(dropout)
        self.residul_dropout = nn.dropout(dropout)

    def forward(self, x, mask):
        B, L, h = x.size()
        q, k, v = self.q(x), self.k(x), self.v(x)

        # reshape for multihead attention
        q = q.view(B, L, self.n_head, self.n_head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.n_head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.n_head_dim).transpose(1, 2)

        # dot product operation
        if not self.flash:
            attn = (q@k.transpose(-2, -1))/sqrt(q.size(-1)) # normalize it with the size of head
            # softmax
            attn = nn.functional.softmax(attn, dim=-1)
            y = attn@v
        else:
            y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # y shape is (B, n_head, L, L)
        # concate all the heads
        y = y.transpose(1, 2).view(B, L, h)
        # times the output matrix
        y = self.o(y)
        return y

    def mlp
        



        


