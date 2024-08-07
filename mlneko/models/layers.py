import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

try:
    import xformers
    use_xformers = True
except:
    use_xformers = False

class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """

    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False):
        super().__init__()

        self.num_heads = num_heads if num_heads else dim // head_dim
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.attention_dim = dim

        self.to_q = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.re_qkv = Rearrange('b l (nh dh) -> b nh l dh', nh=self.num_heads)
        self.re_out = Rearrange('b nh l dh -> b l (nh dh)')

    def forward(self, q, k=None, v=None, attn_mask=None):
        if k is None:
            k, v = q, q
        q = self.re_qkv(self.to_q(q))
        k = self.re_qkv(self.to_k(k))
        v = self.re_qkv(self.to_v(v))

        if use_xformers:
            x = xformers.ops.memory_efficient_attention(q, k, v, scale=self.scale)
        else:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, attn_mask=attn_mask)

        x = self.re_out(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)