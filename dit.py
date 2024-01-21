import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import einops

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm

from xpos_relative_position import XPOS, XPOS2D
from deepseekmoe import DeepseekMoE2D

class namespace :
    pass

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class XposMultiheadAttention2D(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        self_attention=False,
        encoder_decoder_attention=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias = True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias = True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias = True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias = True)
        self.xpos = XPOS2D(self.head_dim, embed_dim)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        need_weights = False,
        is_causal = False,
        k_offset = 0,
        q_offset = 0
    ):
        assert not is_causal
        bsz, qh, qw, embed_dim = query.size()
        bsz, kh, kw, embed_dim = key.size()

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = einops.rearrange(q, 'N H W (nhead dhead) -> (N nhead) H W dhead', nhead = self.num_heads, dhead = self.head_dim)
        k = einops.rearrange(k, 'N H W (nhead dhead) -> (N nhead) H W dhead', nhead = self.num_heads, dhead = self.head_dim)
        v = einops.rearrange(v, 'N H W (nhead dhead) -> (N nhead) H W dhead', nhead = self.num_heads, dhead = self.head_dim)

        if self.xpos is not None:
            k = self.xpos(k, offset_x = k_offset, offset_y = k_offset, downscale = True) # TODO: read paper
            q = self.xpos(q, offset_x = k_offset, offset_y = k_offset, downscale = False)

        q = einops.rearrange(q, '(N nhead) H W dhead -> (N nhead) (H W) dhead', nhead = self.num_heads, dhead = self.head_dim)
        k = einops.rearrange(k, '(N nhead) H W dhead -> (N nhead) (H W) dhead', nhead = self.num_heads, dhead = self.head_dim)
        v = einops.rearrange(v, '(N nhead) H W dhead -> (N nhead) (H W) dhead', nhead = self.num_heads, dhead = self.head_dim)

        src_len = kh * kw
        tgt_len = qh * qw

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).reshape(tgt_len, bsz, embed_dim).transpose(0, 1)

        attn = self.out_proj(attn)
        attn_weights = attn_weights.view(
            bsz, self.num_heads, tgt_len, src_len
        ).transpose(1, 0)

        attn = einops.rearrange(attn, 'N (H W) D -> N H W D', H = qh, W = qw)

        if need_weights:
            return attn, attn_weights
        else :
            return attn, None

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias = True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias = True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias = True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias = True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query, # N, L, D
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        need_weights = False,
        is_causal = False,
    ):
        assert not is_causal
        bsz, tgt_len, embed_dim = query.size()
        bsz, src_len, embed_dim = key.size()

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = einops.rearrange(q, 'N L (nhead dhead) -> (N nhead) L dhead', nhead = self.num_heads, dhead = self.head_dim)
        k = einops.rearrange(k, 'N L (nhead dhead) -> (N nhead) L dhead', nhead = self.num_heads, dhead = self.head_dim)
        v = einops.rearrange(v, 'N L (nhead dhead) -> (N nhead) L dhead', nhead = self.num_heads, dhead = self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).reshape(tgt_len, bsz, embed_dim).transpose(0, 1)

        attn = self.out_proj(attn)
        attn_weights = attn_weights.view(
            bsz, self.num_heads, tgt_len, src_len
        ).transpose(1, 0)

        if need_weights:
            return attn, attn_weights
        else :
            return attn, None
        
class FeedForward(nn.Module) :
    def __init__(self, dim, ff_dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, dim)

    def forward(self, x) :
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x

class TransformerXpos2D(nn.Module) :
    def __init__(self, dim, nhead, ff_dim, layer_norm_eps = 1e-6, use_moe = True) -> None:
        super().__init__()
        self.self_attn = XposMultiheadAttention2D(dim, nhead, self_attention = True, encoder_decoder_attention = False)
        if use_moe :
            cfg = namespace()
            cfg.n_shared_experts = 2 # 2 additional always in use
            cfg.num_experts_per_tok = 4 # Top K experts in use, total 2+4 experts used
            cfg.n_routed_experts = 24 # 24 optional experts, total 2+24 experts
            cfg.scoring_func = 'softmax'
            cfg.aux_loss_alpha = 0.001
            cfg.seq_aux = True
            cfg.norm_topk_prob = False
            cfg.hidden_size = dim
            cfg.intermediate_size = ff_dim
            cfg.moe_intermediate_size = max((int(dim * 0.75) // 16), 1) * 16 # expert size
            cfg.pretraining_tp = 1
            self.ff = DeepseekMoE2D(cfg)
        else :
            self.ff = FeedForward(dim, ff_dim)
        self.norm1 = nn.LayerNorm(dim, eps = layer_norm_eps, elementwise_affine = False)
        self.norm2 = nn.LayerNorm(dim, eps = layer_norm_eps, elementwise_affine = False)
        # init alphas to 0, shifts to 0, scales to 1
        self.scale_shifts_offset = nn.Parameter(torch.zeros(1, 1, 1, dim * 6), requires_grad = True)

    def forward(self, x, scale_shifts) :
        [alpha1, alpha2, beta1, beta2, gamma1, gamma2] = torch.chunk(scale_shifts.view(x.size(0), 1, 1, -1) + self.scale_shifts_offset, 6, dim = 3)

        residual = x
        x = self.norm1(x) * (1.0 + gamma1) + beta1
        x = self.self_attn(x, x, x)[0]
        x = x * alpha1 + residual

        residual = x
        x = self.norm2(x) * (1.0 + gamma2) + beta2
        x = self.ff(x)
        x = x * alpha2 + residual

        return x


class TransformerRegular(nn.Module) :
    def __init__(self, dim, nhead, ff_dim, layer_norm_eps = 1e-6) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(dim, nhead)
        self.ff = FeedForward(dim, ff_dim)
        self.norm1 = nn.LayerNorm(dim, eps = layer_norm_eps, elementwise_affine = False)
        self.norm2 = nn.LayerNorm(dim, eps = layer_norm_eps, elementwise_affine = False)
        # init alphas to 0, shifts to 0, scales to 1
        self.scale_shifts_offset = nn.Parameter(torch.zeros(1, 1, dim * 6), requires_grad = True)

    def forward(self, x, scale_shifts) :
        [alpha1, alpha2, beta1, beta2, gamma1, gamma2] = torch.chunk(scale_shifts.view(x.size(0), 1, -1) + self.scale_shifts_offset, 6, dim = 2)

        residual = x
        x = self.norm1(x) * (1.0 + gamma1) + beta1
        x = self.self_attn(x, x, x)[0]
        x = x * alpha1 + residual

        residual = x
        x = self.norm2(x) * (1.0 + gamma2) + beta2
        x = self.ff(x)
        x = x * alpha2 + residual

        return x


class DiTBlock(nn.Module) :
    def __init__(self, dim, nhead, ff_dim) -> None:
        super().__init__()
        self.block1 = TransformerXpos2D(dim, nhead, ff_dim)
        self.block2 = TransformerRegular(dim, nhead, ff_dim)

    def forward(self, x, base_scale_sifts: torch.Tensor, aux_tokens: torch.Tensor = None) :
        N, H, W, C = x.shape
        [scale_shifts1, scale_shifts2] = torch.chunk(base_scale_sifts, 2, dim = 1)
        x = self.block1(x, scale_shifts1)
        x = rearrange(x, 'n h w c -> n (h w) c')
        if aux_tokens is not None :
            x = torch.cat([x, aux_tokens], dim = 1)
        x = self.block2(x, scale_shifts2)
        if aux_tokens is not None :
            x = x[:, : H * W, :]
            aux_tokens = x[:, H * W: , :]
        else :
            aux_tokens = None
        x = rearrange(x, 'n (h w) c -> n h w c', h = W, w = W)
        return x, aux_tokens

class DitConditional(nn.Module):
    def __init__(
        self,
        dim,
        condition_dim,
        patch_size = 4,
        nhead = 8,
        num_aux_tokens = 4,
        in_dim = None,
        out_dim = None,
        num_blocks: int = 8,
        last_norm = False,
        learned_sinusoidal_cond = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()
        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        time_dim = dim * 2

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
        )

        self.cdt_mlp = nn.Sequential(
            nn.Linear(condition_dim, time_dim),
        )

        self.merge_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.scale_shifts_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_dim, dim * 6 * 2, bias = False)
        )

        self.init_conv = nn.Conv2d(in_dim, dim, patch_size, patch_size, 0)
        self.out_conv = nn.ConvTranspose2d(dim, out_dim, patch_size, patch_size, 0)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks) :
            self.blocks.append(DiTBlock(dim, nhead, dim * 4))

        # "register" tokens
        self.num_aux_tokens = num_aux_tokens
        if num_aux_tokens > 0 :
            self.aux_tokens = nn.Parameter(torch.randn(1, num_aux_tokens, dim))
        else :
            self.aux_tokens = None

        if last_norm :
            self.last_norm = nn.LayerNorm(dim)
        else :
            self.last_norm = None

    def forward(self, x, time, condition):
        x = self.init_conv(x)

        time_embd = self.time_mlp(time)
        cdt_embd = self.cdt_mlp(condition)
        embd = self.merge_mlp(time_embd + cdt_embd)
        base_scale_sifts = self.scale_shifts_mlp(embd)

        aux_tokens = self.aux_tokens.repeat(x.size(0), 1, 1)
        x = rearrange(x, 'n c h w -> n h w c')
        for b in self.blocks :
            x, aux_tokens = b(x, base_scale_sifts = base_scale_sifts, aux_tokens = aux_tokens)
        if self.last_norm is not None :
            x = self.last_norm(x)
        x = rearrange(x, 'n h w c -> n c h w')
        x = self.out_conv(x)
        return x
    
def test() :
    net = DitConditional(dim=512, condition_dim=128, patch_size=4,nhead= 8, num_aux_tokens= 4, in_dim = 3, out_dim=3)
    t = torch.randint(0, 1000, (8, ), dtype = torch.int32)
    c = torch.randn(8, 128)
    x = torch.randn(8, 3, 32, 32)
    out = net(x, t, c)
    print(out.shape)
    l = F.l1_loss(out, torch.randn_like(out))
    l.backward()

if __name__ == '__main__' :
    test()
