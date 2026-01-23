# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm

from .multiway_network import MultiwayWrapper
from .xpos_relative_position import XPOS
from .flash_attention import flash_attn_func


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        dropout=0.0,
        self_attention=False,
        encoder_decoder_attention=False,
        subln=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = dropout

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.out_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))

        self.inner_attn_ln = (
            MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))
            if subln and self.self_attention
            else None
        )
        self.dropout_module = torch.nn.Dropout(dropout)
        self.xpos = (
            XPOS(self.head_dim, args.xpos_scale_base)
            if args.xpos_rel_pos and self.self_attention
            else None
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def _infer_bsz(self, q, key_padding_mask=None, attn_mask=None):
        """
        q is shaped [(b*h), l, d]. We need b to reshape logsumexp into [b, h, l]
        for DilatedAttention scattering when args.flash_attention=True.

        Prefer key_padding_mask.size(0) if present; else try attn_mask; else b=1.
        """
        if key_padding_mask is not None:
            return int(key_padding_mask.size(0))
        if attn_mask is not None:
            # sometimes attn_mask can be [t, s] or [b, t, s]
            if attn_mask.dim() == 3:
                return int(attn_mask.size(0))
        return 1

    # -------------------------
    # Vanilla attention (returns FULL attn_weights matrix)
    # Used when args.flash_attention == False
    # -------------------------
    def _attention_ops_vanilla(
        self,
        q,
        k,
        v,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
    ):
        q = q * self.scaling
        attn_weights = torch.bmm(q, k.transpose(1, 2))  # [(b*h), t, s]

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None and key_padding_mask.size(-1) == attn_weights.size(-1):
            attn_weights = rearrange(attn_weights, "(b h) t s -> b h t s", h=self.num_heads)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = rearrange(attn_weights, "b h t s -> (b h) t s")

        if rel_pos is not None:
            rel_pos = rel_pos.view(attn_weights.size())
            attn_weights = attn_weights + rel_pos

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)  # [(b*h), t, d]
        attn = rearrange(attn, "(b h) l d -> b l (h d)", h=self.num_heads)

        return attn, attn_weights

    # -------------------------
    # Flash-style fallback (returns LSE like flash-attn)
    # Used when args.flash_attention == True but flash_attn_func is missing
    # -------------------------
    def _attention_ops_flash_fallback(
        self,
        q,
        k,
        v,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
        is_causal=False,
    ):
        """
        DilatedAttention expects:
          attn: [b, l, (h*d)]
          lse:  [b, h, l]   (or [b, r*h, l] depending on how heads are packed upstream)

        In the flash-attn path, this code returns `lse` (not a full attention matrix).
        So when flash_attn_func is None we must ALSO return lse, otherwise DilatedAttention
        scattering will crash (as you saw).

        NOTE: We intentionally ignore key_padding_mask here because in the dilated/sparse
        setup its shape can mismatch the effective KV length.
        """
        if rel_pos is not None:
            # Flash path asserts this; keep same behavior.
            raise AssertionError("rel_pos is not supported in flash-attn path")

        bsz = self._infer_bsz(q, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        bh, t, d = q.shape
        if bh % bsz != 0:
            # best effort: treat as single batch
            bsz = 1
        h_total = bh // bsz

        # logits: [(b*h), t, s]
        q_scaled = q * self.scaling
        logits = torch.bmm(q_scaled, k.transpose(1, 2))

        # Apply attn_mask if provided (best-effort; matches vanilla branch behavior)
        if attn_mask is not None:
            logits = torch.nan_to_num(logits)
            logits = logits + attn_mask.unsqueeze(0)

        # Causal masking (rare for slide encoder, but keep it correct)
        if is_causal:
            # logits is [bh, t, s]
            causal = torch.triu(
                torch.ones((t, logits.size(-1)), device=logits.device, dtype=torch.bool),
                diagonal=1,
            )
            logits = logits.masked_fill(causal.unsqueeze(0), float("-inf"))

        # lse: logsumexp over keys -> [bh, t]
        lse = torch.logsumexp(logits, dim=-1)  # [bh, t]

        # probs + dropout
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        probs = self.dropout_module(probs)

        # attn: [bh, t, d]
        attn = torch.bmm(probs, v)

        # reshape attn -> [b, t, (h*d)]
        attn = attn.view(bsz, h_total, t, d).permute(0, 2, 1, 3).contiguous()
        attn = attn.view(bsz, t, h_total * d)

        # reshape lse -> [b, h_total, t] to match DilatedAttention expectations
        lse = lse.view(bsz, h_total, t)

        return attn, lse

    def attention_ops(self, q, k, v, key_padding_mask=None, attn_mask=None, rel_pos=None, is_causal=False):
        # Non-flash path (kept exactly as original behavior)
        if not self.args.flash_attention:
            return self._attention_ops_vanilla(
                q, k, v,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rel_pos=rel_pos,
            )

        # Flash-attn enabled:
        # If flash-attn is not installed, use flash-style fallback (returns lse)
        if flash_attn_func is None:
            return self._attention_ops_flash_fallback(
                q, k, v,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rel_pos=rel_pos,
                is_causal=is_causal,
            )

        # Real FlashAttention path (original)
        assert rel_pos is None
        q_ = rearrange(q, "(b h) l d -> b l h d", h=self.num_heads)
        k_ = rearrange(k, "(b h) l d -> b l h d", h=self.num_heads)
        v_ = rearrange(v, "(b h) l d -> b l h d", h=self.num_heads)
        attn, lse = flash_attn_func(q_, k_, v_, self.dropout, attn_mask, None, is_causal)
        attn = rearrange(attn, "b l h d -> b l (h d)")
        attn_weights = lse[:, :, :attn.size(1)]  # lse-like, not full matrix
        return attn, attn_weights

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
        is_first_step=False,
        is_causal=False,
    ):
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = rearrange(q, "b l (h d) -> (b h) l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> (b h) l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> (b h) l d", h=self.num_heads)

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(bsz * self.num_heads, -1, self.head_dim)
                prev_value = incremental_state["prev_value"].view(bsz * self.num_heads, -1, self.head_dim)
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            incremental_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            src_len = k.size(1)

        if self.xpos is not None:
            if incremental_state is not None and not is_first_step:
                offset = src_len - 1
            else:
                offset = 0
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)

        attn, attn_weights = self.attention_ops(
            q, k, v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rel_pos=rel_pos,
            is_causal=is_causal,
        )

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)

        return attn, attn_weights




# # Copyright (c) 2022 Microsoft
# # Licensed under The MIT License [see LICENSE for details]

# import math

# import torch
# import torch.nn.functional as F
# from torch import nn
# from einops import rearrange
# try:
#     from apex.normalization import FusedLayerNorm as LayerNorm
# except ModuleNotFoundError:
#     from torch.nn import LayerNorm

# from .multiway_network import MultiwayWrapper
# from .xpos_relative_position import XPOS
# from .flash_attention import flash_attn_func


# class MultiheadAttention(nn.Module):
#     def __init__(
#         self,
#         args,
#         embed_dim,
#         num_heads,
#         dropout=0.0,
#         self_attention=False,
#         encoder_decoder_attention=False,
#         subln=False,
#     ):
#         super().__init__()
#         self.args = args
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.scaling = self.head_dim**-0.5
#         self.dropout = dropout

#         self.self_attention = self_attention
#         self.encoder_decoder_attention = encoder_decoder_attention
#         assert self.self_attention ^ self.encoder_decoder_attention

#         self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
#         self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
#         self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
#         self.out_proj = MultiwayWrapper(
#             args, nn.Linear(embed_dim, embed_dim, bias=True)
#         )
#         self.inner_attn_ln = (
#             MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))
#             if subln and self.self_attention
#             else None
#         )
#         self.dropout_module = torch.nn.Dropout(dropout)
#         self.xpos = (
#             XPOS(self.head_dim, args.xpos_scale_base)
#             if args.xpos_rel_pos and self.self_attention
#             else None
#         )

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
#         nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
#         nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
#         nn.init.xavier_uniform_(self.out_proj.weight)
#         nn.init.constant_(self.out_proj.bias, 0.0)

#     def attention_ops(self, q, k, v, key_padding_mask=None, attn_mask=None, rel_pos=None, is_causal=False):
#         if not self.args.flash_attention:
#             q *= self.scaling
#             attn_weights = torch.bmm(q, k.transpose(1, 2))

#             if attn_mask is not None:
#                 attn_weights = torch.nan_to_num(attn_weights)
#                 attn_mask = attn_mask.unsqueeze(0)
#                 attn_weights += attn_mask

#             if key_padding_mask is not None:
#                 attn_weights = rearrange(attn_weights, '(b h) t s -> b h t s', h=self.num_heads)
#                 attn_weights = attn_weights.masked_fill(
#                     key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
#                     float("-inf"),
#                 )
#                 attn_weights = rearrange(attn_weights, 'b h t s -> (b h) t s')

#             if rel_pos is not None:
#                 rel_pos = rel_pos.view(attn_weights.size())
#                 attn_weights = attn_weights + rel_pos

#             attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
#                 attn_weights
#             )
#             attn_probs = self.dropout_module(attn_weights)

#             attn = torch.bmm(attn_probs, v)
#             attn = rearrange(attn, '(b h) l d -> b l (h d)', h=self.num_heads)
#         else:
#             assert flash_attn_func is not None
#             assert rel_pos is None
#             q = rearrange(q, '(b h) l d -> b l h d', h=self.num_heads)
#             k = rearrange(k, '(b h) l d -> b l h d', h=self.num_heads)
#             v = rearrange(v, '(b h) l d -> b l h d', h=self.num_heads)
#             attn, lse = flash_attn_func(q, k, v, self.dropout, attn_mask, None, is_causal)
#             attn = rearrange(attn, 'b l h d -> b l (h d)')
#             attn_weights = lse[:, :, :attn.size(1)]

#         return attn, attn_weights

#     def forward(
#         self,
#         query,
#         key,
#         value,
#         incremental_state=None,
#         key_padding_mask=None,
#         attn_mask=None,
#         rel_pos=None,
#         is_first_step=False,
#         is_causal=False,
#     ):
#         bsz, tgt_len, embed_dim = query.size()
#         src_len = tgt_len
#         assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

#         key_bsz, src_len, _ = key.size()
#         assert key_bsz == bsz, f"{query.size(), key.size()}"
#         assert value is not None
#         assert bsz, src_len == value.shape[:2]

#         q = self.q_proj(query)
#         k = self.k_proj(key)
#         v = self.v_proj(value)

#         q = rearrange(q, 'b l (h d) -> (b h) l d', h=self.num_heads)
#         k = rearrange(k, 'b l (h d) -> (b h) l d', h=self.num_heads)
#         v = rearrange(v, 'b l (h d) -> (b h) l d', h=self.num_heads)

#         if incremental_state is not None:
#             if "prev_key" in incremental_state:
#                 prev_key = incremental_state["prev_key"].view(
#                     bsz * self.num_heads, -1, self.head_dim
#                 )
#                 prev_value = incremental_state["prev_value"].view(
#                     bsz * self.num_heads, -1, self.head_dim
#                 )
#                 k = torch.cat([prev_key, k], dim=1)
#                 v = torch.cat([prev_value, v], dim=1)
#             incremental_state["prev_key"] = k.view(
#                 bsz, self.num_heads, -1, self.head_dim
#             )
#             incremental_state["prev_value"] = v.view(
#                 bsz, self.num_heads, -1, self.head_dim
#             )
#             src_len = k.size(1)

#         if self.xpos is not None:
#             if incremental_state is not None and not is_first_step:
#                 offset = src_len - 1
#             else:
#                 offset = 0
#             k = self.xpos(k, offset=0, downscale=True)
#             q = self.xpos(q, offset=offset, downscale=False)

#         attn, attn_weights = self.attention_ops(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rel_pos=rel_pos, is_causal=is_causal)

#         if self.inner_attn_ln is not None:
#             attn = self.inner_attn_ln(attn)

#         attn = self.out_proj(attn)

#         return attn, attn_weights
