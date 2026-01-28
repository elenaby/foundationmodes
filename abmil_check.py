

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mil_template import MIL
from src.models.layers import GlobalAttention, GlobalGatedAttention, create_mlp

from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel


MODEL_TYPE = "abmil"


# -------------------------
# Debug helpers
# -------------------------
def _tensor_info(x: Any, name: str = "tensor") -> str:
    if x is None:
        return f"{name}=None"
    if not torch.is_tensor(x):
        return f"{name}: type={type(x)} (not torch.Tensor)"
    return (
        f"{name}: type=torch.Tensor shape={tuple(x.shape)} dtype={x.dtype} "
        f"device={x.device} requires_grad={x.requires_grad}"
    )


def _finite_check(x: torch.Tensor, name: str):
    if not torch.is_tensor(x):
        raise TypeError(f"[finite_check] {name} must be torch.Tensor, got {type(x)}")
    if not torch.isfinite(x).all():
        bad = x[~torch.isfinite(x)]
        example = bad.flatten()[:8].detach().cpu()
        raise ValueError(f"[finite_check] Non-finite values in {name}. Example: {example}")


def _maybe_stats(x: torch.Tensor) -> str:
    with torch.no_grad():
        return (
            f"min={float(x.min().item()):.4f} "
            f"max={float(x.max().item()):.4f} "
            f"mean={float(x.mean().item()):.4f}"
        )


# -------------------------
# ABMIL (Regression)
# -------------------------
class ABMIL(MIL):
    """
    ABMIL for REGRESSION (single- or multi-output).

    Debug controls:
      - debug_level:
          0 = off
          1 = key entry/exit prints
          2 = print after every step
          3 = step prints + extra stats
      - debug_max_steps:
          Only print for first N forward() calls. 0 means unlimited.
      - debug_slice_patches:
          Slice length used for finite checks/stats to avoid scanning huge slides.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        dropout: float = 0.25,
        attn_dim: int = 384,
        gate: bool = True,
        output_dim: int = 1,
        debug: bool = False,               # backward compat (if True -> debug_level=2 unless user set)
        debug_level: int = 0,
        debug_max_steps: int = 0,
        debug_slice_patches: int = 2048,
    ):
        # MIL parent uses num_classes; for regression we interpret it as output_dim
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=output_dim)

        # Parameter sanity checks
        if in_dim <= 0 or embed_dim <= 0:
            raise ValueError(f"in_dim and embed_dim must be > 0. Got {in_dim=}, {embed_dim=}")
        if num_fc_layers < 1:
            raise ValueError(f"num_fc_layers must be >= 1, got {num_fc_layers}")
        if attn_dim <= 0:
            raise ValueError(f"attn_dim must be > 0, got {attn_dim}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")

        # Backward compatible behaviour
        if debug and debug_level == 0:
            debug_level = 2

        self.debug_level = int(debug_level)
        self.debug_max_steps = int(debug_max_steps)
        self.debug_slice_patches = int(debug_slice_patches)

        self._debug_forward_calls = 0

        # Modules
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )

        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1,  # K=1 attention head for ABMIL
        )

        # Regression head
        self.regressor = nn.Linear(embed_dim, output_dim)

        # Compatibility alias (older code expects classifier)
        self.classifier = self.regressor

        self.initialize_weights()

        if self.debug_level >= 1:
            print(
                "\n[ABMIL:init]\n"
                f"  in_dim={in_dim} embed_dim={embed_dim} num_fc_layers={num_fc_layers}\n"
                f"  dropout={dropout} attn_dim={attn_dim} gate={gate} output_dim={output_dim}\n"
                f"  patch_embed={self.patch_embed.__class__.__name__}\n"
                f"  global_attn={self.global_attn.__class__.__name__}\n"
                f"  regressor={self.regressor.__class__.__name__} weight={tuple(self.regressor.weight.shape)}\n"
                f"  debug_level={self.debug_level} debug_max_steps={self.debug_max_steps} "
                f"debug_slice_patches={self.debug_slice_patches}\n"
            )
            n_params = sum(p.numel() for p in self.parameters())
            n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"[ABMIL:init] params total={n_params:,} trainable={n_train:,}\n")

    # ---- debug gating ----
    def _dbg_on(self) -> bool:
        if self.debug_level <= 0:
            return False
        if self.debug_max_steps <= 0:
            return True
        return self._debug_forward_calls <= self.debug_max_steps

    def _p(self, msg: str):
        if self._dbg_on():
            print(msg, flush=True)

    def _pt(self, x: Any, name: str):
        if self._dbg_on():
            self._p(_tensor_info(x, name))

    def _ps(self, x: torch.Tensor, name: str):
        if self._dbg_on() and self.debug_level >= 3 and torch.is_tensor(x):
            self._p(f"  stats({name}): {_maybe_stats(x)}")

    # -------------------------
    # Forward attention
    # -------------------------
    def forward_attention(
        self,
        h: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        attn_only: bool = True,
        tag: str = "",
    ):
        if self._dbg_on():
            self._p(f"\n[ABMIL:forward_attention] ENTER {tag}".rstrip())
            self._pt(h, "h_in")
            if attn_mask is not None:
                self._pt(attn_mask, "attn_mask_in")
            self._p(f"  attn_only={attn_only}")

        if not torch.is_tensor(h):
            raise TypeError(f"forward_attention expects torch.Tensor, got {type(h)}")

        if h.dim() == 2:
            h = h.unsqueeze(0)
            if self.debug_level >= 2:
                self._p("[step] h = h.unsqueeze(0)  # [M,D]->[1,M,D]")
                self._pt(h, "h_[1,M,D]")

        if h.dim() != 3:
            raise ValueError(f"forward_attention expects [B,M,D] or [M,D], got {tuple(h.shape)}")

        B, M, D = h.shape
        if D != self.in_dim:
            raise ValueError(f"Expected input dim {self.in_dim}, got {D} with shape={tuple(h.shape)}")

        sl = min(self.debug_slice_patches, M)
        _finite_check(h[:, :sl, :], "h_in_slice")
        if self.debug_level >= 2:
            self._p(f"[step] finite_check(h[:, :{sl}, :]) OK")
            if self.debug_level >= 3:
                self._ps(h[:, :sl, :], "h_in_slice")

        h_embed = self.patch_embed(h)  # [B,M,E]
        if self.debug_level >= 2:
            self._p("[step] h_embed = self.patch_embed(h)")
            self._pt(h_embed, "h_embed_[B,M,E]")
            if self.debug_level >= 3:
                self._ps(h_embed[:, :sl, :], "h_embed_slice")

        _finite_check(h_embed[:, :sl, :], "h_embed_slice")

        A_raw = self.global_attn(h_embed)  # [B,M,K]
        if self.debug_level >= 2:
            self._p("[step] A_raw = self.global_attn(h_embed)")
            self._pt(A_raw, "A_raw_[B,M,K]")
            if self.debug_level >= 3:
                self._ps(A_raw, "A_raw")

        if A_raw.dim() != 3 or A_raw.shape[0] != B or A_raw.shape[1] != M:
            raise ValueError(f"global_attn output expected [B,M,K], got {tuple(A_raw.shape)}")

        A_base = torch.transpose(A_raw, -2, -1)  # [B,K,M]
        if self.debug_level >= 2:
            self._p("[step] A_base = transpose(A_raw, -2, -1)  # [B,M,K]->[B,K,M]")
            self._pt(A_base, "A_base_[B,K,M]")
            if self.debug_level >= 3:
                self._ps(A_base, "A_base")

        if attn_mask is not None:
            if not torch.is_tensor(attn_mask):
                raise TypeError(f"attn_mask must be torch.Tensor, got {type(attn_mask)}")
            if attn_mask.dim() != 2:
                raise ValueError(f"attn_mask must be [B,M], got {tuple(attn_mask.shape)}")
            if attn_mask.shape[0] != B or attn_mask.shape[1] != M:
                raise ValueError(f"attn_mask must match [B,M]. mask={tuple(attn_mask.shape)} vs h={tuple(h.shape)}")

            if attn_mask.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                if self.debug_level >= 2:
                    self._p("[step] attn_mask = attn_mask.float()")
                attn_mask = attn_mask.float()

            A_base = A_base + (1.0 - attn_mask).unsqueeze(1) * torch.finfo(A_base.dtype).min
            if self.debug_level >= 2:
                self._p("[step] A_base = A_base + (1-attn_mask)*(-inf)")
                self._pt(A_base, "A_base_masked_[B,K,M]")

        if self._dbg_on():
            self._p(f"[ABMIL:forward_attention] EXIT {tag}".rstrip())

        return A_base if attn_only else (h_embed, A_base)

    # -------------------------
    # Forward features (pooling)
    # -------------------------
    def forward_features(
        self,
        h: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = True,
        tag: str = "",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if self._dbg_on():
            self._p(f"\n[ABMIL:forward_features] ENTER {tag}".rstrip())
            self._pt(h, "h_in")
            self._p(f"  return_attention={return_attention}")

        h_embed, A_base = self.forward_attention(h, attn_mask=attn_mask, attn_only=False, tag=tag)

        if self.debug_level >= 2:
            self._p("[step] (h_embed, A_base) = forward_attention(..., attn_only=False)")
            self._pt(h_embed, "h_embed_[B,M,E]")
            self._pt(A_base, "A_base_[B,K,M]")

        A = F.softmax(A_base, dim=-1)  # [B,K,M]
        if self.debug_level >= 2:
            self._p("[step] A = softmax(A_base, dim=-1)")
            self._pt(A, "A_softmax_[B,K,M]")
            if self.debug_level >= 3:
                with torch.no_grad():
                    s = A.sum(dim=-1)
                    self._p(f"  softmax sum over M: shape={tuple(s.shape)} "
                            f"min={float(s.min().item()):.6f} max={float(s.max().item()):.6f}")

        wsi_feat = torch.bmm(A, h_embed).squeeze(dim=1)  # [B,E]
        if self.debug_level >= 2:
            self._p("[step] wsi_feat = bmm(A, h_embed).squeeze(1)  # [B,K,E]->[B,E]")
            self._pt(wsi_feat, "wsi_feat_[B,E]")
            if self.debug_level >= 3:
                self._ps(wsi_feat, "wsi_feat")

        if wsi_feat.dim() != 2:
            raise ValueError(f"wsi_feat expected [B,E], got {tuple(wsi_feat.shape)}")

        _finite_check(wsi_feat, "wsi_feat")

        log_dict = {
            "attention": A_base if return_attention else None,
            "attention_softmax": A if return_attention else None,
        }

        if self._dbg_on():
            self._p(f"[ABMIL:forward_features] EXIT {tag}".rstrip())

        return wsi_feat, log_dict

    # -------------------------
    # Head
    # -------------------------
    def forward_head(self, h: torch.Tensor, tag: str = "") -> torch.Tensor:
        if self._dbg_on():
            self._p(f"\n[ABMIL:forward_head] ENTER {tag}".rstrip())
            self._pt(h, "h_in_[B,E]")

        if not torch.is_tensor(h):
            raise TypeError(f"forward_head expects torch.Tensor, got {type(h)}")
        if h.dim() != 2:
            raise ValueError(f"forward_head expects [B,E], got {tuple(h.shape)}")
        if h.shape[1] != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {h.shape[1]}")

        pred = self.regressor(h)  # [B,output_dim]
        if self.debug_level >= 2:
            self._p("[step] pred = self.regressor(h)")
            self._pt(pred, "pred_[B,output_dim]")
            if self.debug_level >= 3:
                self._ps(pred, "pred")

        _finite_check(pred, "pred")

        if self._dbg_on():
            self._p(f"[ABMIL:forward_head] EXIT {tag}".rstrip())

        return pred

    # -------------------------
    # Loss (regression-safe)
    # -------------------------
    @staticmethod
    def compute_loss(
        loss_fn: Optional[nn.Module] = None,
        label: Optional[torch.Tensor] = None,
        predictions: Optional[torch.Tensor] = None,
        debug_level: int = 0,
        tag: str = "",
    ) -> Optional[torch.Tensor]:

        if loss_fn is None or label is None or predictions is None:
            if debug_level >= 2:
                print(f"\n[ABMIL:compute_loss] {tag} loss_fn/label/predictions is None -> None".rstrip(), flush=True)
            return None

        if not torch.is_tensor(label):
            raise TypeError(f"label must be torch.Tensor, got {type(label)}")
        if not torch.is_tensor(predictions):
            raise TypeError(f"predictions must be torch.Tensor, got {type(predictions)}")

        if debug_level >= 2:
            print(f"\n[ABMIL:compute_loss] ENTER {tag}".rstrip(), flush=True)
            print(_tensor_info(predictions, "pred_in"), flush=True)
            print(_tensor_info(label, "label_in"), flush=True)
            print(f"  loss_fn={loss_fn.__class__.__name__}", flush=True)

        # Align shapes for output_dim=1
        if predictions.dim() == 2 and predictions.shape[-1] == 1 and label.dim() == 1:
            predictions = predictions.squeeze(-1)
            if debug_level >= 2:
                print("[step] predictions = predictions.squeeze(-1)  # [B,1]->[B]", flush=True)

        if predictions.dim() == 2 and predictions.shape[-1] == 1 and label.dim() == 2 and label.shape[-1] == 1:
            predictions = predictions.squeeze(-1)
            label = label.squeeze(-1)
            if debug_level >= 2:
                print("[step] squeeze both pred and label  # [B,1]->[B]", flush=True)

        if predictions.shape[0] != label.shape[0]:
            raise ValueError(f"Batch mismatch: pred batch={predictions.shape[0]} vs label batch={label.shape[0]}")

        loss = loss_fn(predictions, label)

        if debug_level >= 2:
            print(_tensor_info(predictions, "pred_aligned"), flush=True)
            print(_tensor_info(label, "label_aligned"), flush=True)
            print(f"[step] loss = loss_fn(predictions, label) -> {float(loss.item()):.6f}", flush=True)
            print(f"[ABMIL:compute_loss] EXIT {tag}".rstrip(), flush=True)

        return loss

    # -------------------------
    # FORWARD (prints after every step)
    # -------------------------
    def forward(
        self,
        h: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
        label: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_slide_feats: bool = False,
        tag: str = "",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        self._debug_forward_calls += 1

        if self._dbg_on():
            self._p(f"\n[ABMIL:forward] ENTER {tag}".rstrip())
            self._pt(h, "h_in")
            if label is not None:
                self._pt(label, "label_in")
            if attn_mask is not None:
                self._pt(attn_mask, "attn_mask_in")
            self._p(f"  return_attention={return_attention} return_slide_feats={return_slide_feats}")
            self._p(f"  debug_level={self.debug_level} forward_call={self._debug_forward_calls}")

        if not torch.is_tensor(h):
            raise TypeError(f"forward expects torch.Tensor, got {type(h)}")

        # Step 1: promote [M,D] -> [1,M,D]
        if h.dim() == 2:
            h = h.unsqueeze(0)
            if self.debug_level >= 2:
                self._p("[step] h = h.unsqueeze(0)  # [M,D]->[1,M,D]")
                self._pt(h, "h_[1,M,D]")

        # Step 2: validate 3D
        if h.dim() != 3:
            raise ValueError(f"forward expects [B,M,D] or [M,D], got {tuple(h.shape)}")

        B, M, D = h.shape
        if self.debug_level >= 2:
            self._p(f"[step] unpack shapes: B={B}, M={M}, D={D}")

        # Step 3: finite check slice
        sl = min(self.debug_slice_patches, M)
        _finite_check(h[:, :sl, :], "h_in_slice")
        if self.debug_level >= 2:
            self._p(f"[step] finite_check(h[:, :{sl}, :]) OK")
            if self.debug_level >= 3:
                self._ps(h[:, :sl, :], "h_in_slice")

        # Step 4: forward_features
        wsi_feats, log_dict = self.forward_features(
            h, attn_mask=attn_mask, return_attention=return_attention, tag=tag
        )
        if self.debug_level >= 2:
            self._p("[step] (wsi_feats, log_dict) = self.forward_features(...)")
            self._pt(wsi_feats, "wsi_feats_[B,E]")
            self._p(f"[step] log_dict keys={list(log_dict.keys())}")

        # Step 5: forward_head
        pred = self.forward_head(wsi_feats, tag=tag)
        if self.debug_level >= 2:
            self._p("[step] pred = self.forward_head(wsi_feats)")
            self._pt(pred, "pred_[B,output_dim]")

        # Step 6: compute_loss
        reg_loss = ABMIL.compute_loss(
            loss_fn=loss_fn,
            label=label,
            predictions=pred,
            debug_level=self.debug_level,
            tag=tag,
        )
        if self.debug_level >= 2:
            self._p("[step] reg_loss = compute_loss(...)")
            self._pt(reg_loss, "reg_loss_scalar")

        # Step 7: results dict
        results_dict = {
            "predictions": pred,
            "logits": pred,  # alias for older code
            "loss": reg_loss,
        }
        if self.debug_level >= 2:
            self._p("[step] results_dict created with keys=['predictions','logits','loss']")

        # Step 8: logging dict
        log_dict["loss"] = float(reg_loss.item()) if reg_loss is not None else -1.0
        if self.debug_level >= 2:
            self._p(f"[step] log_dict['loss'] set to {log_dict['loss']}")

        if return_slide_feats:
            log_dict["slide_feats"] = wsi_feats
            if self.debug_level >= 2:
                self._p("[step] log_dict['slide_feats'] = wsi_feats")
                self._pt(log_dict["slide_feats"], "slide_feats")

        if self._dbg_on():
            self._p(f"[ABMIL:forward] EXIT {tag}".rstrip())

        return results_dict, log_dict


# -------------------------
# HF config/wrapper
# -------------------------
class ABMILRegressionConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(
        self,
        gate: bool = True,
        embed_dim: int = 512,
        attn_dim: int = 384,
        num_fc_layers: int = 1,
        dropout: float = 0.25,
        in_dim: int = 1024,
        output_dim: int = 1,
        debug: bool = False,
        debug_level: int = 0,
        debug_max_steps: int = 0,
        debug_slice_patches: int = 2048,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gate = gate
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout
        self.in_dim = in_dim
        self.output_dim = output_dim
        self.debug = debug
        self.debug_level = debug_level
        self.debug_max_steps = debug_max_steps
        self.debug_slice_patches = debug_slice_patches
        self.auto_map = {
            "AutoConfig": "modeling_abmil.ABMILRegressionConfig",
            "AutoModel": "modeling_abmil.ABMILModel",
        }


class ABMILModel(PreTrainedModel):
    config_class = ABMILRegressionConfig

    def __init__(self, config: ABMILRegressionConfig, **kwargs):
        self.config = config
        for k, v in kwargs.items():
            setattr(config, k, v)

        super().__init__(config)

        self.model = ABMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            attn_dim=config.attn_dim,
            gate=config.gate,
            output_dim=config.output_dim,
            debug=config.debug,
            debug_level=getattr(config, "debug_level", 0),
            debug_max_steps=getattr(config, "debug_max_steps", 0),
            debug_slice_patches=getattr(config, "debug_slice_patches", 2048),
        )

        # Expose common methods
        self.forward = self.model.forward
        self.forward_attention = self.model.forward_attention
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier


AutoConfig.register(ABMILRegressionConfig.model_type, ABMILRegressionConfig)
AutoModel.register(ABMILRegressionConfig, ABMILModel)


