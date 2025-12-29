from __future__ import annotations

import json
import os
from dataclasses import dataclass
from dataclasses import asdict
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

try:
    from diffusers import DDIMScheduler  # type: ignore
except Exception:  # pragma: no cover
    # Keep the rest of this module usable in environments without diffusers
    # (e.g. a minimal MMDet3 conda env for training-only experiments).
    DDIMScheduler = None  # type: ignore


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule used by DiffusionDet (copied in spirit to keep betas identical).
    Returns betas in float32 with shape (timesteps,).
    """
    if timesteps <= 0:
        raise ValueError(f"timesteps must be > 0, got {timesteps}")
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).to(dtype=torch.float32)


def build_ddim_scheduler_from_betas(betas: torch.Tensor) -> DDIMScheduler:
    """
    Build a Diffusers DDIMScheduler that matches the given betas and predicts x0 (prediction_type="sample").
    """
    if DDIMScheduler is None:
        raise ImportError("`diffusers` is required for build_ddim_scheduler_from_betas(); install `diffusers` first.")
    betas = betas.detach().to(dtype=torch.float32).cpu()
    betas_np = np.asarray(betas.numpy())
    return DDIMScheduler(
        num_train_timesteps=int(betas_np.shape[0]),
        trained_betas=betas_np,
        prediction_type="sample",
        clip_sample=False,
        set_alpha_to_one=False,
    )


KeepProbSchedule = Literal["sqrt_alphas_cumprod", "alphas_cumprod", "linear", "constant"]
D3PMKernel = Literal["mask", "uniform"]


@dataclass(frozen=True)
class D3PMConfig:
    num_classes: int
    kernel: D3PMKernel = "mask"
    keep_prob_schedule: KeepProbSchedule = "sqrt_alphas_cumprod"
    keep_prob_const: float = 0.0
    keep_prob_power: float = 1.0
    keep_prob_min: float = 0.0


class D3PMLabelScheduler:
    """
    Minimal D3PM-like scheduler for labels with an absorbing unk state (id=K).

    This mirrors the "mask/uniform + keep_prob(t)" logic used in this repo's Detectron2 DiffusionDet,
    so the same knobs can be exercised via a Diffusers-style pipeline.
    """

    def __init__(self, *, betas: torch.Tensor, config: D3PMConfig):
        self.config = config
        self.num_states = int(config.num_classes) + 1
        self.unk_id = int(config.num_classes)

        betas = betas.detach().to(dtype=torch.float32).cpu()
        self.betas = betas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = alphas_cumprod  # (T,)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # (T,)
        self.num_train_timesteps = int(betas.shape[0])

    def to_dict(self) -> dict:
        return {"betas": self.betas.tolist(), "config": asdict(self.config)}

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, "scheduler_config.json")
        payload = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "D3PMLabelScheduler":
        path = pretrained_model_name_or_path
        if os.path.isdir(path):
            path = os.path.join(path, "scheduler_config.json")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "betas" not in payload or "config" not in payload:
            raise ValueError("Invalid D3PMLabelScheduler config; expected keys: 'betas', 'config'")
        betas = torch.as_tensor(payload["betas"], dtype=torch.float32)
        cfg_dict = dict(payload["config"])
        config = D3PMConfig(**cfg_dict)
        return cls(betas=betas, config=config)

    @classmethod
    def from_cosine_schedule(cls, *, timesteps: int, config: D3PMConfig) -> "D3PMLabelScheduler":
        betas = cosine_beta_schedule(int(timesteps))
        return cls(betas=betas, config=config)

    def keep_prob(self, t: torch.Tensor) -> torch.Tensor:
        schedule = self.config.keep_prob_schedule
        t = t.to(dtype=torch.long).clamp(min=0, max=self.num_train_timesteps - 1)
        if schedule == "sqrt_alphas_cumprod":
            kp = self.sqrt_alphas_cumprod[t.cpu()].to(device=t.device, dtype=torch.float32)
        elif schedule == "alphas_cumprod":
            kp = self.alphas_cumprod[t.cpu()].to(device=t.device, dtype=torch.float32)
        elif schedule == "linear":
            denom = float(max(self.num_train_timesteps - 1, 1))
            kp = 1.0 - (t.to(dtype=torch.float32) / denom)
        elif schedule == "constant":
            kp = torch.full_like(t, float(self.config.keep_prob_const), dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported keep_prob_schedule={schedule!r}")

        power = float(self.config.keep_prob_power)
        if power != 1.0:
            kp = kp.clamp(0.0, 1.0) ** power
        kp = kp.clamp(min=float(self.config.keep_prob_min), max=1.0)
        return kp

    def prior(self, batch: int, num_nodes: int, *, device: torch.device) -> torch.Tensor:
        """
        Inference prior: all unk distribution, shape (B, N, K+1).
        """
        probs = torch.zeros((batch, num_nodes, self.num_states), device=device, dtype=torch.float32)
        probs[:, :, self.unk_id] = 1.0
        return probs

    def q_probs(self, c0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward corruption q(c_t | c_0) as categorical distribution over {0..K-1, unk(K)}.

        Args:
            c0: (B, N) long in [0..K] where K is unk
            t: (B,) long (shared with box diffusion step)
        Returns:
            probs: (B, N, K+1) float, rows sum to 1
        """
        if c0.dim() != 2:
            raise ValueError(f"Expected c0 to be (B,N), got shape={tuple(c0.shape)}")
        if t.dim() != 1 or t.shape[0] != c0.shape[0]:
            raise ValueError(f"Expected t to be (B,), got shape={tuple(t.shape)}")

        device = c0.device
        keep = self.keep_prob(t).to(device=device, dtype=torch.float32).view(-1, 1, 1)
        c0 = c0.to(dtype=torch.long).clamp(min=0, max=self.num_states - 1)
        one_hot = F.one_hot(c0, num_classes=self.num_states).to(dtype=torch.float32)

        kernel = self.config.kernel
        if kernel == "mask":
            probs = one_hot * keep
            is_unk = (c0 == self.unk_id).to(dtype=torch.float32).unsqueeze(-1)
            # non-unk -> unk with (1-keep)
            probs[:, :, self.unk_id] = probs[:, :, self.unk_id] + (1.0 - keep.squeeze(-1)) * (1.0 - is_unk.squeeze(-1))
            # unk stays unk
            probs = probs * (1.0 - is_unk) + one_hot * is_unk
        elif kernel == "uniform":
            uniform = torch.full((c0.shape[0], c0.shape[1], self.num_states), 1.0 / float(self.num_states), device=device)
            probs = one_hot * keep + uniform * (1.0 - keep)
        else:
            raise ValueError(f"Unsupported kernel={kernel!r}")

        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return probs

    def infer_update(self, logits: torch.Tensor, *, t_next: int, device: torch.device) -> torch.Tensor:
        """
        Heuristic reverse update used by this repo's DiffusionDet:
        combine predicted p(c0|x_t) with keep_prob(t_next) into a new categorical state.
        """
        if logits.dim() != 3:
            raise ValueError(f"Expected logits (B,N,K), got shape={tuple(logits.shape)}")
        batch, num_nodes, num_classes = logits.shape
        if num_classes != self.config.num_classes:
            raise ValueError(f"logits last dim must be num_classes={self.config.num_classes}, got {num_classes}")

        p0 = torch.softmax(logits.to(dtype=torch.float32), dim=-1)
        if t_next < 0:
            keep = torch.ones((batch,), device=device, dtype=torch.float32)
        else:
            keep = self.keep_prob(torch.full((batch,), int(t_next), device=device, dtype=torch.long)).to(device=device)
        keep = keep.view(batch, 1, 1)

        out = torch.zeros((batch, num_nodes, self.num_states), device=device, dtype=torch.float32)
        if self.config.kernel == "uniform":
            uniform_prob = (1.0 - keep) / float(self.num_states)
            out[:, :, : self.config.num_classes] = p0 * keep + uniform_prob
            out[:, :, self.unk_id] = uniform_prob.expand(batch, num_nodes, 1).squeeze(-1)
        else:
            out[:, :, : self.config.num_classes] = p0 * keep
            out[:, :, self.unk_id] = (1.0 - keep).expand(batch, num_nodes, 1).squeeze(-1)
        out = out / out.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return out
