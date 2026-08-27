"""End-to-end feature-level VLA policy."""

import torch
from torch import nn

from models.action import ActionDiffusion
from models.multimodal_fusion import MultimodalFusion
from models.video_temporal_encoder import VideoTemporalEncoder


class VLAFeaturePolicy(nn.Module):
    """Temporal fusion followed by a diffusion action expert."""

    def __init__(self, video_encoder: VideoTemporalEncoder,
                 fusion: MultimodalFusion,
                 action_expert: ActionDiffusion, state_dim=14,
                 progress_dim=1):
        super().__init__()
        self.video_encoder = video_encoder
        self.fusion = fusion
        self.action_expert = action_expert
        latent_dim = fusion.queries.shape[-1]
        self.progress_head = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, progress_dim)
        )
        self.state_head = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, state_dim)
        )

    def encode_latents(self, image_features, state, demo_video_features, text_features):
        video_tokens = self.video_encoder(demo_video_features)
        return self.fusion(image_features, state, text_features, video_tokens)

    def loss(self, image, state, demo_video, text, actions):
        latents = self.encode_latents(image, state, demo_video, text)
        return self.action_expert.loss(actions, latents)

    @torch.no_grad()
    def sample_actions(self, image, state, demo_video, text, steps=None):
        latents = self.encode_latents(image, state, demo_video, text)
        return self.action_expert.sample(latents, steps=steps)

    def predict_progress(self, latents):
        return self.progress_head(latents.mean(dim=1))

    def predict_next_state(self, latents):
        return self.state_head(latents.mean(dim=1))

    def training_loss(self, batch, weights=None):
        """Compute imitation, demo-diversity, progress, and state losses."""
        weights = weights or {"action": 1.0, "demo": 0.1, "progress": 0.2, "state": 0.2}
        latents = self.encode_latents(
            batch["image_features"], batch["state"],
            batch["demo_video_features"], batch["text_features"],
        )
        action_loss = self.action_expert.loss(batch["actions"], latents)
        progress_loss = torch.zeros((), device=latents.device)
        state_loss = torch.zeros((), device=latents.device)
        if "progress_target" in batch:
            progress_loss = nn.functional.mse_loss(
                self.predict_progress(latents), batch["progress_target"]
            )
        if "next_state" in batch:
            state_loss = nn.functional.mse_loss(
                self.predict_next_state(latents), batch["next_state"]
            )

        demo_loss = torch.zeros((), device=latents.device)
        if "alt_demo_video_features" in batch:
            alt_latents = self.encode_latents(
                batch["image_features"], batch["state"],
                batch["alt_demo_video_features"], batch["text_features"],
            )
            distance = torch.linalg.vector_norm(
                latents.mean(dim=1) - alt_latents.mean(dim=1), dim=-1
            )
            demo_loss = torch.relu(1.0 - distance).mean()
        total = (
            weights["action"] * action_loss
            + weights["demo"] * demo_loss
            + weights["progress"] * progress_loss
            + weights["state"] * state_loss
        )
        return total, {
            "action": action_loss.detach(), "demo": demo_loss.detach(),
            "progress": progress_loss.detach(), "state": state_loss.detach(),
        }
