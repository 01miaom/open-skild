"""End-to-end feature-level VLA policy."""

import torch
from torch import nn

from models.action import ActionDiffusion
from models.temporal_encoder import TemporalEncoder


class VLAFeaturePolicy(nn.Module):
    """Temporal fusion followed by a diffusion action expert."""

    def __init__(self, temporal_encoder: TemporalEncoder,
                 action_expert: ActionDiffusion):
        super().__init__()
        self.temporal_encoder = temporal_encoder
        self.action_expert = action_expert

    def encode_latents(self, image, state, demo_video, text):
        return self.temporal_encoder(image, state, demo_video, text)

    def loss(self, image, state, demo_video, text, actions):
        latents = self.encode_latents(image, state, demo_video, text)
        return self.action_expert.loss(actions, latents)

    @torch.no_grad()
    def sample_actions(self, image, state, demo_video, text, steps=None):
        latents = self.encode_latents(image, state, demo_video, text)
        return self.action_expert.sample(latents, steps=steps)

