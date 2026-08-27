"""Train the feature-level VLA policy with multi-task objectives."""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from models.action import ActionDiffusion, ActionExpertConfig
from models.multimodal_fusion import MultimodalFusion
from models.video_temporal_encoder import VideoTemporalEncoder
from models.vla import VLAFeaturePolicy


class FeatureDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path, map_location="cpu")
        self.length = self.data["actions"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {key: value[index].float() for key, value in self.data.items()}


def build_model(data, config):
    action_dim = data["actions"].shape[-1]
    state_dim = data["state"].shape[-1]
    video_dim = data["demo_video_features"].shape[-1]
    image_dim = data["image_features"].shape[-1]
    text_dim = data["text_features"].shape[-1]
    d_model = config.get("d_model", 256)
    video = VideoTemporalEncoder(video_dim, d_model, config.get("video_tokens", 4), config.get("nhead", 8), config.get("video_layers", 4))
    fusion = MultimodalFusion(image_dim, state_dim, text_dim, d_model, d_model, config.get("latent_tokens", 8), config.get("nhead", 8), config.get("fusion_layers", 4))
    action = ActionDiffusion(ActionExpertConfig(action_dim=action_dim, horizon=data["actions"].shape[1], condition_dim=d_model, d_model=d_model, nhead=config.get("nhead", 8), num_layers=config.get("action_layers", 4)))
    return VLAFeaturePolicy(video, fusion, action, state_dim=state_dim, progress_dim=data.get("progress_target", torch.zeros(1, 1)).shape[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train/vla.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    dataset = FeatureDataset(config["data"])
    model = build_model(dataset.data, config)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    weights = {"action": config["action_loss_weight"], "demo": config["demo_loss_weight"], "progress": config["progress_loss_weight"], "state": config["state_loss_weight"]}
    model.train()
    for epoch in range(config["epochs"]):
        for batch in loader:
            optimizer.zero_grad()
            loss, metrics = model.training_loss(batch, weights)
            loss.backward()
            optimizer.step()
        values = " ".join(f"{key}={value.item():.4f}" for key, value in metrics.items())
        print(f"epoch={epoch + 1} total={loss.item():.4f} {values}")
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, output)


if __name__ == "__main__":
    main()
