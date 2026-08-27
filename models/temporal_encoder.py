"""Backward-compatible import for the video-only temporal encoder."""

from .video_temporal_encoder import VideoTemporalEncoder

TemporalEncoder = VideoTemporalEncoder

__all__ = ["VideoTemporalEncoder", "TemporalEncoder"]

