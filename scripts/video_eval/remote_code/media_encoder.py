from functools import partial
from typing import Any, Dict, List, Optional

import torch
from torch import nn


class BaseEncoder(nn.Module):
    def __init__(self, parent: nn.Module) -> None:
        super().__init__()
        self._parent = [parent]

    @property
    def parent(self) -> nn.Module:
        return self._parent[0]


class BasicImageEncoder(BaseEncoder):
    def __init__(
        self,
        parent: torch.nn.Module,
        start_tokens: Optional[str] = None,
        end_tokens: Optional[str] = "\n",
    ) -> None:
        super().__init__(parent)
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens

    def embed_tokens(self, tokens: Optional[str]) -> Optional[torch.Tensor]:
        if tokens is None:
            return None
        token_ids = self.parent.tokenizer(tokens).input_ids
        token_ids = torch.tensor(token_ids, device=self.parent.device)
        return self.parent.llm_model_embed_tokens(token_ids)

    def _process_features(
        self,
        features: torch.Tensor,
        start_token_embeds: Optional[torch.Tensor],
        end_token_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if start_token_embeds is not None:
            features = torch.cat([start_token_embeds, features], dim=0)
        if end_token_embeds is not None:
            features = torch.cat([features, end_token_embeds], dim=0)
        return features

    def forward(self, images: List[torch.Tensor], config: Dict[str, Any], device: torch.device) -> List[torch.Tensor]:
        images = torch.stack(images, dim=0)
        features = self.parent.encode_images(images, block_sizes=config.get("block_sizes"))
        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
        )
        return [process_features(f).to(device) for f in features]


class BasicVideoEncoder(BaseEncoder):
    def __init__(
        self,
        parent: torch.nn.Module,
        start_tokens: Optional[str] = None,
        end_tokens: Optional[str] = "\n",
    ) -> None:
        super().__init__(parent)
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens

    def embed_tokens(self, tokens: Optional[str]) -> Optional[torch.Tensor]:
        if tokens is None:
            return None
        token_ids = self.parent.tokenizer(tokens).input_ids
        token_ids = torch.tensor(token_ids, device=self.parent.device)
        return self.parent.llm_model_embed_tokens(token_ids)

    def _process_features(
        self,
        features: torch.Tensor,
        start_token_embeds: Optional[torch.Tensor],
        end_token_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if start_token_embeds is not None:
            start_embeds = torch.stack([start_token_embeds] * features.shape[0], dim=0)
            features = torch.cat([start_embeds, features], dim=1)
        if end_token_embeds is not None:
            end_embeds = torch.stack([end_token_embeds] * features.shape[0], dim=0)
            features = torch.cat([features, end_embeds], dim=1)
        return features.flatten(0, 1)

    def forward(self, videos: List[torch.Tensor], config: Dict[str, Any]) -> List[torch.Tensor]:
        num_frames = [video.shape[0] for video in videos]
        images = torch.cat(videos, dim=0)
        features = self.parent.encode_images(images)
        features = torch.split(features, num_frames)
        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
        )
        return [process_features(f) for f in features]

def pool(x: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    return x.view(x.shape[:dim] + (-1, size) + x.shape[dim + 1 :]).mean(dim + 1)

class TSPVideoEncoder(BasicVideoEncoder):
    def __init__(
        self,
        parent: torch.nn.Module,
        start_tokens: Optional[str] = None,
        end_tokens: Optional[str] = "\n",
        sep_tokens: Optional[str] = None,
    ) -> None:
        super().__init__(parent, start_tokens=start_tokens, end_tokens=end_tokens)
        self.pool_sizes = [[8, 1, 1]]
        self.sep_tokens = sep_tokens

    def _process_features(
        self,
        inputs: torch.Tensor,
        start_token_embeds: Optional[torch.Tensor],
        end_token_embeds: Optional[torch.Tensor],
        sep_token_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        nt, ns = inputs.shape[:2]
        nl = int(ns**0.5)
        outputs = []
        for pool_size in self.pool_sizes:
            features = inputs.view(nt, nl, nl, -1)
            for dim, p in enumerate(pool_size):
                features = pool(features, p, dim=dim)
            features = features.flatten(1, 2)
            features = super()._process_features(
                features,
                start_token_embeds=start_token_embeds,
                end_token_embeds=end_token_embeds,
            )
            if sep_token_embeds is not None:
                features = torch.cat([features, sep_token_embeds], dim=0)
            outputs.append(features)
        return torch.cat(outputs, dim=0)

    def forward(self, videos: List[torch.Tensor], config: Dict[str, Any]) -> List[torch.Tensor]:
        num_frames = [video.shape[0] for video in videos]
        images = torch.cat(videos, dim=0)
        features = self.parent.encode_images(images)
        features = torch.split(features, num_frames)
        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
            sep_token_embeds=self.embed_tokens(self.sep_tokens),
        )
        return [process_features(f) for f in features]
