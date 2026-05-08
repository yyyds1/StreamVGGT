from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class TextTokenizerConfig:
    text_tokenizer_name: Optional[str] = None
    model_name_or_path: Optional[str] = None
    max_position_embeddings: int = 128
    text_embedding_shape: tuple[int, int] = (1, 768)


def _mean_pool_last_hidden_state(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    summed = torch.sum(hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class GemmaTextEmbedder(torch.nn.Module):
    """p2p-compatible EmbeddingGemma prompt encoder.

    Returns one mean-pooled text token with shape [B, 1, 1, 768], matching p2p's
    `text_embedding_shape: [1, 768]` convention.
    """

    def __init__(self, model_name_or_path: str = "google/embeddinggemma-300M", max_position_embeddings: int = 128):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.gemma_embedding = AutoModel.from_pretrained(model_name_or_path).eval()
            self.embed_dim = int(self.gemma_embedding.config.hidden_size)
        else:
            gemma_model = SentenceTransformer(model_name_or_path, device="cpu").eval()
            self.tokenizer = gemma_model.tokenizer
            self.gemma_embedding = gemma_model[0].auto_model.eval()
            self.embed_dim = int(gemma_model.get_sentence_embedding_dimension())
            del gemma_model

        self.n_text_tokens = 1
        self.max_position_embeddings = int(max_position_embeddings)

    def tokenize(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        texts = [text] if isinstance(text, str) else list(text)
        return self.tokenizer(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_position_embeddings,
        )

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim == 3:
            batch_size, n_steps, text_dim = input_ids.shape
        elif input_ids.ndim == 2:
            batch_size, text_dim = input_ids.shape
            n_steps = 1
        else:
            raise ValueError(f"Invalid input_ids shape: {tuple(input_ids.shape)}")

        encoder_device = next(self.gemma_embedding.parameters()).device
        input_ids = input_ids.reshape(-1, text_dim).to(encoder_device)
        attention_mask = attention_mask.reshape(-1, text_dim).to(encoder_device)
        hidden_state = self.gemma_embedding(input_ids, attention_mask).last_hidden_state
        sentence_embedding = _mean_pool_last_hidden_state(hidden_state, attention_mask).float()
        return sentence_embedding.reshape(batch_size, n_steps, self.n_text_tokens, self.embed_dim)


_SHARED_TEXT_EMBEDDER: GemmaTextEmbedder | None = None
_SHARED_TEXT_EMBEDDER_KEY: tuple[str, int] | None = None


def get_text_embedder(config) -> GemmaTextEmbedder | None:
    tokenizer_name = str(getattr(config, "text_tokenizer_name", "") or "").strip().lower()
    if tokenizer_name not in {"gemma", "embeddinggemma", "embeddinggemma-300m", "gemma-300m"}:
        return None

    model_name_or_path = str(
        getattr(config, "model_name_or_path", None)
        or getattr(config, "text_model_name_or_path", None)
        or "google/embeddinggemma-300M"
    )
    max_position_embeddings = int(getattr(config, "max_position_embeddings", 128))
    key = (model_name_or_path, max_position_embeddings)

    global _SHARED_TEXT_EMBEDDER, _SHARED_TEXT_EMBEDDER_KEY
    if _SHARED_TEXT_EMBEDDER is None or _SHARED_TEXT_EMBEDDER_KEY != key:
        _SHARED_TEXT_EMBEDDER = GemmaTextEmbedder(
            model_name_or_path=model_name_or_path,
            max_position_embeddings=max_position_embeddings,
        )
        _SHARED_TEXT_EMBEDDER_KEY = key
    return _SHARED_TEXT_EMBEDDER


def encode_prompt(text: str, config, device: torch.device | str | None = None, dtype: torch.dtype | None = None):
    embedder = get_text_embedder(config)
    if embedder is None:
        return None
    tokenized = embedder.tokenize(text or "")
    with torch.inference_mode():
        text_embedding = embedder(**tokenized)
    # [B, 1, 1, D] -> [B, 1, D]
    text_embedding = text_embedding.squeeze(1)
    if device is not None or dtype is not None:
        text_embedding = text_embedding.to(device=device, dtype=dtype)
    return text_embedding
