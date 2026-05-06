from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from hopfield_study.data import Example


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_LAYERS = (12, 16, 20)


@dataclass(frozen=True)
class FeatureBatch:
    nll: np.ndarray
    response_length: np.ndarray
    hopfield: np.ndarray


def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL, device: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model.to(device)
    return model, tokenizer, torch.device(device)


def extract_features(
    examples: list[Example],
    model,
    tokenizer,
    device: torch.device,
    *,
    batch_size: int = 1,
    max_length: int = 512,
    layers: Iterable[int] = DEFAULT_LAYERS,
    max_response_tokens: int = 32,
) -> FeatureBatch:
    all_nll: list[float] = []
    all_lengths: list[float] = []
    all_hopfield: list[np.ndarray] = []
    selected_layers = tuple(layers)

    for start in tqdm(range(0, len(examples), batch_size), desc="extract"):
        batch = examples[start : start + batch_size]
        full_texts = [ex.full_text for ex in batch]
        prompt_texts = [ex.prompt for ex in batch]
        response_positions = _response_positions(
            tokenizer,
            prompt_texts,
            full_texts,
            max_length=max_length,
            max_response_tokens=max_response_tokens,
            device=device,
        )
        collector = HopfieldEnergyCollector(model, selected_layers, response_positions)
        collector.install()
        try:
            encoding = tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            collector.remove()

        losses = _response_nll(
            outputs.logits.float(),
            input_ids,
            response_positions,
        )
        all_nll.extend(losses.tolist())
        all_lengths.extend([float(len(pos)) for pos in response_positions])
        all_hopfield.extend(collector.features())

    return FeatureBatch(
        nll=np.asarray(all_nll, dtype=np.float32),
        response_length=np.asarray(all_lengths, dtype=np.float32),
        hopfield=np.vstack(all_hopfield).astype(np.float32),
    )


class HopfieldEnergyCollector:
    """Collect compact Q/K Hopfield energy summaries from selected Qwen2 layers."""

    def __init__(self, model, layers: tuple[int, ...], response_positions: list[torch.Tensor]) -> None:
        self.model = model
        self.layers = layers
        self.response_positions = response_positions
        self.handles = []
        self.by_layer: dict[int, np.ndarray] = {}

    def install(self) -> None:
        for layer_idx in self.layers:
            module = self.model.model.layers[layer_idx].self_attn
            handle = module.register_forward_pre_hook(
                self._make_hook(layer_idx),
                with_kwargs=True,
            )
            self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def features(self) -> list[np.ndarray]:
        pieces = []
        for layer_idx in self.layers:
            if layer_idx not in self.by_layer:
                raise RuntimeError(f"Missing Hopfield features for layer {layer_idx}")
            pieces.append(self.by_layer[layer_idx])
        stacked = np.stack(pieces, axis=1)  # (batch, layers, 4)
        combined = np.concatenate(
            [
                stacked.mean(axis=1),
                stacked.std(axis=1),
            ],
            axis=1,
        )
        return [combined[i] for i in range(combined.shape[0])]

    def _make_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get(
                "position_embeddings",
                args[1] if len(args) > 1 else None,
            )
            if hidden_states is None or position_embeddings is None:
                raise RuntimeError("Could not access Qwen2 attention hook inputs")

            with torch.no_grad():
                self.by_layer[layer_idx] = _layer_energy_features(
                    module,
                    hidden_states,
                    position_embeddings,
                    self.response_positions,
                )

        return hook


def _response_positions(
    tokenizer,
    prompts: list[str],
    full_texts: list[str],
    *,
    max_length: int,
    max_response_tokens: int,
    device: torch.device,
) -> list[torch.Tensor]:
    prompt_enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    full_enc = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    prompt_lengths = prompt_enc["attention_mask"].sum(dim=1).tolist()
    full_lengths = full_enc["attention_mask"].sum(dim=1).tolist()

    positions: list[torch.Tensor] = []
    for prompt_len, full_len in zip(prompt_lengths, full_lengths, strict=True):
        start = min(int(prompt_len), max(int(full_len) - 1, 0))
        end_exclusive = int(full_len)
        if end_exclusive - start > 1:
            end_exclusive -= 1  # drop EOS/end-of-text-like final token
        pos = torch.arange(start, end_exclusive, dtype=torch.long, device=device)
        if pos.numel() == 0:
            fallback = max(int(full_len) - 1, 0)
            pos = torch.tensor([fallback], dtype=torch.long, device=device)
        if pos.numel() > max_response_tokens:
            pos = pos[-max_response_tokens:]
        positions.append(pos)
    return positions


def _response_nll(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_positions: list[torch.Tensor],
) -> np.ndarray:
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    losses: list[float] = []
    for i, positions in enumerate(response_positions):
        target_positions = positions[positions > 0] - 1
        if target_positions.numel() == 0:
            losses.append(float("nan"))
            continue
        token_log_probs = log_probs[i, target_positions, :]
        token_ids = target_ids[i, target_positions]
        nll = -token_log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        losses.append(float(nll.mean().item()))
    return np.asarray(losses, dtype=np.float32)


def _layer_energy_features(
    module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    response_positions: list[torch.Tensor],
) -> np.ndarray:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    query_states = query_states.float()
    key_states = key_states.float()
    batch_size, num_heads, seq_len, head_dim = query_states.shape
    num_kv_heads = key_states.shape[1]
    group_size = max(num_heads // num_kv_heads, 1)
    scaling = float(getattr(module, "scaling", head_dim**-0.5))
    inv_scaling = 1.0 / scaling
    seq_positions = torch.arange(seq_len, device=hidden_states.device)

    rows: list[np.ndarray] = []
    for batch_idx in range(batch_size):
        resp_pos = response_positions[batch_idx].to(hidden_states.device)
        energies_per_head: list[torch.Tensor] = []
        for head_idx in range(num_heads):
            kv_head_idx = min(head_idx // group_size, num_kv_heads - 1)
            q = query_states[batch_idx, head_idx, resp_pos, :]  # (response, dim)
            k = key_states[batch_idx, kv_head_idx, :, :]  # (seq, dim)
            scores = torch.matmul(q, k.transpose(0, 1)) * scaling
            causal = seq_positions.unsqueeze(0) <= resp_pos.unsqueeze(1)
            scores = scores.masked_fill(~causal, float("-inf"))
            energy = -inv_scaling * torch.logsumexp(scores, dim=-1)
            energies_per_head.append(energy)
        values = torch.stack(energies_per_head, dim=0).flatten()
        rows.append(
            np.asarray(
                [
                    float(values.mean().item()),
                    float(values.std(unbiased=False).item()),
                    float(values.max().item()),
                    float(torch.stack(energies_per_head, dim=0)[:, -1].mean().item()),
                ],
                dtype=np.float32,
            )
        )
    return np.vstack(rows)
