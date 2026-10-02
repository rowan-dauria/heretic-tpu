# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Architecture description of a supported text model.

`ArchConfig` is built from the configuration as transformers resolves it
(`AutoConfig.from_pretrained`), never from raw config.json keys: released checkpoints
rely on config-class defaults, and transformers 5 rewrites RoPE and sliding-window
fields during config post-init. `check_config` rejects unsupported checkpoints before
any weight shard is downloaded.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from transformers import PretrainedConfig

from .errors import UnsupportedArchitectureError, UnsupportedCheckpointError

if TYPE_CHECKING:
    from .weights import TensorIndex

# Supported text model types (`config.get_text_config().model_type`), each mapped to
# the top-level model types that may wrap it.
SUPPORTED_ARCHITECTURES: dict[str, tuple[str, ...]] = {
    "gemma3_text": ("gemma3", "gemma3_text"),
    "llama": ("llama",),
    "ministral3": ("ministral3", "mistral3"),
    "mistral": ("mistral", "mistral3"),
    "mixtral": ("mixtral",),
    "phi3": ("phi3",),
    "qwen2": ("qwen2",),
    "qwen3": ("qwen3",),
    "qwen3_moe": ("qwen3_moe",),
}

# Planned architectures, rejected until they pass the parity tests.
PLANNED_ARCHITECTURES: tuple[str, ...] = ()

# Rejected because their chat templates reject the system role,
# so upstream Heretic cannot run them either.
SYSTEM_ROLE_ARCHITECTURES = ("gemma", "gemma2")

ROPE_TYPES = ("default", "linear", "dynamic", "llama3", "yarn", "longrope")

ACTIVATIONS = ("silu", "gelu", "gelu_pytorch_tanh")

LAYER_TYPES = ("full_attention", "sliding_attention")

# The window of layers without sliding-window attention.
NO_WINDOW = 2**30


@dataclass(frozen=True)
class RopeSpec:
    """RoPE of one layer, as transformers initialises it."""

    rope_type: str

    # Number of rotated dimensions of each head (the first `rot` dimensions).
    rot: int

    # Sorted items of the layer's resolved `rope_parameters`, with lists as tuples.
    params: tuple[tuple[str, Hashable], ...]

    def get(self, key: str, default: Any = None) -> Any:
        return dict(self.params).get(key, default)


@dataclass(frozen=True)
class ArchConfig:
    """
    Everything the engine needs to know about a model's architecture.

    All fields are hashable, and the default `==` decides whether an existing engine
    (and its compiled executables) can be reused for another checkpoint.
    """

    # Text model type (`config.get_text_config().model_type`).
    model_type: str

    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    max_position_embeddings: int
    head_dim: int

    # One of `ACTIVATIONS`.
    activation: str

    attn_logit_softcapping: float | None

    # As the transformers model class applies it (never for the Gemma 3 wrapper).
    final_logit_softcapping: float | None

    query_pre_attn_scalar: float | None
    norm_topk_prob: bool | None
    num_experts_per_tok: int | None

    # Mixture-of-experts sizes; None for dense models.
    num_experts: int | None
    moe_intermediate_size: int | None

    # Projections that have a bias, as keys of the per-layer parameters
    # (a subset of "q", "k", "v", "o_proj", "gate", "up" and "down_proj"), sorted.
    biases: tuple[str, ...]

    # True when the checkpoint has no output projection and the embedding
    # matrix serves as the head. It changes the parameter pytree.
    tied_head: bool

    # True when config.json has a "vision_config" key (whatever its value).
    # Upstream then loads the model with AutoModelForImageTextToText,
    # which places the text layers under "model.language_model".
    multimodal: bool

    # Per layer: "full_attention" or "sliding_attention", and the attention window
    # (`NO_WINDOW` for full attention).
    layer_types: tuple[str, ...]
    windows: tuple[int, ...]

    # Per layer.
    rope: tuple[RopeSpec, ...]

    # True when the RoPE parameters are given per layer type (Gemma 3).
    # Transformers' YaRN then always truncates the correction range.
    rope_per_layer_type: bool

    # "long_factor" (longrope) or "raise" (dynamic), with the sequence length above
    # which it applies; None otherwise. See "RoPE switching" in docs/DESIGN.md.
    rope_switch: str | None
    rope_switch_len: int | None

    @property
    def rot(self) -> int:
        """Number of rotated dimensions of each head (equal for all layers)."""
        return self.rope[0].rot

    @property
    def attention_scale(self) -> float:
        """The factor transformers multiplies attention scores by."""
        if self.query_pre_attn_scalar is not None:
            return self.query_pre_attn_scalar**-0.5
        return self.head_dim**-0.5

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None

    @property
    def has_qk_norm(self) -> bool:
        """Whether queries and keys are RMS-normalised per head before RoPE."""
        return self.model_type in ("gemma3_text", "qwen3", "qwen3_moe")

    @property
    def has_sandwich_norms(self) -> bool:
        """
        Whether the attention and MLP outputs are normalised before the residual
        addition (Gemma 3). Gemma 3 also uses `(1 + w)` RMSNorm weights and scales
        the embedding output by `sqrt(hidden_size)`.
        """
        return self.model_type == "gemma3_text"

    @classmethod
    def from_hf(
        cls,
        config: PretrainedConfig,
        raw_config: dict[str, Any],
        tensors: TensorIndex,
    ) -> ArchConfig:
        """
        Builds the architecture description from the resolved configuration,
        the parsed config.json (only used for `multimodal`) and the tensor index
        of the checkpoint (bias presence and head tying).
        """

        # Deferred to avoid an import cycle (weights.py works with ArchConfig).
        from .weights import detect_biases

        check_config(config)

        text_config = config.get_text_config()

        head_dim = _head_dim(text_config)
        layer_types, windows = _attention_layout(text_config)
        layer_rope_parameters, rope_per_layer_type = _layer_rope_parameters(text_config)
        rope = tuple(
            RopeSpec(
                rope_type=parameters["rope_type"],
                rot=_rot(parameters, head_dim),
                params=tuple(
                    sorted((key, _freeze(value)) for key, value in parameters.items())
                ),
            )
            for parameters in layer_rope_parameters
        )

        num_experts = None
        moe_intermediate_size = None
        if text_config.model_type == "qwen3_moe":
            num_experts = text_config.num_experts
            moe_intermediate_size = text_config.moe_intermediate_size
        elif text_config.model_type == "mixtral":
            num_experts = text_config.num_local_experts
            moe_intermediate_size = text_config.intermediate_size

        rope_types = {spec.rope_type for spec in rope}
        if "longrope" in rope_types:
            rope_switch = "long_factor"
            rope_switch_len = _original_max_position_embeddings(
                layer_rope_parameters[0], text_config
            )
        elif "dynamic" in rope_types:
            # Transformers recomputes the frequencies once the sequence length exceeds
            # max_position_embeddings (`max_seq_len_cached` after initialisation).
            rope_switch = "raise"
            rope_switch_len = text_config.max_position_embeddings
        else:
            rope_switch = None
            rope_switch_len = None

        return cls(
            model_type=text_config.model_type,
            num_hidden_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            vocab_size=text_config.vocab_size,
            rms_norm_eps=text_config.rms_norm_eps,
            max_position_embeddings=text_config.max_position_embeddings,
            head_dim=head_dim,
            activation=_activation(text_config),
            attn_logit_softcapping=getattr(text_config, "attn_logit_softcapping", None),
            final_logit_softcapping=_final_logit_softcapping(config, text_config),
            query_pre_attn_scalar=getattr(text_config, "query_pre_attn_scalar", None),
            norm_topk_prob=getattr(text_config, "norm_topk_prob", None),
            num_experts_per_tok=getattr(text_config, "num_experts_per_tok", None),
            num_experts=num_experts,
            moe_intermediate_size=moe_intermediate_size,
            biases=detect_biases(tensors, text_config.model_type),
            tied_head=tensors.head_key is None,
            multimodal="vision_config" in raw_config,
            layer_types=layer_types,
            windows=windows,
            rope=rope,
            rope_per_layer_type=rope_per_layer_type,
            rope_switch=rope_switch,
            rope_switch_len=rope_switch_len,
        )


def check_config(config: PretrainedConfig) -> None:
    """
    Rejects unsupported checkpoints using the resolved configuration alone,
    so that no weight shard needs to be downloaded first.
    """

    text_config = config.get_text_config()

    # 1. Pre-quantised checkpoints, before anything else.
    for candidate in (config, text_config):
        quantization_config = getattr(candidate, "quantization_config", None)
        if quantization_config is not None:
            if isinstance(quantization_config, dict):
                quant_method = quantization_config.get("quant_method")
            else:
                quant_method = getattr(quantization_config, "quant_method", None)
            raise UnsupportedCheckpointError(
                f"The checkpoint is pre-quantised (quant_method: {quant_method}), "
                "and quantised checkpoints are not supported. "
                "Please use the unquantised repository of the model instead."
            )

    # 2. Dispatch on the (wrapper, text model) pair.
    _check_model_type(config.model_type, text_config.model_type)

    # 3. Structural checks.
    _check_structure(text_config)


def _check_model_type(model_type: str, text_model_type: str) -> None:
    wrappers = SUPPORTED_ARCHITECTURES.get(text_model_type)
    if wrappers is not None and model_type in wrappers:
        return

    if text_model_type in SYSTEM_ROLE_ARCHITECTURES:
        reason = "its chat template rejects the system role"
    elif text_model_type in PLANNED_ARCHITECTURES:
        reason = "support for it is planned but not available yet"
    elif wrappers is not None:
        reason = f"it is not supported inside a '{model_type}' model"
    else:
        reason = "it is not supported"

    raise UnsupportedArchitectureError(
        f"Model type '{text_model_type}' cannot be used because {reason}. "
        f"Supported model types: {', '.join(sorted(SUPPORTED_ARCHITECTURES))}."
    )


def _check_structure(text_config: PretrainedConfig) -> None:
    model_type = text_config.model_type

    # Per-layer overrides (transformers' heterogeneous configs) would make the layer
    # stack heterogeneous, so it could not be scanned. This must be checked first,
    # because reading an overridden attribute from such a config raises.
    if getattr(text_config, "is_heterogeneous", False):
        raise UnsupportedArchitectureError(
            f"The '{model_type}' configuration has per-layer overrides, "
            "which are not supported."
        )

    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None:
        unknown = sorted(set(layer_types) - set(LAYER_TYPES))
        if unknown:
            raise UnsupportedArchitectureError(
                f"Layer types {unknown} are not supported "
                f"(supported: {', '.join(LAYER_TYPES)})."
            )
        if len(layer_types) != text_config.num_hidden_layers:
            raise UnsupportedArchitectureError(
                f"The configuration lists {len(layer_types)} layer types "
                f"for {text_config.num_hidden_layers} layers."
            )

    # Dense layers among the MoE layers would make the layer stack heterogeneous.
    if model_type == "qwen3_moe" and (
        text_config.mlp_only_layers
        or text_config.decoder_sparse_step != 1
        or text_config.num_experts == 0
    ):
        raise UnsupportedArchitectureError(
            "Qwen3-MoE models with dense layers (mlp_only_layers, decoder_sparse_step "
            "or num_experts) are not supported."
        )

    if getattr(text_config, "use_bidirectional_attention", False):
        raise UnsupportedArchitectureError(
            "Bidirectional attention (use_bidirectional_attention) is not supported."
        )

    activation = _activation(text_config)
    if activation not in ACTIVATIONS:
        raise UnsupportedArchitectureError(
            f"Activation function '{activation}' is not supported "
            f"(supported: {', '.join(ACTIVATIONS)})."
        )

    head_dim = _head_dim(text_config)
    layer_rope_parameters, _ = _layer_rope_parameters(text_config)

    rope_types = []
    for parameters in layer_rope_parameters:
        rope_type = parameters.get("rope_type") if parameters else None
        if rope_type not in ROPE_TYPES:
            raise UnsupportedArchitectureError(
                f"RoPE type '{rope_type}' is not supported "
                f"(supported: {', '.join(ROPE_TYPES)})."
            )
        rope_types.append(rope_type)

        # Only the Phi-3 modelling code rotates part of each head. The others ignore
        # partial_rotary_factor (default RoPE) or fail with it (other RoPE types).
        if model_type != "phi3" and parameters.get("partial_rotary_factor", 1.0) != 1:
            raise UnsupportedArchitectureError(
                f"Partial rotary embeddings are not supported for '{model_type}'."
            )

    for rope_type in ("longrope", "dynamic"):
        if rope_type in rope_types and any(t != rope_type for t in rope_types):
            raise UnsupportedArchitectureError(
                f"RoPE type '{rope_type}' on some layer types but not all "
                "is not supported."
            )

    if len({_rot(parameters, head_dim) for parameters in layer_rope_parameters}) > 1:
        raise UnsupportedArchitectureError(
            "Layers with different numbers of rotary dimensions are not supported."
        )

    if "longrope" in rope_types:
        switch_lengths = {
            _original_max_position_embeddings(parameters, text_config)
            for parameters in layer_rope_parameters
        }
        if len(switch_lengths) > 1:
            raise UnsupportedArchitectureError(
                "longrope with a different original_max_position_embeddings "
                "per layer type is not supported."
            )


def _head_dim(text_config: PretrainedConfig) -> int:
    return (
        getattr(text_config, "head_dim", None)
        or text_config.hidden_size // text_config.num_attention_heads
    )


def _activation(text_config: PretrainedConfig) -> str:
    return getattr(text_config, "hidden_activation", None) or text_config.hidden_act


def _final_logit_softcapping(
    config: PretrainedConfig,
    text_config: PretrainedConfig,
) -> float | None:
    # Gemma3ForConditionalGeneration (the multimodal wrapper) never applies the final
    # soft-capping of its text config; only Gemma3ForCausalLM does.
    if config.model_type == "gemma3":
        return None
    return getattr(text_config, "final_logit_softcapping", None)


def _attention_layout(
    text_config: PretrainedConfig,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    sliding_window = getattr(text_config, "sliding_window", None)

    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is None:
        # Mistral, Phi-3 and Mixtral apply the window (if any) to every layer.
        layer_type = (
            "sliding_attention" if sliding_window is not None else "full_attention"
        )
        layer_types = [layer_type] * text_config.num_hidden_layers

    # A sliding layer without a window attends to everything, as in transformers.
    windows = [
        sliding_window
        if layer_type == "sliding_attention" and sliding_window is not None
        else NO_WINDOW
        for layer_type in layer_types
    ]

    return tuple(layer_types), tuple(windows)


def _layer_rope_parameters(
    text_config: PretrainedConfig,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Returns the resolved RoPE parameters of each layer, and whether they are given
    per layer type (as for Gemma 3) rather than once for all layers.
    """

    rope_parameters = getattr(text_config, "rope_parameters", None) or {}
    layer_types = getattr(text_config, "layer_types", None)

    if layer_types is not None and all(
        layer_type in rope_parameters for layer_type in layer_types
    ):
        return [rope_parameters[layer_type] for layer_type in layer_types], True

    return [rope_parameters] * text_config.num_hidden_layers, False


def _rot(parameters: dict[str, Any], head_dim: int) -> int:
    return int(head_dim * parameters.get("partial_rotary_factor", 1.0))


def _original_max_position_embeddings(
    parameters: dict[str, Any],
    text_config: PretrainedConfig,
) -> int:
    value = parameters.get("original_max_position_embeddings")
    if value is None:
        value = text_config.original_max_position_embeddings
    return value


def _freeze(value: Any) -> Hashable:
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value
