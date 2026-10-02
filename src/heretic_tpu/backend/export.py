# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Export of the abliterated model as a merged Hugging Face checkpoint or as a PEFT LoRA
adapter, and the upload of an export to the Hugging Face Hub.

Both exports run on the host with NumPy, from the checkpoint files and host copies of
the adapters, and never touch the device or the in-memory model, so the model does not
have to be restored afterwards.

The adapters are passed as the facade holds them:
{component: (A [L, M, r, d_in], B [L, M, d_out, r])}, float32, with M = 1 (see
"Parameter layout, components and LoRA" in docs/DESIGN.md). A module whose B is zero
was not modified by the trial.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

import numpy as np
from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open
from safetensors.numpy import save_file

from .errors import UnsupportedCheckpointError
from .weights import (
    CONFIG_FILE,
    GENERATION_CONFIG_FILE,
    INDEX_FILE,
    Checkpoint,
    TensorIndex,
    abliterable_components,
    component_shape,
    disk_key,
    module_path,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .arch import ArchConfig

# Component -> (A [L, M, r, d_in], B [L, M, d_out, r]), float32.
Adapters = Mapping[str, tuple[np.ndarray, np.ndarray]]

ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"

# Tokenizer and processor files that are copied from the source checkpoint when they
# exist at the pinned commit. tokenizer.save_pretrained then rewrites some of them.
TOKENIZER_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)

# Weight files of a merged export. Those in the target directory that are not part of
# the exported shard set are left over from an earlier export and are deleted.
WEIGHT_FILE_PATTERN = re.compile(r"model(-\d{5}-of-\d{5})?\.safetensors")

# Bytes per element of the safetensors dtypes that safe_open(framework="np") can read.
_ITEMSIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "C64": 8,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def save_merged(
    directory: str,
    ckpt: Checkpoint,
    tensors: TensorIndex,
    arch: ArchConfig,
    adapters: Adapters,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """
    Writes the checkpoint with the adapters merged into its weights.

    Only the shard set is exported, under the source's file names. A shard holding no
    modified weight (every module in it has B = 0) is copied byte for byte. The other
    shards are rewritten tensor by tensor in the source's order: each modified weight
    W becomes `(W.astype(float32) + B @ A).astype(storage dtype)`, computed from the
    weight on disk, and every other tensor and the `__metadata__` are kept unchanged.
    The safetensors index, config.json, generation_config.json and the tokenizer and
    processor files of the pinned commit are copied, and the tokenizer is then saved
    over them, which persists its pad token (and its padding side, as transformers
    does when the source tokenizer configuration sets one).

    Before writing, weight files and a safetensors index left in the directory by an
    earlier export that would not be overwritten are deleted, so that no stale file or
    index remains.

    Raises ValueError if the directory is, or lies inside, the checkpoint directory,
    because the source shards are read while the export is written.
    """

    _check_target(directory, ckpt.snapshot_dir)
    host_adapters, _ = _host_adapters(arch, adapters)
    deltas = _modified_weights(tensors, host_adapters)
    tokenizer_dir = _tokenizer_files_dir(ckpt)

    os.makedirs(directory, exist_ok=True)
    _remove_stale_weights(directory, ckpt)

    modified_shards = {tensors.tensors[key].shard for key in deltas}
    for shard in ckpt.shard_files:
        source = os.path.join(ckpt.snapshot_dir, shard)
        target = os.path.join(directory, shard)
        if shard in modified_shards:
            _write_merged_shard(source, target, deltas)
        else:
            _copy(source, target)

    metadata_files = [CONFIG_FILE, GENERATION_CONFIG_FILE]
    if ckpt.index is not None:
        metadata_files.append(INDEX_FILE)
    sources = [(ckpt.snapshot_dir, name) for name in metadata_files]
    sources += [(tokenizer_dir, name) for name in TOKENIZER_FILES]

    for source_dir, name in sources:
        source = os.path.join(source_dir, name)
        if os.path.isfile(source):
            _copy(source, os.path.join(directory, name))

    tokenizer.save_pretrained(directory)


def save_adapter(
    directory: str,
    ckpt: Checkpoint,
    arch: ArchConfig,
    adapters: Adapters,
    *,
    base_model: str,
    model_commit: str | None,
) -> None:
    """
    Writes the adapters as a PEFT LoRA adapter (adapter_config.json and
    adapter_model.safetensors), as upstream's `PeftModel.save_pretrained` would.

    Every module of every abliterable component is a target, with its transformers
    module path, and modules the trial did not modify are saved with B = 0.
    `base_model` (`settings.model`) is recorded as the base model, and the base
    revision is `model_commit` if set, otherwise the commit the checkpoint was resolved
    to (None for a local directory).
    """

    host_adapters, rank = _host_adapters(arch, adapters)

    target_modules = []
    adapter_weights = {}
    for component, (A, B) in host_adapters.items():
        for layer_index in range(arch.num_hidden_layers):
            for module_index in range(A.shape[1]):
                path = module_path(arch, component, layer_index, module_index)
                target_modules.append(path)
                # PEFT's keys, without the adapter name.
                key = f"base_model.model.{path}"
                adapter_weights[f"{key}.lora_A.weight"] = A[layer_index, module_index]
                adapter_weights[f"{key}.lora_B.weight"] = B[layer_index, module_index]

    config = {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        # The adapters are applied at full strength.
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank,
        "revision": model_commit or ckpt.sha,
        "target_modules": sorted(target_modules),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }

    os.makedirs(directory, exist_ok=True)
    save_file(
        adapter_weights,
        os.path.join(directory, ADAPTER_WEIGHTS_FILE),
        metadata={"format": "pt"},
    )
    # Formatted as PEFT writes it.
    with open(
        os.path.join(directory, ADAPTER_CONFIG_FILE), "w", encoding="utf-8"
    ) as file:
        file.write(json.dumps(config, indent=2, sort_keys=True))


def push_to_hub(
    repo_id: str,
    export: Callable[[str], None],
    *,
    private: bool,
    token: str,
) -> None:
    """
    Uploads an export to a Hugging Face Hub model repository.

    `export(directory)` writes the export (for example a partial application of
    `save_merged` or `save_adapter`) into a fresh temporary directory. The repository
    is then created with the requested visibility, unless it exists already, and the
    directory is uploaded in one commit, so exactly the exported files are uploaded.
    """

    with tempfile.TemporaryDirectory(prefix="heretic-tpu-export-") as directory:
        export(directory)

        api = HfApi(token=token)
        repo_id = api.create_repo(repo_id, private=private, exist_ok=True).repo_id
        api.upload_folder(
            repo_id=repo_id,
            folder_path=directory,
            commit_message="Upload model",
        )


def _check_target(directory: str, source_dir: str) -> None:
    """Refuses a target directory that is, or lies inside, the source directory."""

    # Symlinks are resolved first, and each ancestor is compared by identity, which
    # also catches aliases such as differently cased paths.
    path = os.path.realpath(directory)
    while True:
        if os.path.exists(path) and os.path.samefile(path, source_dir):
            raise ValueError(
                f"Cannot export into {directory}, because it is or lies inside the "
                f"directory of the source checkpoint ({source_dir}), whose files are "
                "read during the export. Please choose another directory."
            )

        parent = os.path.dirname(path)
        if parent == path:
            return
        path = parent


def _host_adapters(
    arch: ArchConfig,
    adapters: Adapters,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int]:
    """
    The adapters as float32 NumPy arrays, in sorted component order, and their rank.
    Raises ValueError unless they match the architecture.
    """

    components = abliterable_components(arch)
    if sorted(adapters) != components:
        raise ValueError(
            f"Expected adapters for the components {components}, "
            f"got {sorted(adapters)}."
        )

    host_adapters = {}
    for component in components:
        # Host copies of device arrays can have the device's layout (for example,
        # from a TPU, B with its rank axis major), and safetensors writes the buffer
        # of a NumPy array without regard to its strides.
        A, B = (
            np.ascontiguousarray(array, dtype=np.float32)
            for array in adapters[component]
        )
        host_adapters[component] = (A, B)

    # Every component must have the rank of the first one.
    first_A = host_adapters[components[0]][0]
    rank = first_A.shape[2] if first_A.ndim == 4 else 0

    for component, (A, B) in host_adapters.items():
        d_out, d_in = component_shape(arch, component)
        shape_A = (arch.num_hidden_layers, 1, rank, d_in)
        shape_B = (arch.num_hidden_layers, 1, d_out, rank)
        if A.shape != shape_A or B.shape != shape_B:
            raise ValueError(
                f"The adapters of {component} have the shapes {A.shape} and "
                f"{B.shape}, but {shape_A} and {shape_B} were expected."
            )

    return host_adapters, rank


def _modified_weights(
    tensors: TensorIndex,
    adapters: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Safetensors key -> (B, A) of every module the adapters modify (non-zero B)."""

    deltas = {}

    for component, (A, B) in adapters.items():
        for layer_index in range(A.shape[0]):
            for module_index in range(A.shape[1]):
                key, info = disk_key(tensors, component, layer_index, module_index)
                shape = (B.shape[2], A.shape[3])
                if info.shape != shape:
                    raise ValueError(
                        f"Tensor '{key}' has the shape {info.shape}, but the adapters "
                        f"of {component} are for {shape}."
                    )

                # Modules with B = 0 are left untouched (merging a zero delta could
                # still change a weight's bytes, for example -0.0 to +0.0).
                if np.any(B[layer_index, module_index]):
                    deltas[key] = (
                        B[layer_index, module_index],
                        A[layer_index, module_index],
                    )

    return deltas


def _tokenizer_files_dir(ckpt: Checkpoint) -> str:
    """
    The directory holding the source's tokenizer and processor files, which are
    downloaded at the pinned commit for a Hub model (only those that exist there).
    """

    if ckpt.sha is None:
        return ckpt.snapshot_dir

    return snapshot_download(
        ckpt.model,
        revision=ckpt.sha,
        allow_patterns=list(TOKENIZER_FILES),
    )


def _remove_stale_weights(directory: str, ckpt: Checkpoint) -> None:
    for name in os.listdir(directory):
        if WEIGHT_FILE_PATTERN.fullmatch(name) and name not in ckpt.shard_files:
            os.remove(os.path.join(directory, name))

    # An index left by an earlier sharded export would point at deleted shards.
    index_path = os.path.join(directory, INDEX_FILE)
    if ckpt.index is None and os.path.lexists(index_path):
        os.remove(index_path)


@contextmanager
def _replacing(path: str) -> Iterator[str]:
    """
    Yields a temporary path next to `path`, which is moved into place once it has been
    written. An interrupted export therefore never leaves a truncated file, and a
    symlink at `path` (for example into a Hugging Face cache) is replaced rather than
    written through.
    """

    temporary = f"{path}.tmp"
    try:
        yield temporary
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.remove(temporary)
        raise


def _copy(source: str, target: str) -> None:
    # Follows symlinks, which the shards and other files of a Hugging Face cache
    # snapshot are.
    with _replacing(target) as temporary:
        shutil.copyfile(source, temporary)


def _write_merged_shard(
    source: str,
    target: str,
    deltas: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Writes a copy of a shard with the deltas of the modified weights merged in.

    Merging keeps every tensor's shape and dtype, so the tensors are written in the
    source's order with the same offsets, under a header serialised as safetensors
    serialises it. Only one tensor is held in memory at a time.
    """

    with safe_open(source, framework="np") as file:
        keys = file.offset_keys()

        header: dict[str, Any] = {}
        metadata = file.metadata()
        if metadata is not None:
            header["__metadata__"] = metadata

        offset = 0
        for key in keys:
            tensor = file.get_slice(key)
            dtype = tensor.get_dtype()
            shape = tensor.get_shape()
            if dtype not in _ITEMSIZES:
                raise UnsupportedCheckpointError(
                    f"Cannot rewrite {os.path.basename(source)}, because tensor "
                    f"'{key}' is stored as {dtype}, which cannot be read with NumPy."
                )

            size = math.prod(shape) * _ITEMSIZES[dtype]
            header[key] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + size],
            }
            offset += size

        header_bytes = json.dumps(
            header,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        # Like safetensors, pad the header with spaces so that the data is 8-byte aligned.
        header_bytes += b" " * (-len(header_bytes) % 8)

        with _replacing(target) as temporary, open(temporary, "wb") as output:
            output.write(len(header_bytes).to_bytes(8, "little"))
            output.write(header_bytes)

            for key in keys:
                array = file.get_tensor(key)
                if key in deltas:
                    B, A = deltas[key]
                    array = (array.astype(np.float32) + B @ A).astype(array.dtype)
                # The raw bytes, without a copy (also for dtypes such as bfloat16
                # that the buffer protocol does not support).
                output.write(np.ascontiguousarray(array).reshape(-1).view(np.uint8))
