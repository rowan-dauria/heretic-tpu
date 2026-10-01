# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""JAX inference engine for heretic-tpu. Must not import from the rest of heretic_tpu."""

import jax

# On TPU, a float32 matmul at the default precision is a single bfloat16 pass.
# Upstream computes float32 matmuls in full IEEE float32, and the abliteration and
# ARA maths depend on it (see "Matmul precision" in docs/DESIGN.md). This only affects
# dots with float32 operands, so bfloat16 decoder matmuls are unchanged.
jax.config.update("jax_default_matmul_precision", "highest")
