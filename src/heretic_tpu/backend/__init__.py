# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""JAX inference engine for heretic-tpu. Must not import from the rest of heretic_tpu."""

import jax

# On TPU, a float32 matmul at the default precision is a single bfloat16 pass.
# Upstream computes float32 matmuls in full IEEE float32, and the abliteration and
# ARA maths depend on it (see "Matmul precision" in docs/DESIGN.md). An ordinary dot
# with bfloat16 operands computes the same at HIGHEST as at DEFAULT (exact products,
# float32 accumulation), so bfloat16 decoder matmuls are unchanged. A ragged dot is the
# exception: its TPU kernel rejects bfloat16 operands at HIGHEST, so the
# mixture-of-experts matmuls (`layers._grouped_linear`) pass their precision.
jax.config.update("jax_default_matmul_precision", "highest")
