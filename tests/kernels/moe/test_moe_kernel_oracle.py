# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoEKernelOracle ABC introduced in PR series for #37753.

This file contains a single canonical demonstration that
`UnquantizedMoEKernelOracle` methods delegate one-to-one to the
existing module-level functions in `oracle/unquantized.py`. Each method
on `UnquantizedMoEKernelOracle` follows the same `return module_fn(args)`
pattern, so verifying delegation for one method (`make_kernel`) gives
high confidence in the rest.
"""

from unittest import mock
from unittest.mock import patch

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle import UnquantizedMoEKernelOracle
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.platforms import current_platform


class TestUnquantizedDelegation:
    """UnquantizedMoEKernelOracle methods must delegate to the existing
    module-level functions; behaviour is bit-identical."""

    def test_make_kernel_delegates(self) -> None:
        quant_config = object()
        moe_config = object()
        experts_cls = TritonExperts
        sentinel_kernel = object()

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.unquantized."
            "make_unquantized_moe_kernel",
            return_value=sentinel_kernel,
        ) as mocked:
            out = UnquantizedMoEKernelOracle().make_kernel(
                quant_config,
                moe_config,
                UnquantizedMoeBackend.TRITON,
                experts_cls,
            )

        mocked.assert_called_once_with(
            quant_config,
            moe_config,
            UnquantizedMoeBackend.TRITON,
            experts_cls,
            None,  # routing_tables default
        )
        assert out is sentinel_kernel


# FP8 per-tensor weight alignment tests. AITER's CK 2-stage FP8 MoE GEMM needs
# hidden % 128 == 0 and intermediate % 256 == 0 (% 128 for SwiGLU), so
# ``maybe_roundup_sizes`` rounds misaligned sizes up and leaves aligned ones alone.


@pytest.mark.parametrize(
    ("activation", "hidden", "intermediate", "expected"),
    [
        ("SILU", 2048, 1536, (2048, 1536)),  # already aligned, untouched
        ("SILU", 4096, 3072, (4096, 3072)),
        ("SILU", 2048, 1408, (2048, 1536)),
        ("SILU", 2048, 192, (2048, 256)),
        ("GELU", 4096, 704, (4096, 768)),
        ("SWIGLUOAI", 2048, 1408, (2048, 1408)),  # SwiGLU only needs 128
        ("SWIGLUOAI", 2880, 2880, (2944, 2944)),
        ("SWIGLUOAI_UNINTERLEAVE", 2880, 1440, (2944, 1536)),
    ],
)
@pytest.mark.parametrize(
    "backend_name",
    [
        pytest.param(
            "AITER",
            marks=pytest.mark.skipif(
                not is_aiter_found_and_supported(),
                reason="Only test on ROCm with AITER installed and supported",
            ),
        ),
    ],
)
def test_maybe_round_up_hidden_size_and_intermediate_size(
    backend_name, activation, hidden, intermediate, expected
):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,
        maybe_round_up_hidden_size_and_intermediate_size,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8StaticTensorSym,
    )

    actual = maybe_round_up_hidden_size_and_intermediate_size(
        Fp8MoeBackend[backend_name],
        kFp8StaticTensorSym,
        MoEActivation[activation],
        hidden,
        intermediate,
    )
    assert actual == expected, f"Expected {expected}, got {actual}."


_requires_aiter = pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)


@pytest.mark.parametrize(
    ("backend_name", "weight_key_name", "expect_round_up"),
    [
        pytest.param("AITER", "kFp8StaticTensorSym", True, marks=_requires_aiter),
        ("TRITON", "kFp8StaticTensorSym", False),
        pytest.param("AITER", "kFp8StaticChannelSym", False, marks=_requires_aiter),
        pytest.param("AITER", "kFp8Static128BlockSym", False, marks=_requires_aiter),
    ],
)
@pytest.mark.parametrize(("hidden", "intermediate"), [(2880, 1408)])
@pytest.mark.parametrize("activation_name", ["SILU"])
def test_maybe_round_up_calls_fp8_moe_round_up_sizes_only_when_needed(
    backend_name,
    weight_key_name,
    expect_round_up,
    hidden,
    intermediate,
    activation_name,
):
    """The AITER round-up is applied only to AITER with per-tensor weights."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,
        maybe_round_up_hidden_size_and_intermediate_size,
    )
    from vllm.model_executor.layers.quantization.utils import quant_utils

    with mock.patch(
        "vllm.model_executor.layers.fused_moe.oracle.fp8.fp8_moe_round_up_sizes"
    ) as round_up:
        maybe_round_up_hidden_size_and_intermediate_size(
            Fp8MoeBackend[backend_name],
            getattr(quant_utils, weight_key_name),
            MoEActivation[activation_name],
            hidden,
            intermediate,
        )

    if expect_round_up:
        round_up.assert_called_once_with(
            MoEActivation[activation_name], hidden, intermediate
        )
    else:
        round_up.assert_not_called()


@pytest.mark.parametrize(
    ("hidden", "inter", "hidden_pad", "inter_pad"),
    [
        (2880, 1408, 2944, 1536),  # both padded
        (2048, 1408, 2048, 1536),  # intermediate only
        (2880, 1536, 2944, 1536),  # hidden only
        (704, 640, 768, 768),  # both padded, small
        (2048, 1536, 2048, 1536),  # aligned, nothing to zero
    ],
)
def test_maybe_zero_moe_weight_padding(hidden, inter, hidden_pad, inter_pad):
    """The padding of torch.empty-allocated FP8 expert weights is zeroed before
    the AITER shuffle, and the checkpoint slices are left untouched."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        maybe_zero_moe_weight_padding,
    )

    moe_config = make_dummy_moe_config(
        num_experts=2, hidden_dim=hidden, intermediate_size=inter
    )
    moe_config.hidden_dim = hidden_pad
    moe_config.intermediate_size_per_partition = inter_pad

    fp8_dtype = current_platform.fp8_dtype()
    w13 = torch.ones(2, 2 * inter_pad, hidden_pad).to(fp8_dtype)
    w2 = torch.ones(2, hidden_pad, inter_pad).to(fp8_dtype)

    maybe_zero_moe_weight_padding(moe_config, w13, w2)

    # fp8 has no comparison ops on ROCm; 0 and 1 convert to float exactly.
    w13, w2 = w13.float(), w2.float()

    # The padding is zero.
    assert (w13[:, inter:inter_pad] == 0).all(), "Expected zero w13 gate padding."
    assert (w13[:, inter_pad + inter :] == 0).all(), "Expected zero w13 up padding."
    assert (w13[:, :, hidden:] == 0).all(), "Expected zero w13 hidden padding."
    assert (w2[:, hidden:] == 0).all(), "Expected zero w2 hidden padding."
    assert (w2[:, :, inter:] == 0).all(), "Expected zero w2 intermediate padding."

    # The checkpoint values are still ones.
    assert (w13[:, :inter, :hidden] == 1).all(), "Expected w13 gate to be all ones."
    assert (w13[:, inter_pad : inter_pad + inter, :hidden] == 1).all(), (
        "Expected w13 up to be all ones."
    )
    assert (w2[:, :hidden, :inter] == 1).all(), "Expected w2 to be all ones."
