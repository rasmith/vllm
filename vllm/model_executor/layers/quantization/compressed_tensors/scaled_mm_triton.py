from typing import Optional, Type

import torch
import triton
import triton.language as tl


# This function handles some cases that can cause certain failure, e.g.
# a tensor that has shape = (72, 48) but stride = (5120, 1).  It can happen,
# for example by saving a tensor using torch.save() and then adjusting its
# size afterwards and then trying to use it.  Unfortunately,
# torch.is_contiguous() doesn't help since a transposed tensor doesn't return
# True, even though it can be stored contiguously in memory.
#
# There is a way to handle this case, which I learned about from here:
#
# https://github.com/pytorch/pytorch/blob/
# a874ec85e83cfe75e7238296022d53d7e20860df/aten/src/ATen/native/
# cuda/Blas.cpp#L58
#
# This doesn't happen very often fortunately, because the only solution is
# inefficient.
def prepare_matrix_for_triton(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    if not is_not_transpose and not is_transpose:
        return torch.clone(x, memory_format=torch.contiguous_format)
    return x


def get_hip_autotune_config():
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 1,
                'waves_per_eu': 0
            },
            num_warps=4,
            num_stages=0),
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8,
                'waves_per_eu': 0
            },
            num_warps=4,
            num_stages=0),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 16,
                'GROUP_SIZE_M': 1,
                'waves_per_eu': 0
            },
            num_warps=4,
            num_stages=0),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 16,
                'GROUP_SIZE_M': 4,
                'waves_per_eu': 0
            },
            num_warps=8,
            num_stages=0),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 1,
                'waves_per_eu': 9
            },
            num_warps=8,
            num_stages=0),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8,
                'waves_per_eu': 0
            },
            num_warps=4,
            num_stages=0),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 1,
                'waves_per_eu': 0
            },
            num_warps=4,
            num_stages=0),
    ]


def has_scalar(x):
    return x.shape[0] == 1 and x.shape[1] == 1


def get_scale_a_block_size(nargs):
    if nargs['has_scalar_a']:
        return 1
    return nargs['BLOCK_SIZE_M']


def get_scale_b_block_size(nargs):
    if nargs['has_scalar_b']:
        return 1
    return nargs['BLOCK_SIZE_N']


# @triton.autotune(
# configs=get_hip_autotune_config(),
# key=['M', 'N', 'K'],
# )
@triton.heuristics({
    "BLOCK_SIZE_SCALE_A": get_scale_a_block_size,
    "BLOCK_SIZE_SCALE_B": get_scale_b_block_size
})
@triton.jit
def scaled_mm_kernel(a_ptr, b_ptr, scale_a_ptr, scale_b_ptr, c_ptr, bias_ptr,
                     M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                     stride_c, stride_cm, stride_cn,
                     ACCUMULATOR_DTYPE: tl.constexpr, has_scalar_a,
                     has_scalar_b, BLOCK_SIZE_SCALE_A: tl.constexpr,
                     BLOCK_SIZE_SCALE_B: tl.constexpr,
                     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                     BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                     SPLIT_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # pid_m = pid // num_pid_n
    # pid_n = pid % num_pid_n

    accumulator_dtype = ACCUMULATOR_DTYPE
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                           dtype=accumulator_dtype)

    # NOTE: Some tensor inputs are so large, they will cause int32 overflow
    # so it is necessary to use tl.int64 for all the offsets, else SEGV will
    # eventually occur.

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = BLOCK_SIZE_K * pid_z + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    offsets_a = (stride_am * offsets_am[:, None] +
                 stride_ak * offsets_k[None, :])
    offsets_b = (stride_bk * offsets_k[:, None] +
                 stride_bn * offsets_bn[None, :])

    # NOTE: BLOCK_SIZE_SCALE_A could be 1 or BLOCK_SIZE_M, so need to create
    # appropriate offsets and masks for each case. Same goes for
    # BLOCK_SIZE_SCALE_B.
    offsets_scale_am = (tl.arange(0, BLOCK_SIZE_SCALE_A) +
                        (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M)
    masks_scale_am = offsets_scale_am < M

    offsets_scale_bn = (tl.arange(0, BLOCK_SIZE_SCALE_B) +
                        (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N)
    masks_scale_bn = offsets_scale_bn < N

    offsets_scale_a = (offsets_scale_am[:, None].to(tl.int64) +
                       tl.arange(0, 1)[None, :].to(tl.int64))
    offsets_scale_b = (offsets_scale_bn[:, None].to(tl.int64) +
                       tl.arange(0, 1)[None, :].to(tl.int64))

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    scale_a_ptrs = scale_a_ptr + offsets_scale_a
    scale_b_ptrs = scale_b_ptr + offsets_scale_b

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    # Apply scale at end.
    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_a = tl.load(scale_a_ptrs, masks_scale_a)
    # Need to broadcast to the appropriate size, if scale_a is already
    # (BLOCK_SIZE_M, 1) then it will broadcast to its own shape. Same goes
    # for scale_b below.
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))
    accumulator = scale_a * accumulator.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs, masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    accumulator = scale_b.T * accumulator.to(tl.float32)

    # Convert to output format.
    c = accumulator.to(c_ptr.type.element_ty)

    # Add bias, it's already in output format, so add it after conversion.
    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        c += bias

    # Save output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    c_ptrs = (c_ptr + pid_z * stride_c + stride_cm * offs_cm[:, None] +
              stride_cn * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


# input   - [M, K]
# weight - [K, N]
def scaled_mm_triton(input: torch.Tensor,
                     weight: torch.Tensor,
                     scale_a: torch.Tensor,
                     scale_b: torch.Tensor,
                     out_dtype: Type[torch.dtype],
                     bias: Optional[torch.Tensor] = None,
                     block_size_m: int = 32,
                     block_size_n: int = 32,
                     block_size_k: int = 32,
                     group_size_m: int = 8,
                     split_k: int = 1,
                     num_warps=1,
                     matrix_instr_nonkdim=16,
                     kpack=1) -> torch.Tensor:
    M, K = input.shape
    N = weight.shape[1]

    assert N > 0 and K > 0 and M > 0
    assert weight.shape[0] == K
    assert input.dtype == weight.dtype
    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape == torch.Size([1, 1]) or scale_a.shape == torch.Size(
        [M, 1])
    assert scale_b.shape == torch.Size([1, 1]) or scale_b.shape == torch.Size(
        [N, 1])
    assert torch.empty((1, 1), dtype=out_dtype).is_floating_point()
    assert bias is None or bias.is_floating_point()

    if M * N * split_k >= torch.iinfo(torch.int).max:
        split_k = 1

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(
            N, META['BLOCK_SIZE_N']),
        split_k,
    )

    result = torch.empty((split_k, M, N), dtype=out_dtype, device=input.device)

    input = prepare_matrix_for_triton(input)
    weight = prepare_matrix_for_triton(weight)

    accumulator_dtype = tl.float32 if input.is_floating_point() else tl.int32

    # A = input, B = weight, C = result
    # A = M x K, B = K x N, C = M x N
    scaled_mm_kernel[grid](
        input,
        weight,
        scale_a,
        scale_b,
        result,
        bias,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        result.stride(0),
        result.stride(1),
        result.stride(2),
        accumulator_dtype,
        has_scalar(scale_a),
        has_scalar(scale_b),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        SPLIT_K=split_k,
        num_warps=1,
        matrix_instr_nonkdim=16,
        kpack=1,
    )

    return result.sum(0)


def scaled_mm_torch(a: torch.Tensor,
                    b: torch.Tensor,
                    scale_a: torch.Tensor,
                    scale_b: torch.Tensor,
                    out_dtype: Type[torch.dtype],
                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    out = scale_a * out
    out = scale_b.T * out
    out = out.to(out_dtype)
    if bias is not None:
        out = out + bias

    return out


def main():

    which_test_fn = 0

    out_dtype = torch.float16

    golden_functions = [
        lambda a, b, scale_a, scale_b, bias: scaled_mm_torch(
            a, b, scale_a, scale_b, out_dtype, bias)
    ]
    golden_fn = golden_functions[which_test_fn]

    test_functions = [
        lambda a, b, scale_a, scale_b, bias: scaled_mm_triton(
            a, b, scale_a, scale_b, out_dtype, bias),
    ]
    test_fn = test_functions[which_test_fn]

    test_cases = [
        # M        K     N
        # Toy cases
        # (32, 32, 32),
        # (1, 17, 15),
        # (15, 49, 19),
        # (64, 96, 32),
        # (27, 14, 103),
        # (15, 179, 51),
        # (15, 1792, 512),
        # Realistic cases
        # (1, 17920, 5120),
        (1, 5120, 35840),
        (1, 5120, 5120),
        (1, 5120, 7680),
        (131072, 17920, 5120),
        (131072, 5120, 35840),
        (131072, 5120, 5120),
        (131072, 5120, 7680),
        (15, 17920, 5120),
        (15, 5120, 35840),
        (15, 5120, 5120),
        (15, 5120, 7680),
    ]

    use_bias = True

    use_scalar_scale_a = True
    use_scalar_scale_b = False

    comparisons = [torch.allclose]

    comparison = comparisons[which_test_fn]

    import time

    torch.manual_seed(0)

    test_out_dtype = torch.bfloat16

    for test_case in test_cases:
        M, K, N = test_case
        a = torch.randint(0, 127, (M, K), dtype=torch.int8, device='cuda')
        b = torch.randint(0, 127, (K, N), dtype=torch.int8, device='cuda')

        if use_scalar_scale_a:
            scale_a = torch.rand((1, 1), device='cuda')
        else:
            scale_a = torch.rand((M, 1), device='cuda')

        if use_scalar_scale_b:
            scale_b = torch.rand((1, 1), device='cuda')
        else:
            scale_b = torch.rand((1, 1), device='cuda')

        bias = None
        if use_bias:
            bias = torch.rand((N, ), device='cuda', dtype=out_dtype) * 10

        print("=" * 5 + f" Testing: mm_triton M={M}, K={K}, N={N}" + "=" * 5)

        # Compute and time test result.
        start = time.time()
        c_check = test_fn(a, b, scale_a, scale_b, bias)
        end = time.time()

        print(f"c_check time: {end - start}")
        print(f"c_check.dtype = {c_check.dtype}")

        # a_cpu = a.cpu()
        # b_cpu = b.cpu()
        # scale_a_cpu = scale_a.cpu()
        # scale_b_cpu = scale_b.cpu()
        # bias_cpu = None if bias is None else bias.cpu()

        # Compute and time golden result.
        # start = time.time()
        # c_actual = golden_fn(a_cpu, b_cpu, scale_a_cpu, scale_b_cpu, bias_cpu)
        # end = time.time()

        # print(f"c_actual time: {end - start}")
        # print(f"c_actual.dtype = {c_actual.dtype}")

        # # Drrruuumrolll...
        # comparison_result = comparison(c_check.cpu(),
        # c_actual,
        # rtol=1e-1,
        # atol=1e-1)
        # print(f"compare?: {comparison_result}")

        # if not comparison_result:
        # torch.set_printoptions(sci_mode=False)
        # print(f"c_check = {c_check}")
        # print(f"c_actual = {c_actual}")
        # break


if __name__ == "__main__":
    main()
