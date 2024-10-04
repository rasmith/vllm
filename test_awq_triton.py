"""Tests for the AWQ Triton kernel.

Run `pytest tests/kernels/test_awq_triton.py`.
"""
# import pytest
import torch

from vllm.model_executor.layers.quantization.awq_triton import (
    AWQ_TRITON_SUPPORTED_GROUP_SIZES, awq_dequantize_triton, awq_gemm_triton)
from vllm.utils import seed_everything

device = "cuda"


def reverse_awq_order(t: torch.Tensor):
    bits = 4
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_order_tensor = torch.arange(
        t.shape[-1],
        dtype=torch.int32,
        device=t.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    t = t[:, reverse_order_tensor] & 0xF
    return t


# qweights - [R     , C // 8], int32
# scales   - [R // G, C     ], float16
# zeros    - [R // G, C // 8], int32
def awq_dequantize_torch(qweight: torch.Tensor, scales: torch.Tensor,
                         qzeros: torch.Tensor,
                         group_size: int) -> torch.Tensor:

    if group_size == -1:
        group_size = qweight.shape[0]

    bits = 4
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    iweights = torch.bitwise_right_shift(qweight[:, :, None],
                                         shifts[None, None, :]).to(torch.int8)

    iweights = iweights.view(iweights.shape[0], -1)

    zeros = torch.bitwise_right_shift(qzeros[:, :, None],
                                      shifts[None, None, :]).to(torch.int8)
    zeros = zeros.view(qzeros.shape[0], -1)
    zeros = reverse_awq_order(zeros)

    iweights = reverse_awq_order(iweights)

    qv = iweights
    zv = zeros.repeat_interleave(group_size, dim=0)
    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)

    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = zeros.repeat_interleave(group_size, dim=0)
    print(f"iweights.shape = {iweights.shape}, zeros.shape = {zeros.shape}"
          f", scales.shape = {scales.shape}")
    return (iweights - zeros) * scales, scales, zeros, iweights, qv, zv


# qweights - [R     , C // 8], int32
# scales   - [R // G, C     ], float16
# zeros    - [R // G, C // 8], int32
def test_dequantize(qweight_rows, qweight_cols, group_size):

    if group_size == -1:
        group_size = qweight_rows

    qweight_dtype = torch.int32
    scales_rows = qweight_rows // group_size
    scales_cols = qweight_cols * 8
    scales_dtype = torch.float16
    zeros_rows = scales_rows
    zeros_cols = qweight_cols
    zeros_dtype = torch.int32

    seed_everything(0)

    qweight = torch.randint(0,
                            torch.iinfo(torch.int32).max,
                            (qweight_rows, qweight_cols),
                            dtype=qweight_dtype,
                            device=device)
    scales = torch.rand(scales_rows,
                        scales_cols,
                        dtype=scales_dtype,
                        device=device)
    zeros = torch.randint(0,
                          torch.iinfo(torch.int32).max,
                          (zeros_rows, zeros_cols),
                          dtype=zeros_dtype,
                          device=device)

    iweights_triton = awq_dequantize_triton(qweight, scales, zeros)

    assert (not torch.any(torch.isinf(iweights_triton))
            and not torch.any(torch.isnan(iweights_triton)))

    iweights_torch = awq_dequantize_torch(qweight, scales, zeros, group_size)

    torch.testing.assert_close(iweights_triton, iweights_torch)


# input   - [N, K]
# qweight - [K, M // 8]
# qzeros  - [K // G, M // 8]
# scales  - [K // G, M]
def test_gemm(N, K, M, splitK, group_size):

    import time
    print(f"test_gemm")
    if group_size == -1:
        group_size = K

    split_k_iters = splitK

    input_rows = N
    input_cols = K
    input_dtype = torch.float16
    qweight_rows = input_cols
    qweight_cols = M // 8
    qweight_dtype = torch.int32
    scales_rows = qweight_rows // group_size
    scales_cols = M
    scales_dtype = torch.float16
    qzeros_rows = scales_rows
    qzeros_cols = qweight_cols
    qzeros_dtype = torch.int32

    torch.manual_seed(0)

    input = torch.rand((input_rows, input_cols),
                       dtype=input_dtype,
                       device=device)
    qweight = torch.randint(0,
                            torch.iinfo(torch.int32).max,
                            (qweight_rows, qweight_cols),
                            dtype=qweight_dtype,
                            device=device)
    qzeros = torch.randint(0,
                           torch.iinfo(torch.int32).max,
                           (qzeros_rows, qzeros_cols),
                           dtype=qzeros_dtype,
                           device=device)
    scales = torch.rand((scales_rows, scales_cols),
                        dtype=scales_dtype,
                        device=device)

    start_gpu = time.time()

    output_triton = torch.ops._rocm_C.awq_gemm_test(input, qweight, scales, qzeros, split_k_iters)
    end_gpu = time.time()
    print(f"gpu_time = {end_gpu - start_gpu}")

    # output_triton = awq_gemm_triton(input, qweight, scales, qzeros,
                                    # split_k_iters)

    # assert (not torch.any(torch.isinf(output_triton))
            # and not torch.any(torch.isnan(output_triton)))

    start_cpu = time.time()
    # dequantized_weights = awq_dequantize_triton(qweight, scales, qzeros)
    dequantized_weights,ss,zz,iw,qv,zv= awq_dequantize_torch(qweight.detach().cpu(), 
                scales.detach().cpu(), qzeros.detach().cpu(), group_size)
    # torch.set_printoptions(threshold=10_000, sci_mode=False)
    # print(f"dequantized_weights[:, 1] = {dequantized_weights[:, 1]}")
    # print(f"a[0,:]={input[0,:]}")
    # print(f"ss[:,1]={ss[:,1]}")
    # print(f"zz[:,1]={zz[:,1]}")
    # print(f"iw[:,1]={iw[:,1]}")
    # print(f"qv[:,1]={qv[:,1]}")
    # print(f"zv[:,1]={zv[:,1]}")
    # print(f"qzeros[:,0]={qzeros[:,0]}")
    # print(f"qweight[:,0]={qweight[:,0]}")
    torch.set_printoptions(profile='default')
    torch.set_printoptions(sci_mode=False)

    output_torch = torch.matmul(input.detach().cpu(), dequantized_weights.detach().cpu())
    torch.cuda.synchronize()

    end_cpu = time.time()
    print(f"cpu_time = {end_cpu - start_cpu}")

    # assert (not torch.any(torch.isinf(output_torch))
            # and not torch.any(torch.isnan(output_torch)))

    # torch.testing.assert_close(output_triton.cpu(),
                               # output_torch.cpu(),
                               # atol=1e-1,
                               # rtol=1e-1)
    print(f"cpu = {output_torch.cpu()}")
    print(f"gpu = {output_triton.cpu()}")

def main():
#  N= 17, K = 128, M = 24, splitK = 8, group_size = 128
    print("="*10)
    # N = 64
    # K = 1024
    # M = 64
    # N = 17
    # K = 32
    # M = 24
    #N = 14, K = 128, M = 32, splitK = 8, group_size = 64
    # N = 14
    # K = 128
    # M = 32
    # splitK = 8
    # group_size = 64
    # N = 1
    # K = 3584
    # M = 448
    N = 1
    K = 128
    M = 32
    splitK = 8
    group_size = 16

    test_gemm(N, K, M, splitK, group_size)
    # test_dequantize(1, 32, 16)

if __name__ == "__main__":
    main()
