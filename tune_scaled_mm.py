import argparse

import itertools
import pdb
from tqdm.asyncio import tqdm

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.quantization.compressed_tensors\
    .scaled_mm_triton import scaled_mm_triton

def get_pruner(M, N, K, a, b):
    def pruner(config):
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, SPLIT_K, GROUP_SIZE_M, num_warps, mfma, kpack = config
        # This is because torch.sum() will have integer overflow.
        if M * N * SPLIT_K >= torch.iinfo(torch.int).max:
            return False
        num_bytes_per_block_a = BLOCK_SIZE_K * BLOCK_SIZE_M * a.element_size()
        num_bytes_per_block_b = BLOCK_SIZE_K * BLOCK_SIZE_N * b.element_size()
        # Will run out of LDS.
        if num_bytes_per_block_a + num_bytes_per_block_b > 65536:
            return False
        return True
    return pruner


def main():
    shapes = [
        # M        K     N
        # (1, 17920, 5120),
        # (1, 5120, 35840),
        # (1, 5120, 5120),
        # (1, 5120, 7680),
        (131072, 17920, 5120),
        # (131072, 5120, 35840),
        # (131072, 5120, 5120),
        # (131072, 5120, 7680),
        # (15, 17920, 5120),
        # (15, 5120, 35840),
        # (15, 5120, 5120),
        # (15, 5120, 7680),
    ]

    use_scalar_a_choices = [True, False]
    use_scalar_b_choices = [True, False]
    block_m_choices = [16, 32]  #, 64]#, 128, 256]
    block_n_choices = [16, 32]  #, 64]#, 128, 256]
    block_k_choices = [32]  #, 64]#, 128, 256]
    split_k_choices = [1, 2, 4, 8]
    num_warps_choices = [1, 2, 4, 8]
    group_m_choices = [1, 4, 8, 16, 32]
    matrix_instr_nonkdim_range = [16, 32]
    kpack_range = [1, 2]

    config_choices = lambda: itertools.product(
        block_m_choices, block_n_choices, block_k_choices, split_k_choices,
        group_m_choices, num_warps_choices, matrix_instr_nonkdim_range, kpack_range
    )
    scale_choices = lambda: itertools.product(use_scalar_a_choices,
                                              use_scalar_b_choices)

    bias = None  # never used
    out_dtype = torch.float16

    quantiles = [0.5, 0.2, 0.8]
    warmup = 20
    rep = 100

    torch.manual_seed(0)

    for shape in shapes:
        print(f"shape = {shape}")
        M, K, N = shape
        a = torch.randint(0, 127, (M, K), dtype=torch.int8, device='cuda')
        b = torch.randint(0, 127, (K, N), dtype=torch.int8, device='cuda')
        for use_scale in scale_choices():
            print(f"scale = {use_scale}")
            use_scalar_scale_a, use_scalar_scale_b = use_scale
            if use_scalar_scale_a:
                scale_a = torch.rand((1, 1), device='cuda')
            else:
                scale_a = torch.rand((M, 1), device='cuda')
            if use_scalar_scale_b:
                scale_b = torch.rand((1, 1), device='cuda')
            else:
                scale_b = torch.rand((N, 1), device='cuda')


            config_choices_list  = list(filter(get_pruner(M, N, K, a, b), config_choices()))
            num_choices = len(config_choices_list)
            print(f"num_choices = {num_choices}")

            tune_benchmark_obj = triton.testing.Benchmark(
                x_names=[
                    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "SPLIT_K",
                    "GROUP_SIZE_M", "num_warps", "mfma", "kpack"
                ],
                # x_vals=filter(prune_config_choices(M, K, N), config_choices()),
                x_vals=tqdm(config_choices_list, total=num_choices),
                x_log=True,
                line_arg="provider",
                line_vals=["triton"],
                line_names=["Triton"],
                styles=[("blue", "-"), ("green", "-")],
                ylabel="ms",
                plot_name="Tuning performance",
                args={},
            )

            @triton.testing.perf_report(tune_benchmark_obj)
            def bench_config(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, SPLIT_K,
                             GROUP_SIZE_M, num_warps, mfma, kpack, provider):
                # print(f"BLOCK_SIZE_M = {BLOCK_SIZE_M}, BLOCK_SIZE_N = {BLOCK_SIZE_N}, BLOCK_SIZE_K = {BLOCK_SIZE_K} , SPLIT_K = {SPLIT_K}, GROUP_SIZE_M = {GROUP_SIZE_M}, num_warps = {num_warps}, mfma = {mfma}, kpack = {kpack}")

                import time
                start = time.time()
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: scaled_mm_triton(a, b, scale_a, scale_b, out_dtype,
                                             bias, BLOCK_SIZE_M, BLOCK_SIZE_N,
                                             BLOCK_SIZE_K, GROUP_SIZE_M, SPLIT_K,
                                             num_warps, mfma, kpack),
                    quantiles=quantiles,
                    warmup=warmup,
                    rep=rep)
                end = time.time()

                return ms, max_ms, min_ms

            result_data_frames = bench_config.run(return_df=True)
            best_data_frames = result_data_frames[
                result_data_frames['Triton'] == result_data_frames['Triton'].min()]
            print(f"best[M={M},K={K},N={N}]=")
            print(f"{best_data_frames}")

if __name__ == '__main__':
    main()
