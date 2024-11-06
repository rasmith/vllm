import argparse

import itertools
from tqdm.asyncio import tqdm

import json

import multiprocessing as mp

import os

default_config = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 32,
    "SPLIT_K": 1,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "mfma": 16,
    "kpack": 1,
}
default_values = tuple(default_config.values())


def get_pruner(M, N, K, a_element_size, b_element_size):
    import torch

    def pruner(config):
        return True
        (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, SPLIT_K, GROUP_SIZE_M,
         num_warps, mfma, kpack) = config
        # This is because torch.sum() will have integer overflow.
        if M * N * SPLIT_K >= torch.iinfo(torch.int).max:
            return False
        num_bytes_per_block_a = BLOCK_SIZE_K * BLOCK_SIZE_M * a_element_size
        num_bytes_per_block_b = BLOCK_SIZE_K * BLOCK_SIZE_N * b_element_size
        # Will run out of LDS.
        if num_bytes_per_block_a + num_bytes_per_block_b > 65536:
            return False
        return True

    return pruner


def collect_results(result_configs, M, N, K, use_scalar_scale_a,
                    use_scalar_scale_b, results):
    baseline = result_configs[
        (default_config['BLOCK_SIZE_M'] == result_configs['BLOCK_SIZE_M'])
        & (default_config['BLOCK_SIZE_N'] == result_configs['BLOCK_SIZE_N'])
        & (default_config['BLOCK_SIZE_K'] == result_configs['BLOCK_SIZE_K'])
        & (default_config['SPLIT_K'] == result_configs['SPLIT_K'])
        & (default_config['GROUP_SIZE_M'] == result_configs['GROUP_SIZE_M'])
        & (default_config['num_warps'] == result_configs['num_warps'])
        & (default_config['mfma'] == result_configs['mfma'])
        & (default_config['kpack'] == result_configs['kpack'])]

    best_configs = result_configs[result_configs['Triton'] ==
                                  result_configs['Triton'].min()]
    best_configs = [row.to_dict() for index, row in best_configs.iterrows()]
    baseline = baseline.iloc[0].to_dict()

    for best_config in best_configs:
        best_config['speedup'] = baseline['Triton'] / best_config['Triton']

    result = {
        'M': M,
        'N': N,
        'K': K,
        'use_scalar_scale_a': use_scalar_scale_a,
        'use_scalar_scale_b': use_scalar_scale_b,
        'baseline': baseline,
        'best': best_configs,
    }
    key = f"{M}-{N}-{K}-{use_scalar_scale_a}-{use_scalar_scale_b}"
    results[key] = result


def run_benchmark(update_callback, a, b, scale_a, scale_b, out_dtype, bias,
                  config_choices_list):
    import torch
    import triton
    import triton.language as tl
    from vllm.model_executor.layers.quantization.compressed_tensors\
        .scaled_mm_triton import scaled_mm_triton
    M, K = a.shape
    N = b.shape[1]
    quantiles = [0.5, 0.2, 0.8]
    warmup = 20
    rep = 100
    num_choices = len(config_choices_list)

    tune_benchmark_obj = triton.testing.Benchmark(
        x_names=[
            "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "SPLIT_K",
            "GROUP_SIZE_M", "num_warps", "mfma", "kpack"
        ],
        x_vals=config_choices_list,
        x_log=True,
        line_arg="provider",
        line_vals=["triton"],
        line_names=["Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=None,
        args={},
    )

    @triton.testing.perf_report(tune_benchmark_obj)
    def bench_config(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, SPLIT_K,
                     GROUP_SIZE_M, num_warps, mfma, kpack, provider):
        bench_function = lambda: scaled_mm_triton(
            a, b, scale_a, scale_b, out_dtype, bias, BLOCK_SIZE_M,
            BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, SPLIT_K, num_warps, mfma,
            kpack)
        ms, min_ms, max_ms = triton.testing.do_bench(bench_function,
                                                     quantiles=quantiles,
                                                     warmup=warmup,
                                                     rep=rep)
        update_callback()
        return ms, max_ms, min_ms

    result_data_frames = bench_config.run(return_df=True)
    return result_data_frames


def compute_total_benchmarks(partition_func, get_pruner, shapes,
                             config_choices, scale_choices, a_element_size,
                             b_element_size):
    count = 0
    for shape in shapes:
        M, K, N = shape
        for use_scale in scale_choices():
            pruner = get_pruner(M, N, K, a_element_size, b_element_size)
            work_list = list(filter(pruner, config_choices()))
            count += len(partition_func(work_list))
    return count


def tune(update_callback, start_callback, partition_func, event_queue,
         output_file):
    import torch
    import triton
    import triton.language as tl
    from vllm.model_executor.layers.quantization.compressed_tensors\
        .scaled_mm_triton import scaled_mm_triton
    shapes = [
        # M        K     N
        # (1, 17920, 5120),
        # (1, 5120, 35840),
        (1, 5120, 5120),
        (1, 5120, 7680),
        # (131072, 17920, 5120),
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
    block_k_choices = [32]  #32, 64]  #, 128, 256]
    split_k_choices = [1, 2, 4, 8]
    # split_k_choices = [1, 8]
    # num_warps_choices = [1, 2, 4, 8]
    num_warps_choices = [1, 4, 8]
    # group_m_choices = [1, 4, 8, 16, 32]
    group_m_choices = [1, 8, 16, 32]
    matrix_instr_nonkdim_range = [16, 32]
    kpack_range = [1, 2]

    config_choices = lambda: itertools.product(
        block_m_choices, block_n_choices, block_k_choices, split_k_choices,
        group_m_choices, num_warps_choices, matrix_instr_nonkdim_range,
        kpack_range)
    scale_choices = lambda: itertools.product(use_scalar_a_choices,
                                              use_scalar_b_choices)

    bias = None  # never used
    out_dtype = torch.float16

    torch.manual_seed(0)

    a_element_size = 1
    b_element_size = 1

    num_total_benchmarks = compute_total_benchmarks(partition_func, get_pruner,
                                                    shapes, config_choices,
                                                    scale_choices,
                                                    a_element_size,
                                                    b_element_size)
    start_callback(num_total_benchmarks)

    results = {}
    for shape in shapes:
        M, K, N = shape
        a = torch.randint(0, 127, (M, K), dtype=torch.int8, device='cuda')
        b = torch.randint(0, 127, (K, N), dtype=torch.int8, device='cuda')
        for use_scale in scale_choices():
            use_scalar_scale_a, use_scalar_scale_b = use_scale
            if use_scalar_scale_a:
                scale_a = torch.rand((1, 1), device='cuda')
            else:
                scale_a = torch.rand((M, 1), device='cuda')
            if use_scalar_scale_b:
                scale_b = torch.rand((1, 1), device='cuda')
            else:
                scale_b = torch.rand((N, 1), device='cuda')
            config_choices_list = list(
                filter(get_pruner(M, N, K, a_element_size, b_element_size),
                       config_choices()))
            work_list = partition_func(config_choices_list)
            result_configs = run_benchmark(update_callback, a, b, scale_a,
                                           scale_b, out_dtype, bias, work_list)

            collect_results(result_configs, M, N, K, use_scalar_scale_a,
                            use_scalar_scale_b, results)

    json_output = json.dumps(results, sort_keys=True, indent=4)
    if (output_file):
        with open(output_file, 'w') as f:
            f.write(json_output)
    else:
        print(json_output)


def listener_function(event_queue, num_jobs):
    bars = {}
    for event in iter(event_queue.get, None):
        event_type = event['type']
        pid = event['pid']
        if event_type == 'start':
            bars[pid] = tqdm(desc=f"GPU{pid}", total=event['count'])
        elif event_type == 'update':
            bars[pid].update()


def partition_list(l, pid, num_jobs):
    n = len(l)
    count = (n + num_jobs - 1) // num_jobs
    start = pid * count
    end = start + count
    work_list = l[start:end]
    if default_values not in work_list:
        work_list.append(default_values)
    return work_list


def worker_function(pid, num_jobs, parent_connection, event_queue,
                    output_file):
    update_callback = lambda: event_queue.put({'pid': pid, 'type': 'update'})

    start_callback = lambda count: event_queue.put({
        'pid': pid,
        'type': 'start',
        'count': count
    })
    partition_func = lambda l: partition_list(l, pid, num_jobs)

    tune(update_callback, start_callback, partition_func, event_queue,
         output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', type=int)
    args = parser.parse_args()
    print(f"#jobs = {args.jobs}")

    num_jobs = min(args.jobs, 8)

    mp.set_start_method('spawn')

    event_queue = mp.Queue()

    listener = mp.Process(target=listener_function,
                          args=(
                              event_queue,
                              num_jobs,
                          ))
    listener.start()

    worker_infos = []
    for pid in range(num_jobs):
        os.environ["HIP_VISIBLE_DEVICES"] = f"{pid}"
        parent_conn, child_conn = mp.Pipe()
        output_file = f"results_{pid}.txt"
        worker = mp.Process(target=worker_function,
                            args=(pid, num_jobs, child_conn, event_queue,
                                  output_file))
        worker.start()
        worker_infos.append({'worker': worker, 'connection': parent_conn})

    for info in worker_infos:
        info['worker'].join()

    event_queue.put(None)
    listener.join()


if __name__ == '__main__':
    main()
