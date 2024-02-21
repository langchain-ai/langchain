# https://github.com/fpgaminer/GPTQ-triton
"""
Mostly the same as the autotuner in Triton, but with a few changes like using 40 runs instead of 100.
"""

import builtins
import math
import time
from typing import Dict

import triton


class Autotuner(triton.KernelInterface):
    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        prune_configs_by: Dict = None,
        nearest_power_of_two: bool = False,
    ):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
                'perf_model': performance model used to predicate running time with different configs, returns running time
                'top_k': number of configs to bench
                'prune_num_stages_by'(optional): a function used to prune num_stages. It take configs:List[Config] as its input, and returns pruned configs.
                'nearest_power_of_two'(optional): whether to round key arguments to the nearest power of two when caching tuning results
        """
        if not configs:
            self.configs = [triton.Config({}, num_warps=4, num_stages=2)]
        else:
            self.configs = configs
        self.key_idx = [arg_names.index(k) for k in key]
        self.nearest_power_of_two = nearest_power_of_two
        self.cache = {}
        # hook to reset all required tensor to zeros before relaunching a kernel
        self.hook = lambda args: 0
        if reset_to_zero is not None:
            self.reset_idx = [arg_names.index(k) for k in reset_to_zero]

            def _hook(args):
                for i in self.reset_idx:
                    args[i].zero_()

            self.hook = _hook
        self.arg_names = arg_names
        # prune configs
        if prune_configs_by:
            perf_model, top_k = (
                prune_configs_by["perf_model"],
                prune_configs_by["top_k"],
            )
            if "early_config_prune" in prune_configs_by:
                early_config_prune = prune_configs_by["early_config_prune"]
        else:
            perf_model, top_k, early_config_prune = None, None, None
        self.perf_model, self.configs_top_k = perf_model, top_k
        self.early_config_prune = early_config_prune
        self.fn = fn

    def _bench(self, *args, config, **meta):
        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols."
            )
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.kwargs)

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(self.nargs)
            self.hook(args)
            self.fn.run(
                *args,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                **current,
            )

        try:
            # In testings using only 40 reps seems to be close enough and it appears to be what PyTorch uses
            # PyTorch also sets fast_flush to True, but I didn't see any speedup so I'll leave the default
            return triton.testing.do_bench(
                kernel_call, percentiles=(0.5, 0.2, 0.8), rep=40
            )
        except triton.compiler.OutOfResources:
            return (float("inf"), float("inf"), float("inf"))

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        if len(self.configs) > 1:
            key = tuple(args[i] for i in self.key_idx)

            # This reduces the amount of autotuning by rounding the keys to the nearest power of two
            # In my testing this gives decent results, and greatly reduces the amount of tuning required
            if self.nearest_power_of_two:
                key = tuple([2 ** int(math.log2(x) + 0.5) for x in key])

            if key not in self.cache:
                # prune configs
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()
                timings = {
                    config: self._bench(*args, config=config, **kwargs)
                    for config in pruned_configs
                }
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                self.hook(args)
                self.configs_timings = timings
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if config.pre_hook is not None:
            config.pre_hook(self.nargs)
        return self.fn.run(
            *args,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
            **kwargs,
            **config.kwargs,
        )

    def prune_configs(self, kwargs):
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.kwargs,
                        num_stages=config.num_stages,
                        num_warps=config.num_warps,
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[
                    :top_k
                ]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        for config in self.prune_configs(kwargs):
            self.fn.warmup(
                *args,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                **kwargs,
                **config.kwargs,
            )
        self.nargs = None


def autotune(
    configs, key, prune_configs_by=None, reset_to_zero=None, nearest_power_of_two=False
):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.
    .. highlight:: python
    .. code-block:: python
            @triton.autotune(configs=[
                    triton.Config(meta={'BLOCK_SIZE': 128}, num_warps=4),
                    triton.Config(meta={'BLOCK_SIZE': 1024}, num_warps=8),
                    ],
                    key=['x_size'] # the two above configs will be evaluated anytime
                                                    # the value of x_size changes
            )
            @triton.jit
            def kernel(x_ptr, x_size, **META):
                    BLOCK_SIZE = META['BLOCK_SIZE']
    :note: When all the configurations are evaluated, the kernel will run multiple time.
                    This means that whatever value the kernel updates will be updated multiple times.
                    To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
                    reset the value of the provided tensor to `zero` before running any configuration.
    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It take configs:List[Config] as its input, and returns pruned configs.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    """

    def decorator(fn):
        return Autotuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            prune_configs_by,
            nearest_power_of_two,
        )

    return decorator


def matmul248_kernel_config_pruner(configs, nargs):
    """
    The main purpose of this function is to shrink BLOCK_SIZE_* when the corresponding dimension is smaller.
    """
    m = max(2 ** int(math.ceil(math.log2(nargs["M"]))), 16)
    n = max(2 ** int(math.ceil(math.log2(nargs["N"]))), 16)
    k = max(2 ** int(math.ceil(math.log2(nargs["K"]))), 16)

    used = set()
    for config in configs:
        block_size_m = min(m, config.kwargs["BLOCK_SIZE_M"])
        block_size_n = min(n, config.kwargs["BLOCK_SIZE_N"])
        block_size_k = min(k, config.kwargs["BLOCK_SIZE_K"])
        group_size_m = config.kwargs["GROUP_SIZE_M"]

        if (
            block_size_m,
            block_size_n,
            block_size_k,
            group_size_m,
            config.num_stages,
            config.num_warps,
        ) in used:
            continue

        used.add(
            (
                block_size_m,
                block_size_n,
                block_size_k,
                group_size_m,
                config.num_stages,
                config.num_warps,
            )
        )
        yield triton.Config(
            {
                "BLOCK_SIZE_M": block_size_m,
                "BLOCK_SIZE_N": block_size_n,
                "BLOCK_SIZE_K": block_size_k,
                "GROUP_SIZE_M": group_size_m,
            },
            num_stages=config.num_stages,
            num_warps=config.num_warps,
        )
