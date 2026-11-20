"""Benchmark the compact top-k verify sampler: the stock chain vs the fused kernel.

One verify round's target sampling for ``rows = bs * 7`` rows (an EAGLE/MTP chain of 7
verify tokens) at top-k 20 and temperature 0.6, with and without top-p 0.95:

- ``reference``: uno_utils.sample_uno_tree_target_tokens (flashinfer.top_k sorted, the
  compiled support normalisation, fast_sample);
- ``fused``: sample_uno_tree_target_tokens_fused (SGLANG_OPT_SPEC_FUSED_COMPACT_VERIFY_
  SAMPLING), PDL where supported.
"""

import functools
from types import SimpleNamespace

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.srt.speculative import uno_utils
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-small"
)

WIDTH = 7
TOP_K = 20


def _lm_logits(rows: int, vocab: int) -> torch.Tensor:
    """A peaked head over a noisy tail on the BF16 grid, like an LM head's output."""
    gen = torch.Generator(device="cuda").manual_seed(rows)
    tail = torch.randn(rows, vocab, generator=gen, device="cuda") * 2.0
    head = 14.0 + 3.0 * torch.rand(rows, 1, generator=gen, device="cuda")
    gaps = torch.empty(rows, 40, device="cuda").exponential_(1.5, generator=gen)
    head = head - torch.cumsum(gaps, dim=1)
    where = torch.stack(
        [torch.randperm(vocab, generator=gen, device="cuda")[:40] for _ in range(rows)]
    )
    tail.scatter_(1, where, head)
    return tail.to(torch.bfloat16).float()


def _info(temps, top_ks, top_ps, top_p: bool) -> SimpleNamespace:
    return SimpleNamespace(
        temperatures=temps,
        top_ks=top_ks,
        top_ps=top_ps,
        need_top_p_sampling=top_p,
    )


def reference(logits, temps, top_ks, top_ps, *, top_p):
    return uno_utils.sample_uno_tree_target_tokens(
        next_token_logits=logits,
        sampling_info=_info(temps, top_ks, top_ps, top_p),
        batch_size=temps.shape[0],
        verify_width=WIDTH,
        max_top_k=TOP_K,
    )


def fused(logits, temps, top_ks, top_ps, *, top_p):
    return uno_utils.sample_uno_tree_target_tokens_fused(
        next_token_logits=logits,
        sampling_info=_info(temps, top_ks, top_ps, top_p),
        batch_size=temps.shape[0],
        verify_width=WIDTH,
        max_top_k=TOP_K,
    )


FN_MAP = {
    "reference": reference,
    "fused": fused,
}


@marker.parametrize("vocab_size", [151936, 248320], [248320])
@marker.parametrize("rows", [7, 14, 28, 56, 112, 224, 448], [7, 28, 224])
@marker.parametrize("top_p", [True, False], [True, False])
@marker.benchmark("impl", list(FN_MAP))
def benchmark(vocab_size: int, rows: int, top_p: bool, impl: str):
    bs = rows // WIDTH
    args = (
        _lm_logits(rows, vocab_size),
        torch.full((bs, 1), 0.6, device="cuda"),
        torch.full((bs,), TOP_K, dtype=torch.int32, device="cuda"),
        torch.full((bs,), 0.95, device="cuda"),
    )
    return marker.do_bench(
        functools.partial(FN_MAP[impl], top_p=top_p),
        input_args=args,
        graph_clone_args=[0],
        graph_clone_kwargs=None,
    )


if __name__ == "__main__":
    benchmark.run()
