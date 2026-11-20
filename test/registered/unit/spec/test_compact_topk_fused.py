"""SGLANG_OPT_SPEC_FUSED_COMPACT_VERIFY_SAMPLING: the compact top-k branch, fused.

The switch is off by default and then nothing changes; on, eagle_sample's compact branch
takes the fused sampler, and everything after it (acceptance, the broadcast, the
simulated-acceptance tail with its full-vocabulary bonus argmax) is unchanged. The draws
are fast_sample's (same shape, drawn before the top-k). Without top-p one kernel samples
each row; it must reproduce the torch chain (_normalize_sparse_topk_probs, then
fast_sample's exponential race) on the same draws for any order of the k entries. With
top-p the support is sorted by one kernel (flashinfer.top_k(sorted=True)'s order, values
bit for bit) and normalised by main's own compiled _sparse_target_support_from_topk,
which must get bitwise main's arguments and so give main's probabilities; one kernel then
draws. The kernels run in the Triton interpreter on a CPU-only host (on the GPU
otherwise). The GPU comparisons against flashinfer.top_k live in
test/registered/kernels/ops/speculative/test_compact_topk_sampling.py.
"""

import os
import sys
import types
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

_HAS_CUDA = torch.cuda.is_available()
if not _HAS_CUDA:
    # Without a GPU the kernels run in the Triton interpreter; its Triton library
    # calls (tl.sort, tl.cumsum) are interpreted only when this is set before Triton
    # is first imported.
    os.environ.setdefault("TRITON_INTERPRET", "1")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.speculative import (  # noqa: E402
    eagle_utils,
    spec_utils,
    uno_utils,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_TINY = torch.finfo(torch.float32).tiny

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


def _kernels():
    """compact_topk_sampling: compiled with a GPU, interpreted without (TRITON_INTERPRET
    is set above before Triton is imported)."""
    from sglang.kernels.ops.speculative import compact_topk_sampling

    return compact_topk_sampling


_DEV = "cuda" if _HAS_CUDA else "cpu"


def _to_dev(kw):
    return {k: v.to(_DEV) if isinstance(v, torch.Tensor) else v for k, v in kw.items()}


def _launch(kernels, **kw):
    """launch_compact_topk_sample on _DEV (inputs built on the CPU), outputs on the CPU;
    libdevice exp on a GPU, numpy's in the interpreter."""
    tokens, (probs, sids, pos) = kernels.launch_compact_topk_sample(
        **_to_dev(kw), precise_exp=_HAS_CUDA, debug=True
    )
    return tokens.cpu(), (probs.cpu(), sids.cpu(), pos.cpu())


# ---------------------------------------------------------------- torch reference


def _stable_top_k(logits, k):
    """Top-k by value descending, ties by index ascending (the fused kernel's order)."""
    order = torch.sort(logits, dim=-1, descending=True, stable=True).indices[:, :k]
    return logits.gather(1, order), order


def _reference_chain(vals, ids, q, temps, top_ks, top_ps, verify_width, logits):
    """uno_utils' chain on a sorted support: probs, drawn position, token. With top-p,
    main's compiled _sparse_target_support_from_topk itself; without, the same ops
    eagerly (the fused kernel's reference)."""
    rows, k = vals.shape
    if top_ps is not None:
        _, probs = uno_utils._sparse_target_support_from_topk(
            topk_logits=vals.contiguous(),
            topk_ids=ids.contiguous(),
            next_token_logits=logits,
            temperatures=temps,
            top_ks=top_ks,
            top_ps=top_ps,
            batch_size=rows // verify_width,
            forward_width=verify_width,
            max_top_k=k,
        )
        probs = probs.flatten(0, 1)
    else:
        exp = lambda t: torch.repeat_interleave(t.reshape(-1), verify_width).reshape(
            rows, 1
        )
        ranks = torch.arange(k, device=vals.device)[None, :]
        probs = uno_utils._normalize_sparse_topk_probs(
            vals, logits, exp(temps), ranks < exp(top_ks), None
        )
    scores = probs.float() / q.clamp_min(_TINY)
    pos = scores.argmax(dim=-1)
    return probs, pos, ids.gather(1, pos[:, None]).squeeze(1)


def _bits(t):
    """A tensor's bit pattern, so equality also covers signed zeros and NaN payloads."""
    t = t.contiguous()
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype == torch.float64:
        return t.view(torch.int64)
    return t


class _Recorder:
    """Wraps a function: records each call's keyword arguments (tensors cloned)."""

    def __init__(self, real):
        self.real = real
        self.calls = []

    def __call__(self, **kw):
        # copies with the caller's strides, so a stride comparison sees the real layout
        self.calls.append(
            {
                k: (
                    torch.empty_strided(
                        v.shape, v.stride(), dtype=v.dtype, device=v.device
                    ).copy_(v)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in kw.items()
            }
        )
        return self.real(**kw)


def _adversarial_rows(vocab, gen):
    """Logit rows that stress the cap: ties in and across the top-k (BF16 grid), a
    constant row, signed zeros at the maximum, a support holding all the mass, an
    exp-underflowing tail, a huge range, an infinity and a NaN."""
    x = (torch.randn(vocab, generator=gen) * 3.0).to(torch.bfloat16).float()
    rows = [x]
    tied = x.clone()
    tied[torch.randperm(vocab, generator=gen)[:30]] = float(x.max()) + 1.0
    rows.append(tied)
    rows.append(torch.full((vocab,), 1.5))
    zeros = -torch.rand(vocab, generator=gen) - 1.0
    zeros[torch.randperm(vocab, generator=gen)[:6]] = 0.0
    zeros[torch.randperm(vocab, generator=gen)[:6]] = -0.0
    rows.append(zeros)
    alone = torch.full((vocab,), float("-inf"))
    alone[torch.randperm(vocab, generator=gen)[:40]] = torch.randn(40, generator=gen)
    rows.append(alone)
    under = torch.randn(vocab, generator=gen) - 200.0
    under[torch.randperm(vocab, generator=gen)[:5]] = 50.0
    rows.append(under)
    rows.append(torch.randn(vocab, generator=gen) * 3e4)
    flat = torch.randn(vocab, generator=gen) * 0.01
    rows.append(flat)
    inf = torch.randn(vocab, generator=gen)
    inf[7] = float("inf")
    rows.append(inf)
    nan = torch.randn(vocab, generator=gen)
    nan[torch.randperm(vocab, generator=gen)[:1]] = float("nan")
    rows.append(nan)
    return torch.stack(rows)


# ---------------------------------------------------------------- wiring


class _Wiring:
    """eagle_sample on the CPU up to the compact branch and the simulated bonus."""

    BS, TOKENS, VOCAB, TOP_K = 2, 7, 64, 20

    def __init__(self, case, *, fused, mode="real-draft-token", supported=True):
        self.case = case
        gen = torch.Generator().manual_seed(3)
        rows = self.BS * self.TOKENS
        self.logits = torch.randn(rows, self.VOCAB, generator=gen)
        self.sampled = torch.randint(
            0, self.VOCAB, (self.BS, self.TOKENS), generator=gen
        )
        self.fused, self.mode, self.supported = fused, mode, supported
        self.ref = MagicMock(return_value=self.sampled.clone())
        self.fused_fn = MagicMock(return_value=self.sampled.clone())
        self.sim = MagicMock(side_effect=lambda **kw: kw["accept_index"])

    def run(self):
        bs, tokens = self.BS, self.TOKENS
        sampling_info = SimpleNamespace(
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            sampling_seed=None,
            has_watermark_candidates=False,
            is_all_greedy=False,
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            need_min_p_sampling=False,
            temperatures=torch.full((bs, 1), 0.6),
            top_ks=torch.full((bs,), self.TOP_K, dtype=torch.int32),
            top_ps=torch.full((bs,), 0.95),
        )
        batch = SimpleNamespace(
            device="cpu",
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            seq_lens=torch.full((bs,), 100),
            req_pool_indices=torch.arange(bs),
            sampling_info=sampling_info,
            return_logprob=False,
        )
        verify_input = SimpleNamespace(
            draft_token=torch.zeros(bs * tokens, dtype=torch.int64),
            draft_token_num=tokens,
            max_tree_depth=tokens,
            retrieve_index=torch.arange(bs * tokens).view(bs, tokens),
            retrieve_next_token=torch.zeros((bs, tokens), dtype=torch.int64),
            retrieve_next_sibling=torch.zeros((bs, tokens), dtype=torch.int64),
            tree_topk=1,
            draft_probs=None,
        )
        spec = SimpleNamespace(
            speculative_use_rejection_sampling=False,
            speculative_use_block_verification=False,
        )
        parallel = SimpleNamespace(tp_group=SimpleNamespace(world_size=1))
        with ExitStack() as stack:
            enter = lambda p: stack.enter_context(p)
            enter(patch.object(eagle_utils, "_COMPACT_TOPK_FUSED", self.fused))
            enter(patch.object(eagle_utils, "_compact_topk_fused_logged", False))
            enter(patch.object(eagle_utils, "_is_cuda", True))
            enter(patch.object(eagle_utils, "_is_cpu", False))
            enter(patch.object(eagle_utils, "_is_hip", False))
            enter(patch.object(eagle_utils, "_is_xpu", False))
            enter(patch.object(eagle_utils, "get_spec", return_value=spec))
            enter(
                patch("sglang.srt.runtime_context.get_parallel", return_value=parallel)
            )
            enter(
                patch.object(
                    eagle_utils,
                    "verify_tree_greedy_func",
                    side_effect=lambda **kw: (
                        kw["predicts"],
                        kw["accept_index"],
                        kw["accept_token_num"],
                    ),
                )
            )
            enter(patch.object(uno_utils, "sample_uno_tree_target_tokens", self.ref))
            enter(
                patch.object(
                    uno_utils, "sample_uno_tree_target_tokens_fused", self.fused_fn
                )
            )
            enter(
                patch.object(
                    uno_utils,
                    "compact_topk_fused_supported",
                    return_value=self.supported,
                )
            )
            enter(patch.object(spec_utils, "SIMULATE_ACC_LEN", 4.8))
            enter(patch.object(spec_utils, "SIMULATE_ACC_TOKEN_MODE", self.mode))
            enter(patch.object(spec_utils, "generate_simulated_accept_index", self.sim))
            eagle_utils.eagle_sample(
                verify_input,
                batch,
                SimpleNamespace(
                    next_token_logits=self.logits.clone(),
                    next_token_logits_vocab_shard=None,
                ),
                uno_target_max_top_k=self.TOP_K,
                sparse_bonus_argmax=True,
            )
        return self.sim.call_args.kwargs["target_predict"]

    def full_argmax(self):
        return self.logits.argmax(-1).view(self.BS, self.TOKENS)


class TestCompactTopKFusedWiring(CustomTestCase):
    def test_switch_is_off_by_default_and_explicit(self):
        if os.environ.get("SGLANG_OPT_SPEC_FUSED_COMPACT_VERIFY_SAMPLING") is None:
            self.assertFalse(envs.SGLANG_OPT_SPEC_FUSED_COMPACT_VERIFY_SAMPLING.get())
            self.assertFalse(eagle_utils._COMPACT_TOPK_FUSED)
            # does not follow the NVIDIA runtime master switch
            with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True):
                self.assertFalse(
                    envs.SGLANG_OPT_SPEC_FUSED_COMPACT_VERIFY_SAMPLING.get()
                )
        with envs.SGLANG_OPT_SPEC_FUSED_COMPACT_VERIFY_SAMPLING.override(True):
            self.assertTrue(envs.SGLANG_OPT_SPEC_FUSED_COMPACT_VERIFY_SAMPLING.get())

    def test_off_keeps_the_reference_branch(self):
        w = _Wiring(self, fused=False)
        bonus = w.run()
        w.ref.assert_called_once()
        w.fused_fn.assert_not_called()
        self.assertTrue(torch.equal(bonus, w.full_argmax()))

    def test_on_takes_the_fused_sampler_and_keeps_the_simulated_bonus(self):
        w = _Wiring(self, fused=True)
        with self.assertLogs(eagle_utils.logger, level="INFO") as logs:
            bonus = w.run()
        self.assertIn(
            "SGLANG_OPT_SPEC_FUSED_COMPACT_VERIFY_SAMPLING engaged",
            "\n".join(logs.output),
        )
        w.ref.assert_not_called()
        w.fused_fn.assert_called_once()
        kwargs = w.fused_fn.call_args.kwargs
        self.assertEqual(
            sorted(kwargs),
            [
                "batch_size",
                "max_top_k",
                "next_token_logits",
                "sampling_info",
                "verify_width",
            ],
        )
        self.assertEqual(
            (kwargs["batch_size"], kwargs["verify_width"], kwargs["max_top_k"]),
            (w.BS, w.TOKENS, w.TOP_K),
        )
        # the simulated real-draft-token bonus is still the full-vocabulary argmax
        self.assertTrue(torch.equal(bonus, w.full_argmax()))

    def test_fixed_token_mode_keeps_the_sampled_targets(self):
        w = _Wiring(self, fused=True, mode="fixed")
        target = w.run()
        w.fused_fn.assert_called_once()
        self.assertTrue(torch.equal(target, w.sampled))

    def test_unsupported_logits_keep_the_reference_branch(self):
        w = _Wiring(self, fused=True, supported=False)
        bonus = w.run()
        w.ref.assert_called_once()
        w.fused_fn.assert_not_called()
        self.assertTrue(torch.equal(bonus, w.full_argmax()))

    def test_supported_needs_fp32_cuda_rows_and_the_exponential_race(self):
        def logits(is_cuda=True, dim=2, dtype=torch.float32, vocab=248320):
            return SimpleNamespace(
                is_cuda=is_cuda, dim=lambda: dim, dtype=dtype, shape=(14, vocab)
            )

        self.assertTrue(uno_utils.compact_topk_fused_supported(logits()))
        self.assertTrue(uno_utils.compact_topk_fused_supported(logits(vocab=1 << 25)))
        self.assertFalse(
            uno_utils.compact_topk_fused_supported(logits(vocab=(1 << 25) + 1))
        )
        self.assertFalse(uno_utils.compact_topk_fused_supported(logits(is_cuda=False)))
        self.assertFalse(uno_utils.compact_topk_fused_supported(logits(dim=3)))
        self.assertFalse(
            uno_utils.compact_topk_fused_supported(logits(dtype=torch.bfloat16))
        )
        with envs.SGLANG_OPT_USE_GUMBEL_SAMPLE.override(False):
            self.assertFalse(uno_utils.compact_topk_fused_supported(logits()))

    def test_fused_wrapper_passes_top_p_only_when_sampled_and_shapes_rows(self):
        bs, width, k = 2, 7, 20
        info = SimpleNamespace(
            temperatures=torch.full((bs, 1), 0.6),
            top_ks=torch.full((bs,), k, dtype=torch.int32),
            top_ps=torch.full((bs,), 0.9),
            need_top_p_sampling=False,
        )
        out = torch.arange(bs * width)
        fake = types.ModuleType("fake_cts")
        fake.compact_topk_sample = MagicMock(return_value=out)
        kw = dict(
            next_token_logits=torch.zeros(bs * width, 64),
            sampling_info=info,
            batch_size=bs,
            verify_width=width,
            max_top_k=k,
        )
        with (
            patch.dict(
                sys.modules,
                {"sglang.kernels.ops.speculative.compact_topk_sampling": fake},
            ),
            patch("sglang.kernels.jit.utils.is_arch_support_pdl", return_value=True),
        ):
            tokens = uno_utils.sample_uno_tree_target_tokens_fused(**kw)
            kwargs = fake.compact_topk_sample.call_args.kwargs
            self.assertIsNone(kwargs["top_ps"])
            self.assertTrue(kwargs["pdl"])
            self.assertEqual((kwargs["verify_width"], kwargs["max_top_k"]), (width, k))
            info.need_top_p_sampling = True
            uno_utils.sample_uno_tree_target_tokens_fused(**kw)
            self.assertIs(
                fake.compact_topk_sample.call_args.kwargs["top_ps"], info.top_ps
            )
        self.assertEqual(tokens.shape, (bs, width))
        self.assertTrue(torch.equal(tokens.flatten(), out))


@unittest.skipUnless(_HAS_TRITON, "needs Triton (interpreter on CPU)")
class TestCompactTopKFusedHost(CustomTestCase):
    """The host side: the top-k backend, the draws, their order."""

    def test_clusters_top_k_takes_the_pdl_flag_and_other_devices_fall_back(self):
        kernels = _kernels()
        logits = SimpleNamespace(is_cuda=True, device=torch.device("cpu"))
        fi = types.ModuleType("flashinfer")
        fi_topk = types.ModuleType("flashinfer.topk")
        fi_topk.topk_clusters_exact = MagicMock(return_value=("ids", "vals"))
        fi.topk = fi_topk
        fi.top_k = MagicMock(return_value=("v2", "i2"))
        with patch.dict(sys.modules, {"flashinfer": fi, "flashinfer.topk": fi_topk}):
            with patch.object(kernels, "_clusters_top_k", return_value=True):
                self.assertEqual(
                    kernels.unsorted_top_k(logits, 20, True), ("vals", "ids")
                )
                args, kwargs = fi_topk.topk_clusters_exact.call_args
                self.assertEqual(args[1], 20)
                self.assertEqual(
                    kwargs, dict(output_values=True, out_dtype=torch.int32, pdl=True)
                )
            with patch.object(kernels, "_clusters_top_k", return_value=False):
                self.assertEqual(kernels.unsorted_top_k(logits, 20, True), ("v2", "i2"))
                self.assertEqual(
                    fi.top_k.call_args.kwargs, dict(sorted=False, deterministic=False)
                )

    def test_draws_are_fast_samples_and_come_before_the_top_k(self):
        kernels = _kernels()
        rows, vocab, k = 14, 300, 20
        logits = torch.randn(rows, vocab)
        calls = []

        def top_k(x, kk, pdl):
            calls.append(("top_k", torch.random.get_rng_state().clone()))
            vals, ids = _stable_top_k(x, kk)
            return vals, ids.to(torch.int32)

        def launch(**kw):
            calls.append(("launch", kw["q"].clone()))
            return torch.zeros(rows, dtype=torch.int64)

        torch.manual_seed(5)
        with (
            patch.object(kernels, "unsorted_top_k", side_effect=top_k),
            patch.object(kernels, "launch_compact_topk_sample", side_effect=launch),
        ):
            kernels.compact_topk_sample(
                logits,
                temperatures=torch.full((2, 1), 0.6),
                top_ks=torch.full((2,), k),
                top_ps=None,
                verify_width=7,
                max_top_k=k,
                pdl=False,
            )
        self.assertEqual([c[0] for c in calls], ["top_k", "launch"])
        # fast_sample's draw on the compact probs: empty_like(probs).exponential_(1.0)
        torch.manual_seed(5)
        q_ref = torch.empty_like(
            torch.empty(rows, k), dtype=torch.float32
        ).exponential_(1.0)
        self.assertTrue(torch.equal(calls[1][1], q_ref))
        # the draw is made before the top-k (no other RNG use in between)
        torch.manual_seed(5)
        torch.empty(rows, k).exponential_(1.0)
        self.assertTrue(torch.equal(calls[0][1], torch.random.get_rng_state()))


# ---------------------------------------------------------------- top-p route


@unittest.skipUnless(_HAS_TRITON, "needs Triton (interpreter on CPU)")
class TestCompactTopKTopPRoute(CustomTestCase):
    """With top-p the support is sorted by one kernel and normalised by main's own
    compiled _sparse_target_support_from_topk: the sort is flashinfer.top_k(sorted=
    True)'s order with the values copied bit for bit, and the normalisation gets
    bitwise main's arguments, so the probabilities (the top-p cap included) are main's."""

    VOCAB = 4096

    @classmethod
    def setUpClass(cls):
        cls.kernels = _kernels()

    def test_sort_kernel_is_value_descending_then_id_ascending(self):
        """NaN, +-0, +-inf, ties, subnormals, ids up to 2**25 - 1, k from 1 to 128:
        torch.sort's order (NaN first, -0 == +0) with ties by id, values bit for bit."""
        g = torch.Generator().manual_seed(9)
        specials = torch.tensor(
            [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1e-45, -1e-45]
            + [3e38, -3e38, 1.0, 1.0, -2.5, -2.5]
        )
        top = self.kernels.SORT_MAX_VOCAB - 1
        rows = 6
        for k in (1, 2, 7, 16, 20, 33, 64, 100, 128):
            vals = (torch.randn(rows, k, generator=g) * 4).to(torch.bfloat16).float()
            pick = torch.randint(0, specials.numel(), (rows, k), generator=g)
            where = torch.rand(rows, k, generator=g) < 0.3
            vals = torch.where(where, specials[pick], vals).contiguous()
            # distinct ids spread over [0, 2**25), the largest one included
            ids = torch.stack(
                [torch.randperm(1000, generator=g)[:k] for _ in range(rows)]
            ) * (top // 1000)
            ids[0, -1] = top
            # reference: id ascending, then a stable value-descending torch.sort
            by_id = ids.argsort(dim=-1, stable=True)
            v1, i1 = vals.gather(1, by_id), ids.gather(1, by_id)
            order = torch.sort(v1, dim=-1, descending=True, stable=True).indices
            ref_vals, ref_ids = v1.gather(1, order), i1.gather(1, order)
            for id_dtype in (torch.int32, torch.int64):
                with self.subTest(k=k, ids=id_dtype):
                    sv, si = self.kernels.sort_top_k(
                        vals.to(_DEV), ids.to(id_dtype).contiguous().to(_DEV)
                    )
                    sv, si = sv.cpu(), si.cpu()
                    self.assertEqual((sv.dtype, si.dtype), (torch.float32, torch.int64))
                    self.assertTrue(torch.equal(si, ref_ids))
                    self.assertTrue(torch.equal(_bits(sv), _bits(ref_vals)))

    def _main_and_fused(self, logits, info, bs, width, k, shuffle_seed):
        """main's sample_uno_tree_target_tokens and the fused path on the same seed and
        the same top-k (the fused one shuffled, as an unsorted top-k returns it), with
        their calls to _sparse_target_support_from_topk recorded."""

        def sorted_top_k(x, kk, sorted, deterministic):
            vals, ids = _stable_top_k(x, kk)
            return vals.contiguous(), ids.contiguous()

        g = torch.Generator().manual_seed(shuffle_seed)

        def unsorted_top_k(x, kk, pdl):
            vals, ids = _stable_top_k(x, kk)
            perm = torch.stack(
                [torch.randperm(kk, generator=g) for _ in range(x.shape[0])]
            ).to(x.device)
            return (
                vals.gather(1, perm).contiguous(),
                ids.gather(1, perm).to(torch.int32).contiguous(),
            )

        kw = dict(
            next_token_logits=logits,
            sampling_info=info,
            batch_size=bs,
            verify_width=width,
            max_top_k=k,
        )
        real = uno_utils._sparse_target_support_from_topk
        main_rec, fused_rec = _Recorder(real), _Recorder(real)
        with (
            patch.object(uno_utils, "_flashinfer_top_k", side_effect=sorted_top_k),
            patch.object(uno_utils, "_sparse_target_support_from_topk", main_rec),
        ):
            torch.manual_seed(7)
            tok_main = uno_utils.sample_uno_tree_target_tokens(**kw)
        with (
            patch.object(self.kernels, "unsorted_top_k", side_effect=unsorted_top_k),
            patch.object(uno_utils, "_sparse_target_support_from_topk", fused_rec),
            patch("sglang.kernels.jit.utils.is_arch_support_pdl", return_value=False),
        ):
            torch.manual_seed(7)
            tok_fused = uno_utils.sample_uno_tree_target_tokens_fused(**kw)
        return (tok_main, main_rec.calls), (tok_fused, fused_rec.calls)

    def _assert_same_call(self, want, got):
        self.assertEqual(set(want), set(got))
        for name in want:
            a, b = want[name], got[name]
            if isinstance(a, torch.Tensor):
                self.assertEqual(
                    (a.shape, a.dtype, a.stride()), (b.shape, b.dtype, b.stride()), name
                )
                self.assertTrue(torch.equal(_bits(a), _bits(b)), name)
            else:
                self.assertEqual(a, b, name)

    def test_normalisation_gets_mains_arguments_and_tokens_match(self):
        """Mixed per-request temperature, top-k and top-p; BF16-grid rows with ties;
        contiguous logits and a column view (main hands the compiled normalisation the
        logits as they are, and the top-k a contiguous copy)."""
        width, k = 3, 20
        gen = torch.Generator().manual_seed(5)
        settings = [
            # (temperatures, top_ks, top_ps)
            ([0.6], [20], [0.95]),
            ([1e-3, 7.5], [1, 19], [1e-6, 0.999999]),
            ([0.6, 0.9], [13, 20], [0.8, 0.3]),
            ([2.0, 0.6, 1.0], [7, 3, 20], [0.9, 1e-3, 1.0]),
        ]
        for n, (temps, top_ks, top_ps) in enumerate(settings):
            bs = len(temps)
            logits = (torch.randn(bs * width, self.VOCAB, generator=gen) * 3.0).to(
                torch.bfloat16
            )
            logits = logits.float().to(_DEV)
            wide = torch.cat([logits, torch.zeros_like(logits[:, :5])], dim=1)
            info = SimpleNamespace(
                temperatures=torch.tensor(temps).view(bs, 1).to(_DEV),
                top_ks=torch.tensor(top_ks, dtype=torch.int32).to(_DEV),
                top_ps=torch.tensor(top_ps).to(_DEV),
                need_top_p_sampling=True,
            )
            for layout, x in (("contiguous", logits), ("column view", wide[:, :-5])):
                with self.subTest(setting=n, layout=layout):
                    (tok_main, main_calls), (tok_fused, fused_calls) = (
                        self._main_and_fused(x, info, bs, width, k, shuffle_seed=n)
                    )
                    self.assertEqual((len(main_calls), len(fused_calls)), (1, 1))
                    self._assert_same_call(main_calls[0], fused_calls[0])
                    self.assertEqual(
                        main_calls[0]["next_token_logits"].stride(), x.stride()
                    )
                    self.assertTrue(torch.equal(tok_main, tok_fused))

    def test_adversarial_rows_give_mains_arguments_and_tokens(self):
        """Ties in and across the top-k, signed zeros at the maximum, a support holding
        all the mass, underflow, a huge range, an infinity and a NaN. The infinity and
        NaN rows normalise to all-NaN probs: both paths draw index 0 there."""
        width, k = 2, 20
        logits = _adversarial_rows(self.VOCAB, torch.Generator().manual_seed(6))
        bs = logits.shape[0] // width
        info = SimpleNamespace(
            temperatures=torch.tensor([0.6, 1.0, 1e-3, 7.5, 0.9]).view(bs, 1).to(_DEV),
            top_ks=torch.tensor([20, 13, 1, 20, 7], dtype=torch.int32).to(_DEV),
            top_ps=torch.tensor([0.95, 0.5, 1.0, 1e-6, 0.8]).to(_DEV),
            need_top_p_sampling=True,
        )
        (tok_main, main_calls), (tok_fused, fused_calls) = self._main_and_fused(
            logits.to(_DEV), info, bs, width, k, shuffle_seed=11
        )
        self.assertTrue(torch.equal(tok_main, tok_fused))
        self._assert_same_call(main_calls[0], fused_calls[0])


# ---------------------------------------------------------------- sampling kernel


@unittest.skipUnless(_HAS_TRITON, "needs Triton (interpreter on CPU)")
class TestCompactTopKFusedKernel(CustomTestCase):
    """The kernel (in the Triton interpreter without a GPU) against the torch chain on
    the CPU."""

    VOCAB = 2048

    @classmethod
    def setUpClass(cls):
        cls.kernels = _kernels()

    def _run(self, logits, k, temps, top_ks, top_ps, width, *, perm_seed=None, q=None):
        rows = logits.shape[0]
        vals, ids = _stable_top_k(logits, k)
        if q is None:
            q = torch.empty(rows, k).exponential_(1.0)
        in_vals, in_ids = vals, ids
        if perm_seed is not None:
            g = torch.Generator().manual_seed(perm_seed)
            perm = torch.stack([torch.randperm(k, generator=g) for _ in range(rows)])
            in_vals, in_ids = vals.gather(1, perm), ids.gather(1, perm)
        tokens, (probs, sids, pos) = _launch(
            self.kernels,
            vals=in_vals.contiguous(),
            ids=in_ids.to(torch.int32).contiguous(),
            q=q,
            logits=logits,
            temperatures=temps,
            top_ks=top_ks,
            top_ps=top_ps,
            verify_width=width,
        )
        # on the kernels' device: with top-p the reference is main's compiled graph,
        # which the fused path calls itself, so the probs compare bit for bit
        dev = lambda t: t.to(_DEV) if t is not None else None
        ref = _reference_chain(
            dev(vals),
            dev(ids),
            dev(q),
            dev(temps),
            dev(top_ks),
            dev(top_ps),
            width,
            dev(logits),
        )
        ref = tuple(t.cpu() for t in ref)
        return (tokens, probs, sids, pos), (ref, ids)

    def test_matches_the_torch_chain_on_the_same_draws(self):
        torch.manual_seed(0)
        cases = [
            # (bs, k, temperature, top_ks, top_ps)
            (1, 20, [0.6], [20], [0.95]),
            (2, 20, [0.6, 1.0], [20, 13], [0.95, 0.5]),
            (2, 7, [1.3, 0.6], [7, 2], None),
            (2, 128, [0.6, 0.9], [128, 64], [0.95, 1.0]),
        ]
        width = 7
        for bs, k, temps, top_ks, top_ps in cases:
            for family in ("randn", "bf16"):
                with self.subTest(bs=bs, k=k, family=family, top_p=top_ps is not None):
                    logits = torch.randn(bs * width, self.VOCAB) * 3.0
                    if family == "bf16":
                        logits = logits.to(torch.bfloat16).float()
                    (tok, probs, sids, pos), ((p_ref, pos_ref, tok_ref), ids) = (
                        self._run(
                            logits,
                            k,
                            torch.tensor(temps).view(bs, 1),
                            torch.tensor(top_ks, dtype=torch.int32),
                            torch.tensor(top_ps) if top_ps is not None else None,
                            width,
                        )
                    )
                    self.assertTrue(torch.equal(sids, ids))
                    if top_ps is not None:
                        # main's own compiled normalisation on main's arguments
                        self.assertTrue(torch.equal(_bits(probs), _bits(p_ref)))
                    torch.testing.assert_close(probs, p_ref, rtol=2e-6, atol=1e-7)
                    self.assertTrue(torch.equal(probs == 0, p_ref == 0))
                    self.assertTrue(torch.equal(pos, pos_ref))
                    self.assertTrue(torch.equal(tok, tok_ref))

    def test_any_order_of_the_k_entries_gives_the_same_result(self):
        """Both paths: the one kernel (top-p off) and the sort kernel (top-p on)."""
        torch.manual_seed(1)
        width, k = 7, 20
        logits = (torch.randn(2 * width, self.VOCAB) * 3.0).to(torch.bfloat16).float()
        q = torch.empty(2 * width, k).exponential_(1.0)
        for top_ps in (None, torch.tensor([0.95, 0.95])):
            args = (
                logits,
                k,
                torch.tensor([[0.6], [0.6]]),
                torch.tensor([20, 20], dtype=torch.int32),
                top_ps,
                width,
            )
            base = self._run(*args, q=q)[0]
            for seed in range(3):
                with self.subTest(top_p=top_ps is not None, seed=seed):
                    out = self._run(*args, q=q, perm_seed=seed)[0]
                    for a, b in zip(base, out):
                        self.assertTrue(torch.equal(a, b))

    def test_top_p_off_is_top_p_one(self):
        torch.manual_seed(2)
        width, k = 7, 20
        logits = torch.randn(width, self.VOCAB) * 3.0
        q = torch.empty(width, k).exponential_(1.0)
        common = (
            logits,
            k,
            torch.tensor([[0.6]]),
            torch.tensor([20], dtype=torch.int32),
        )
        # different code (the one-kernel path vs main's normalisation): the same draws,
        # tokens and order, probs within a few ulp
        (tok0, p0, sids0, pos0) = self._run(*common, None, width, q=q)[0]
        (tok1, p1, sids1, pos1) = self._run(*common, torch.tensor([1.0]), width, q=q)[0]
        self.assertTrue(torch.equal(sids0, sids1))
        self.assertTrue(torch.equal(pos0, pos1))
        self.assertTrue(torch.equal(tok0, tok1))
        torch.testing.assert_close(p0, p1, rtol=2e-6, atol=1e-7)

    def test_rows_of_nan_draw_index_0(self):
        """An infinite or NaN logit in the top-k makes the row's probs NaN; both paths
        then draw index 0, as sampling_argmax does, and stay inside the row."""
        torch.manual_seed(4)
        width, k = 2, 20
        logits = torch.randn(2 * width, self.VOCAB) * 3.0
        logits[1, 5] = float("inf")
        logits[3, 9] = float("nan")
        q = torch.empty(2 * width, k).exponential_(1.0)
        for top_ps in (None, torch.tensor([0.95, 0.9])):
            with self.subTest(top_p=top_ps is not None):
                (tok, _, sids, pos), ((_, pos_ref, tok_ref), _) = self._run(
                    logits,
                    k,
                    torch.tensor([[0.6], [1.0]]),
                    torch.tensor([20, 20], dtype=torch.int32),
                    top_ps,
                    width,
                    q=q,
                )
                self.assertEqual(pos[[1, 3]].tolist(), [0, 0])
                self.assertEqual(tok[[1, 3]].tolist(), sids[[1, 3], 0].tolist())
                self.assertTrue(torch.equal(pos, pos_ref))
                self.assertTrue(torch.equal(tok, tok_ref))

    def test_top_p_is_the_full_distributions(self):
        """Top-p rescaled to the support's share of the full distribution (#41916): a
        flat tail outside the top-k makes the cut differ from a top-k-first top-p."""
        torch.manual_seed(3)
        width, k = 7, 20
        logits = torch.randn(2 * width, self.VOCAB) * 0.5
        head = torch.stack([torch.randperm(self.VOCAB)[:k] for _ in range(2 * width)])
        logits.scatter_(1, head, 4.0 + torch.linspace(0.0, 2.0, k).expand(2 * width, k))
        q = torch.empty(2 * width, k).exponential_(1.0)
        temps = torch.tensor([[0.6], [1.0]])
        top_ks = torch.tensor([20, 13], dtype=torch.int32)
        top_ps = torch.tensor([0.5, 0.3])
        (_, probs, _, pos), ((p_ref, pos_ref, _), _) = self._run(
            logits, k, temps, top_ks, top_ps, width, q=q
        )
        torch.testing.assert_close(probs, p_ref, rtol=2e-6, atol=1e-7)
        self.assertTrue(torch.equal(probs == 0, p_ref == 0))
        self.assertTrue(torch.equal(pos, pos_ref))
        vals, _ = _stable_top_k(logits, k)
        exp = lambda t: torch.repeat_interleave(t.reshape(-1), width).reshape(-1, 1)
        scaled = (vals / exp(temps)).masked_fill(
            ~(torch.arange(k)[None, :] < exp(top_ks)), float("-inf")
        )
        support = torch.softmax(scaled, dim=-1)
        topk_first = (support.cumsum(-1) - support) > exp(top_ps)
        self.assertFalse(torch.equal(probs == 0, topk_first))


if __name__ == "__main__":
    unittest.main()
