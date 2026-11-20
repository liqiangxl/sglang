"""compact_topk_sampling on the GPU against the compact verify branch it replaces.

Vocabulary 248,320, rows 7 / 14 / 21 / 28 (verify width 7):
(b) on the reference's own support and the same draws: with top-p, the probabilities
    are bitwise main's on every row (main's own compiled normalisation on the sorted
    support), so no entry is cut on one side only; without top-p, the one-kernel path's
    probs differ from main's by a few ulp at most; the drawn tokens match apart from
    exact logit ties;
(c) uno_utils.sample_uno_tree_target_tokens and the fused path on the same seed draw
    the same tokens apart from ties;
(d) a CUDA graph (with PDL where supported) replays what the eager kernels give.
Each case prints one ``REPORT {json}`` line.
"""

import json
import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative import uno_utils
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-small")

_TINY = torch.finfo(torch.float32).tiny
# COMPACT_TOPK_VOCAB lowers the vocabulary for small GPUs.
VOCAB = int(os.environ.get("COMPACT_TOPK_VOCAB", "248320"))
ROWS = (7, 14, 21, 28)
WIDTH = 7
TRIALS = int(os.environ.get("COMPACT_TOPK_TRIALS", "40"))


def _report(**kw):
    print("REPORT " + json.dumps(kw, sort_keys=True), flush=True)


def _ulp(a, b):
    """Elementwise ulp distance of two non-negative fp32 tensors."""
    return (a.view(torch.int32).long() - b.view(torch.int32).long()).abs()


def _lm_logits(rows, vocab, generator):
    """Logit rows shaped like an LM's (a peaked head over a noisy tail) on the BF16 grid,
    as the lm_head's BF16 GEMM leaves them, so the top-k holds ties."""
    dev = "cuda"
    tail = torch.randn(rows, vocab, generator=generator, device=dev) * 2.0
    head = 14.0 + 3.0 * torch.rand(rows, 1, generator=generator, device=dev)
    gaps = torch.empty(rows, 40, device=dev).exponential_(1.5, generator=generator)
    head = head - torch.cumsum(gaps, dim=1)
    where = torch.stack(
        [
            torch.randperm(vocab, generator=generator, device=dev)[:40]
            for _ in range(rows)
        ]
    )
    tail.scatter_(1, where, head)
    return tail.to(torch.bfloat16).float()


def _logits(family, rows, gen):
    if family == "randn":
        return torch.randn(rows, VOCAB, generator=gen, device="cuda") * 3.0
    return _lm_logits(rows, VOCAB, gen)


def _info(bs, k=20, temperature=0.6, top_p=0.95, mixed=False):
    dev = "cuda"
    if mixed:
        temps = torch.tensor([0.6, 1.0, 0.8, 0.6][:bs], device=dev).view(bs, 1)
        top_ks = torch.tensor([20, 13, 5, 20][:bs], dtype=torch.int32, device=dev)
        top_ps = torch.tensor([0.95, 0.8, 1.0, 0.5][:bs], device=dev)
    else:
        temps = torch.full((bs, 1), temperature, device=dev)
        top_ks = torch.full((bs,), k, dtype=torch.int32, device=dev)
        top_ps = torch.full((bs,), top_p if top_p is not None else 1.0, device=dev)
    return SimpleNamespace(
        temperatures=temps,
        top_ks=top_ks,
        top_ps=top_ps,
        need_top_p_sampling=top_p is not None,
    )


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestCompactTopKSamplingGPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.jit.utils import is_arch_support_pdl
        from sglang.kernels.ops.speculative import compact_topk_sampling

        cls.k = compact_topk_sampling
        cls.pdl = is_arch_support_pdl()

    def test_b_fused_chain_vs_the_compiled_reference(self):
        gen = torch.Generator(device="cuda").manual_seed(12)
        k = 20
        for rows in ROWS:
            bs = rows // WIDTH
            for family in ("randn", "lm"):
                for setting in ("default", "no_top_p", "mixed"):
                    info = (
                        _info(bs, mixed=True)
                        if setting == "mixed"
                        else _info(bs, top_p=None if setting == "no_top_p" else 0.95)
                    )
                    stats = dict(
                        n_elem=0,
                        elem_bitident=0,
                        n_rows=0,
                        row_bitident=0,
                        max_abs=0.0,
                        max_ulp=0,
                        mask_flips=0,
                        tok_ident=0,
                        pos_ident=0,
                        tie_explained=0,
                        topk_value_rows_equal=0,
                    )
                    for t in range(TRIALS):
                        logits = _logits(family, rows, gen)
                        ids3, probs3 = uno_utils._build_sparse_target_support(
                            next_token_logits=logits,
                            sampling_info=info,
                            batch_size=bs,
                            forward_width=WIDTH,
                            max_top_k=k,
                        )
                        ids = ids3.flatten(0, 1)
                        p_ref = probs3.flatten(0, 1).float()
                        seed = 1000 + t
                        torch.manual_seed(seed)
                        tok_ref = uno_utils._sample_from_support(ids3, probs3).flatten()
                        torch.manual_seed(seed)
                        q = torch.empty(
                            (rows, k), dtype=torch.float32, device="cuda"
                        ).exponential_(1.0)
                        tok, (p_new, sids, pos) = self.k.launch_compact_topk_sample(
                            vals=logits.gather(1, ids).contiguous(),
                            ids=ids.contiguous(),
                            q=q,
                            logits=logits,
                            temperatures=info.temperatures,
                            top_ks=info.top_ks,
                            top_ps=info.top_ps if info.need_top_p_sampling else None,
                            verify_width=WIDTH,
                            debug=True,
                        )
                        same_vals = torch.equal(
                            logits.gather(1, ids), logits.gather(1, sids)
                        )
                        stats["topk_value_rows_equal"] += rows if same_vals else 0
                        diff = (p_new - p_ref).abs()
                        ulp = _ulp(p_new, p_ref)
                        # one side cut by top-p, the other not
                        flip = (p_new == 0) != (p_ref == 0)
                        stats["mask_flips"] += int(flip.sum())
                        stats["n_elem"] += p_new.numel()
                        stats["elem_bitident"] += int((ulp == 0).sum())
                        stats["n_rows"] += rows
                        stats["row_bitident"] += int((ulp == 0).all(dim=1).sum())
                        stats["max_abs"] = max(stats["max_abs"], float(diff.max()))
                        stats["max_ulp"] = max(
                            stats["max_ulp"], int(torch.where(flip, 0, ulp).max())
                        )
                        pos_ref = (p_ref / q.clamp_min(_TINY)).argmax(dim=-1)
                        stats["pos_ident"] += int((pos == pos_ref).sum())
                        same = tok == tok_ref
                        stats["tok_ident"] += int(same.sum())
                        tied = logits.gather(1, tok[:, None]) == logits.gather(
                            1, tok_ref[:, None]
                        )
                        stats["tie_explained"] += int((~same & tied.squeeze(1)).sum())
                    _report(
                        test="b_chain",
                        rows=rows,
                        family=family,
                        setting=setting,
                        **stats,
                    )
                    self.assertEqual(stats["topk_value_rows_equal"], stats["n_rows"])
                    if info.need_top_p_sampling:
                        # main's own compiled normalisation on the same values
                        self.assertEqual(stats["row_bitident"], stats["n_rows"])
                    self.assertEqual(stats["mask_flips"], 0)
                    self.assertLessEqual(stats["max_ulp"], 64)
                    unexplained = (
                        stats["n_rows"] - stats["tok_ident"] - stats["tie_explained"]
                    )
                    self.assertLessEqual(unexplained, max(1, stats["n_rows"] // 1000))

    def test_c_whole_sampler_identity(self):
        gen = torch.Generator(device="cuda").manual_seed(13)
        k = 20
        for rows in ROWS:
            bs = rows // WIDTH
            info = _info(bs)
            for family in ("randn", "lm"):
                st = dict(
                    n=0, ref_vs_ref=0, new_vs_ref=0, new_vs_new=0, new_vs_ref_tie=0
                )
                for t in range(TRIALS):
                    logits = _logits(family, rows, gen)
                    outs = {}
                    for name, fused in (
                        ("ref", 0),
                        ("ref2", 0),
                        ("new", 1),
                        ("new2", 1),
                    ):
                        torch.manual_seed(2000 + t)
                        fn = (
                            uno_utils.sample_uno_tree_target_tokens_fused
                            if fused
                            else uno_utils.sample_uno_tree_target_tokens
                        )
                        outs[name] = fn(
                            next_token_logits=logits,
                            sampling_info=info,
                            batch_size=bs,
                            verify_width=WIDTH,
                            max_top_k=k,
                        ).flatten()
                    st["n"] += rows
                    st["ref_vs_ref"] += int((outs["ref"] == outs["ref2"]).sum())
                    same = outs["new"] == outs["ref"]
                    st["new_vs_ref"] += int(same.sum())
                    st["new_vs_new"] += int((outs["new"] == outs["new2"]).sum())
                    tied = (
                        logits.gather(1, outs["new"][:, None])
                        == logits.gather(1, outs["ref"][:, None])
                    ).squeeze(1)
                    st["new_vs_ref_tie"] += int((~same & tied).sum())
                _report(test="c_whole", rows=rows, family=family, **st)
                self.assertLessEqual(
                    st["n"] - st["new_vs_ref"] - st["new_vs_ref_tie"],
                    max(1, st["n"] // 1000),
                )

    def test_d_graph_replays_the_eager_kernels(self):
        gen = torch.Generator(device="cuda").manual_seed(14)
        k = 20
        for rows, top_p in [(r, p) for r in ROWS for p in (0.95, None)]:
            bs = rows // WIDTH
            info = _info(bs, top_p=top_p)
            logits = torch.empty(rows, VOCAB, device="cuda")
            q = torch.empty(rows, k, device="cuda")
            head = torch.zeros(rows, dtype=torch.int32, device="cuda")

            def body(pdl):
                head.zero_()  # a producer kernel ahead of the PDL launches
                vals, ids = self.k.unsorted_top_k(logits, k, pdl)
                return self.k.launch_compact_topk_sample(
                    vals=vals,
                    ids=ids,
                    q=q,
                    logits=logits,
                    temperatures=info.temperatures,
                    top_ks=info.top_ks,
                    top_ps=info.top_ps if info.need_top_p_sampling else None,
                    verify_width=WIDTH,
                    pdl=pdl,
                )

            logits.copy_(_logits("randn", rows, gen))
            q.exponential_(1.0)
            body(self.pdl)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                g_tokens = body(self.pdl)
            bad = 0
            for _ in range(TRIALS):
                logits.copy_(_logits("randn", rows, gen))
                q.exponential_(1.0)
                graph.replay()
                e_tokens = body(False)
                torch.cuda.synchronize()
                bad += int((g_tokens != e_tokens).sum())
            _report(
                test="d_graph",
                rows=rows,
                top_p=info.need_top_p_sampling,
                pdl=self.pdl,
                replays=TRIALS,
                mismatches=bad,
            )
            self.assertEqual(bad, 0)


if __name__ == "__main__":
    unittest.main()
