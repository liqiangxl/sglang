"""Fused sampling on the compact top-k verify support.

The kernels behind SGLANG_OPT_SPEC_FUSED_COMPACT_VERIFY_SAMPLING.

The compact verify branch (``uno_utils.sample_uno_tree_target_tokens``) samples one
target token per verify row from that row's top-k support. Its chain is
``flashinfer.top_k(sorted=True)`` (on SM10x the clusters kernel, then ``torch.sort``
and a gather of the ids), the compiled support normalisation
(``_sparse_target_support_from_topk``) and ``fast_sample`` (exponential draws, clamp,
divide, row argmax, gathers). Here the draws stay torch's ``exponential_`` of the same
shape (the same random stream, drawn before the top-k), and the top-k stays the same
backend without its sort (optionally with PDL). Then:

- without top-p, one kernel does the rest per row (``_compact_topk_sample_kernel``):
  sort the k entries by value, descending, ties by vocabulary id, ascending; the
  temperature, the per-request top-k mask, softmax over k and the renormalisation,
  with the ops of the compiled reference; the exponential race
  (``argmax(probs / max(q, tiny))``, first index of the maximum) and the token gather;
- with top-p, one kernel sorts the support (``_sort_top_k_kernel``: the same order, the
  values copied bit for bit, int64 ids, the layout ``flashinfer.top_k(sorted=True)``
  returns), main's own compiled ``_sparse_target_support_from_topk`` normalises it
  (the full-distribution top-p cap inlined with the softmax, running sum, cut and
  renormalisation, as on main), and one kernel draws (``_race_kernel``: the
  exponential race and the token gather). The probabilities are main's: the same
  compiled graph on the same values (rows differ from main's only in the order of
  entries with equal values, whose order main leaves unspecified).
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_F32_TINY = tl.constexpr(1.1754943508222875e-38)  # torch.finfo(torch.float32).tiny
_ID_BITS = tl.constexpr(31)  # vocabulary ids < 2**31
_ID_MASK = tl.constexpr((1 << 31) - 1)
# The sort kernel's 64-bit key: 32 bits of value order, 25 of id (ids < 2**25), 7 of
# position in the row (k <= 128).
SORT_MAX_VOCAB = 1 << 25
_SORT_ID_MASK = tl.constexpr((1 << 25) - 1)
_POS_BITS = tl.constexpr(7)
_POS_MASK = tl.constexpr((1 << 7) - 1)
# ``x ^ ones`` rather than ``~x`` on uint32: same bits, and the Triton interpreter's
# ``~`` builds a -1 that NumPy 2 rejects for unsigned dtypes.
_U32_ONES = tl.constexpr(0xFFFFFFFF)


@triton.jit
def _compact_topk_sample_kernel(
    vals_ptr,  # [rows, K] top-k values (logits dtype), any order
    ids_ptr,  # [rows, K] their vocabulary ids (int32 / int64)
    q_ptr,  # [rows, K] fp32 exponential draws
    temps_ptr,  # [bs] fp32
    top_ks_ptr,  # [bs] int
    tokens_ptr,  # [rows] int64 out
    dbg_probs_ptr,  # [rows, K] fp32 out (STORE_DEBUG): final probs, sorted order
    dbg_ids_ptr,  # [rows, K] int64 out (STORE_DEBUG): sorted vocabulary ids
    dbg_pos_ptr,  # [rows] int64 out (STORE_DEBUG): drawn position
    K,
    F,
    BLOCK_K: tl.constexpr,
    STORE_DEBUG: tl.constexpr,
    USE_GDC: tl.constexpr,
    PRECISE_EXP: tl.constexpr,
):
    """The whole chain after the top-k for one row, without top-p."""
    row = tl.program_id(0)
    if USE_GDC:
        # PDL consumer: no global access before the producer grid has completed.
        tl.extra.cuda.gdc_wait()
    req = row // F
    offs = tl.arange(0, BLOCK_K)
    in_k = offs < K
    base = row.to(tl.int64) * K

    v = tl.load(vals_ptr + base + offs, mask=in_k, other=0.0).to(tl.float32)
    ids = tl.load(ids_ptr + base + offs, mask=in_k, other=0).to(tl.int64)

    # Order key: value descending (radix order of the fp32 bits, both zeros as +0),
    # then vocabulary id ascending.
    bits = v.to(tl.uint32, bitcast=True)
    bits = tl.where(v == 0.0, 0, bits)
    ordv = tl.where((bits >> 31) != 0, bits ^ _U32_ONES, bits | 0x80000000)
    key = (ordv.to(tl.int64) << _ID_BITS) | (_ID_MASK - ids)
    key = tl.where(in_k, key, -1)
    key = tl.sort(key, descending=True)
    ordv = (key >> _ID_BITS).to(tl.uint32)
    ids = _ID_MASK - (key & _ID_MASK)
    bits = tl.where((ordv >> 31) != 0, ordv & 0x7FFFFFFF, ordv ^ _U32_ONES)
    v = bits.to(tl.float32, bitcast=True)

    # uno_utils._normalize_sparse_topk_probs without top-p, op for op as Inductor
    # emits it.
    temp = tl.load(temps_ptr + req)
    top_k = tl.load(top_ks_ptr + req)
    valid = in_k & (offs < top_k)
    x = v / temp
    x = tl.where(valid, x, float("-inf"))
    x_max = tl.max(x, axis=0)
    if PRECISE_EXP:
        e = libdevice.exp(x - x_max)
    else:
        e = tl.exp(x - x_max)
    probs = e / tl.sum(e, axis=0)
    probs = probs / tl.maximum(tl.sum(probs, axis=0), 1e-12)

    # spec_utils.fast_sample: argmax(probs / max(q, tiny)), first index of the maximum,
    # never a NaN (as sampling_argmax): a row of NaN draws index 0.
    q = tl.load(q_ptr + base + offs, mask=in_k, other=1.0)
    q = tl.maximum(q, _F32_TINY)
    score = tl.div_rn(probs, q)
    score = tl.where(in_k & (score == score), score, -1.0)
    best = tl.max(score, axis=0)
    pos = tl.min(tl.where(score == best, offs, BLOCK_K), axis=0)
    token = tl.sum(tl.where(offs == pos, ids, 0), axis=0)
    tl.store(tokens_ptr + row, token)

    if STORE_DEBUG:
        tl.store(dbg_probs_ptr + base + offs, probs, mask=in_k)
        tl.store(dbg_ids_ptr + base + offs, ids, mask=in_k)
        tl.store(dbg_pos_ptr + row, pos.to(tl.int64))


@triton.jit
def _sort_top_k_kernel(
    vals_ptr,  # [rows, K] top-k values, any order
    ids_ptr,  # [rows, K] their vocabulary ids (int32 / int64, < 2**25)
    out_vals_ptr,  # [rows, K] out, vals' dtype: the values, sorted
    out_ids_ptr,  # [rows, K] int64 out: their ids, in the same order
    K,
    BLOCK_K: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    """Sort one row as flashinfer.top_k(sorted=True) returns it: value descending (NaN
    first, as torch.sort), ties by id ascending (an order torch.sort leaves open). The
    key carries the entry's position, so each value is copied with its exact bits."""
    row = tl.program_id(0)
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
    offs = tl.arange(0, BLOCK_K)
    in_k = offs < K
    base = row.to(tl.int64) * K
    v = tl.load(vals_ptr + base + offs, mask=in_k, other=0.0).to(tl.float32)
    ids = tl.load(ids_ptr + base + offs, mask=in_k, other=0).to(tl.int64)
    raw = v.to(tl.uint32, bitcast=True)
    bits = tl.where(v == 0.0, 0, raw)
    ordv = tl.where((bits >> 31) != 0, bits ^ _U32_ONES, bits | 0x80000000)
    ones = tl.full([BLOCK_K], _U32_ONES, tl.uint32)
    ordv = tl.where(v != v, ones, ordv)
    # a tensor, not a constexpr, on the left of the subtractions (the interpreter's
    # constexpr - tensor stays a constexpr)
    id_mask = tl.full([BLOCK_K], _SORT_ID_MASK, tl.int64)
    key = (
        (ordv.to(tl.uint64) << 32)
        | ((id_mask - ids).to(tl.uint64) << _POS_BITS)
        | offs.to(tl.uint64)
    )
    key = tl.where(in_k, key, 0)  # every real key is > 0
    key = tl.sort(key, descending=True)
    pos = (key & _POS_MASK).to(tl.int32)
    sid = id_mask - ((key >> _POS_BITS) & _SORT_ID_MASK).to(tl.int64)
    sv = tl.load(vals_ptr + base + pos, mask=in_k)
    tl.store(out_vals_ptr + base + offs, sv, mask=in_k)
    tl.store(out_ids_ptr + base + offs, sid, mask=in_k)


@triton.jit
def _race_kernel(
    probs_ptr,  # [rows, K] fp32 probs, sorted order
    ids_ptr,  # [rows, K] int64 ids, the same order
    q_ptr,  # [rows, K] fp32 exponential draws
    tokens_ptr,  # [rows] int64 out
    pos_ptr,  # [rows] int64 out (STORE_POS): drawn position
    K,
    BLOCK_K: tl.constexpr,
    STORE_POS: tl.constexpr,
):
    """spec_utils.fast_sample on one row and the token gather: argmax(probs /
    max(q, tiny)), first index of the maximum, never a NaN (as sampling_argmax: a row
    of NaN, from a NaN or infinite logit, draws index 0)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    in_k = offs < K
    base = row.to(tl.int64) * K
    probs = tl.load(probs_ptr + base + offs, mask=in_k, other=0.0)
    q = tl.load(q_ptr + base + offs, mask=in_k, other=1.0)
    q = tl.maximum(q, _F32_TINY)
    score = tl.div_rn(probs, q)
    score = tl.where(in_k & (score == score), score, -1.0)  # scores are >= 0
    best = tl.max(score, axis=0)
    pos = tl.min(tl.where(score == best, offs, BLOCK_K), axis=0)  # < K
    token = tl.load(ids_ptr + base + pos)
    tl.store(tokens_ptr + row, token.to(tl.int64))
    if STORE_POS:
        tl.store(pos_ptr + row, pos.to(tl.int64))


def _num_warps(block_k: int) -> int:
    return 1 if block_k <= 32 else (2 if block_k <= 64 else 4)


def _block_k(k: int) -> int:
    return max(16, triton.next_power_of_2(k))


def sort_top_k(
    vals: torch.Tensor, ids: torch.Tensor, pdl: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """(values [rows, K] in vals' dtype, ids [rows, K] int64), each row sorted by value
    descending and then id ascending, as flashinfer.top_k(sorted=True) returns them
    (which leaves the order of equal values open). One launch."""
    rows, k = vals.shape
    assert ids.shape == (rows, k) and k <= (1 << 7)
    assert vals.is_contiguous() and ids.is_contiguous()
    out_vals = torch.empty_like(vals)
    out_ids = torch.empty((rows, k), dtype=torch.int64, device=vals.device)
    block_k = _block_k(k)
    extra = {"launch_pdl": True} if pdl else {}
    _sort_top_k_kernel[(rows,)](
        vals,
        ids,
        out_vals,
        out_ids,
        k,
        BLOCK_K=block_k,
        USE_GDC=pdl,
        num_warps=_num_warps(block_k),
        **extra,
    )
    return out_vals, out_ids


def race(
    probs: torch.Tensor, ids: torch.Tensor, q: torch.Tensor, store_pos: bool = False
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """fast_sample's exponential race on sorted probs and the token gather: (tokens
    [rows] int64, drawn position [rows] int64 or None)."""
    rows, k = probs.shape
    assert ids.shape == (rows, k) and q.shape == (rows, k)
    assert probs.dtype == torch.float32 and q.dtype == torch.float32
    assert probs.is_contiguous() and ids.is_contiguous() and q.is_contiguous()
    tokens = torch.empty((rows,), dtype=torch.int64, device=probs.device)
    pos = torch.empty_like(tokens) if store_pos else tokens
    block_k = _block_k(k)
    _race_kernel[(rows,)](
        probs,
        ids,
        q,
        tokens,
        pos,
        k,
        BLOCK_K=block_k,
        STORE_POS=store_pos,
        num_warps=_num_warps(block_k),
    )
    return tokens, (pos if store_pos else None)


def launch_compact_topk_sample(
    *,
    vals: torch.Tensor,
    ids: torch.Tensor,
    q: torch.Tensor,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: Optional[torch.Tensor],
    verify_width: int,
    pdl: bool = False,
    precise_exp: bool = True,
    debug: bool = False,
    num_warps: Optional[int] = None,
):
    """Sample on a given top-k support (any order of the k entries) and draws.

    Without ``top_ps``: one kernel. With ``top_ps``: the sort kernel, main's compiled
    _sparse_target_support_from_topk on the sorted support and ``logits``, and the
    race kernel.

    Returns ``tokens [rows]`` int64; with ``debug`` also
    ``(probs [rows, K], sorted ids [rows, K], drawn position [rows])``.
    """
    rows, k = vals.shape
    assert ids.shape == (rows, k) and q.shape == (rows, k)
    assert vals.is_contiguous() and ids.is_contiguous() and q.is_contiguous()
    assert q.dtype == torch.float32
    if top_ps is not None:
        from sglang.srt.speculative.uno_utils import _sparse_target_support_from_topk

        assert logits.shape[-1] <= SORT_MAX_VOCAB
        svals, sids = sort_top_k(vals, ids, pdl=pdl)
        _, probs = _sparse_target_support_from_topk(
            topk_logits=svals,
            topk_ids=sids,
            next_token_logits=logits,
            temperatures=temperatures,
            top_ks=top_ks,
            top_ps=top_ps,
            batch_size=rows // verify_width,
            forward_width=verify_width,
            max_top_k=k,
        )
        probs = probs.flatten(0, 1)  # as uno_utils._sample_from_support flattens it
        tokens, pos = race(probs, sids, q, store_pos=debug)
        if debug:
            return tokens, (probs, sids, pos)
        return tokens

    dev = vals.device
    block_k = _block_k(k)
    tokens = torch.empty((rows,), dtype=torch.int64, device=dev)
    if debug:
        dbg = (
            torch.empty((rows, k), dtype=torch.float32, device=dev),
            torch.empty((rows, k), dtype=torch.int64, device=dev),
            torch.empty((rows,), dtype=torch.int64, device=dev),
        )
    else:
        dbg = (tokens, tokens, tokens)
    extra = {"launch_pdl": True} if pdl else {}
    _compact_topk_sample_kernel[(rows,)](
        vals,
        ids,
        q,
        temperatures.reshape(-1),
        top_ks.reshape(-1),
        tokens,
        dbg[0],
        dbg[1],
        dbg[2],
        k,
        verify_width,
        BLOCK_K=block_k,
        STORE_DEBUG=debug,
        USE_GDC=pdl,
        PRECISE_EXP=precise_exp,
        num_warps=num_warps if num_warps is not None else _num_warps(block_k),
        # as Inductor compiles the reference (Triton's default may change)
        enable_fp_fusion=True,
        **extra,
    )
    if debug:
        return tokens, dbg
    return tokens


@functools.lru_cache(maxsize=None)
def _clusters_top_k(device_index: int, algo: Optional[str]) -> bool:
    """Whether flashinfer.top_k(sorted=True) serves this device with the clusters kernel
    (the compact branch's top-k on SM10x), which takes a pdl flag."""
    from flashinfer import topk as fi_topk

    return fi_topk.can_use_clusters_topk(
        algo,
        torch.device("cuda", device_index),
        False,
        fi_topk.TopKTieBreak.NONE,
        False,
    )


def unsorted_top_k(
    logits: torch.Tensor, k: int, pdl: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """(values, ids) of each row's top k, in any order: the clusters kernel that
    flashinfer.top_k(sorted=True) runs before its sort, with PDL when asked; elsewhere
    flashinfer.top_k(sorted=False) itself (no PDL)."""
    if logits.is_cuda and _clusters_top_k(
        logits.device.index if logits.device.index is not None else 0,
        os.environ.get("FLASHINFER_TOPK_ALGO"),
    ):
        from flashinfer.topk import topk_clusters_exact

        ids, vals = topk_clusters_exact(
            logits, k, output_values=True, out_dtype=torch.int32, pdl=pdl
        )
        return vals, ids
    from flashinfer import top_k

    return top_k(logits, k, sorted=False, deterministic=False)


def compact_topk_sample(
    logits: torch.Tensor,
    *,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: Optional[torch.Tensor],
    verify_width: int,
    max_top_k: int,
    pdl: bool,
) -> torch.Tensor:
    """One sampled target token per row of ``logits [bs * verify_width, vocab]`` from
    its top-``max_top_k`` support, int64 ``[rows]``.

    Draws ``exponential_`` on a ``[rows, max_top_k]`` fp32 tensor exactly as
    ``fast_sample`` does on the compact probabilities, so the random stream is the
    reference's; it is drawn first, then the top-k (on ``logits.contiguous()``, as
    main's), then the sampling. With top-p, main's compiled normalisation gets
    ``logits`` itself, as on main.
    """
    rows = logits.shape[0]
    q = torch.empty(
        (rows, max_top_k), dtype=torch.float32, device=logits.device
    ).exponential_(1.0)
    vals, ids = unsorted_top_k(logits.contiguous(), max_top_k, pdl)
    return launch_compact_topk_sample(
        vals=vals,
        ids=ids,
        q=q,
        logits=logits,
        temperatures=temperatures,
        top_ks=top_ks,
        top_ps=top_ps,
        verify_width=verify_width,
        pdl=pdl,
    )
