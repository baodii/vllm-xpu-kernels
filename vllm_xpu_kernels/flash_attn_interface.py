# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch

#isort: off
try:
    from . import _vllm_fa2_C  # noqa: F401
    FA2_UNAVAILABLE_REASON = None
    FA2_AVAILABLE = True
except ImportError as e:
    FA2_UNAVAILABLE_REASON = str(e)
    FA2_AVAILABLE = False

#isort: on

DEFAULT_FA_VERSION = 2


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: Optional[list[int]] = None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # FA3 Only
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits: int = 0,
    # Version selector
    fa_version: int = DEFAULT_FA_VERSION,
    s_aux: Optional[torch.Tensor] = None,
):
    assert cu_seqlens_k is not None or seqused_k is not None, \
        "cu_seqlens_k or seqused_k must be provided"
    assert cu_seqlens_k is None or seqused_k is None, \
        "cu_seqlens_k and seqused_k cannot be provided at the same time"

    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)
    # custom op does not support non-tuple input
    real_window_size: tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)
    print(q.shape, k.shape, v.shape)
    print(cu_seqlens_q)
    print(cu_seqlens_k)
    print(block_table)
    print(f"seqused_k: {seqused_k}")
    print(f"cu_seqlens_k: {cu_seqlens_k}")

    if fa_version == 2:
        if scheduler_metadata is not None and q_descale is not None \
            and k_descale is not None and v_descale is not None:
            raise NotImplementedError(
                "FA2 does not support scheduler_metadata, q_descale, "
                "k_descale, v_descale")
        if num_splits > 1:
            raise NotImplementedError("FA2 does not support num_splits > 1")

        if max_seqlen_q == 1:
            batch_size = block_table.shape[0]
            num_heads_kv = k.shape[2]
            head_size = k.shape[-1]
            block_size = k.shape[1]
            batch_k = torch.zeros([batch_size, max_seqlen_k, num_heads_kv, head_size], dtype=k.dtype, device=k.device)
            batch_v = torch.zeros([batch_size, max_seqlen_k, num_heads_kv, head_size], dtype=v.dtype, device=v.device)
            for i in range(batch_size):
                seq_k_len = cu_seqlens_k[i+1] - cu_seqlens_k[i]
                num_blocks = (seq_k_len + block_size - 1) // block_size
                block_indices = block_table[i, :num_blocks]
                k_slice = k[block_indices].view(-1, num_heads_kv, head_size)
                k_slice = k_slice[:seq_k_len]
                v_slice = v[block_indices].view(-1, num_heads_kv, head_size)
                v_slice = v_slice[:seq_k_len]
                batch_k[i, :seq_k_len] = k_slice
                batch_v[i, :seq_k_len] = v_slice

            batch_q = q.view(batch_size, -1, q.shape[-2], q.shape[-1])
            batch_k = batch_k.permute(0, 2, 1, 3).contiguous()
            batch_v = batch_v.permute(0, 2, 1, 3).contiguous()
            batch_q = batch_q.permute(0, 2, 1, 3).contiguous()
            print(f"batch_q: {batch_q.shape}, batch_k: {batch_k.shape}, batch_v: {batch_v.shape}")

            out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd(
                batch_q,
                batch_k,
                batch_v,
                out,
                cu_seqlens_q,
                # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp
                # still wants it so we pass all zeros
                dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
                seqused_k,
                None,
                block_table,
                alibi_slopes,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                s_aux,
                False,
                causal,
                real_window_size[0],
                real_window_size[1],
                softcap,
                return_softmax_lse and dropout_p > 0,
                None,
            )

    else:
        raise NotImplementedError("not support yet")
    return (out, softmax_lse) if return_softmax_lse else out
