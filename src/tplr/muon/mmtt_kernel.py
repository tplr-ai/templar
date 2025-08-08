# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import torch
import triton
import triton.language as tl


def _cfg():
    bm = (32, 64, 128)
    bk = (32, 64)
    gm = (4, 8)
    nw = (4, 8)
    ns = (3, 4, 5)
    return [
        triton.Config(dict(BM=i, BK=j, GM=g), num_warps=w, num_stages=s)
        for i in bm
        for j in bk
        for g in gm
        for w in nw
        for s in ns
    ]


@triton.autotune(configs=_cfg(), key=["m", "k"])
@triton.jit
def _mmtt(
    a,
    b,
    m,
    k,
    sxm,
    sxk,
    sym,
    syn,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
):
    p = tl.program_id(axis=0)
    nm = tl.cdiv(m, BM)
    nn = nm
    gcap = GM * nn
    gid = p // gcap
    m0 = gid * GM
    gsz = tl.minimum(nm - m0, GM)
    rm = m0 + ((p % gcap) % gsz)
    rn = (p % gcap) // gsz
    if rm > rn:
        return
    ixm = (rm * BM + tl.arange(0, BM)) % m
    ixn = (rn * BM + tl.arange(0, BM)) % m
    ik = tl.arange(0, BK)
    pa = a + ixm[:, None] * sxm + ik[None, :] * sxk
    pb = a + ixn[:, None] * sxm + ik[None, :] * sxk
    acc = tl.zeros((BM, BM), dtype=tl.float32)
    s = tl.cdiv(k, BK)
    for _ in range(0, s):
        mk = ik[None, :] < k
        va = tl.load(pa, mask=mk, other=0)
        vb = tl.load(pb, mask=mk, other=0)
        acc = tl.dot(va, tl.permute(vb, (1, 0)), acc)
        pa += BK * sxk
        pb += BK * sxk
        k -= BK
    out = acc.to(a.dtype.element_ty)
    om = rm * BM + tl.arange(0, BM)
    on = rn * BM + tl.arange(0, BM)
    ptr = b + sym * om[:, None] + syn * on[None, :]
    msk = (om[:, None] < m) & (on[None, :] < m)
    tl.store(ptr, out, mask=msk)
    if rm < rn:
        ptr_t = b + sym * on[:, None] + syn * om[None, :]
        msk_t = (on[:, None] < m) & (om[None, :] < m)
        tl.store(ptr_t, tl.permute(out, (1, 0)), mask=msk_t)


def matmul_transpose_assign(x, y):
    x = x.contiguous()
    m, k = x.shape
    grid = lambda META: (triton.cdiv(m, META["BM"]) * triton.cdiv(m, META["BM"]),)
    _mmtt[grid](
        x,
        y,
        m,
        k,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
    )


def matmul_transpose(x):
    m, _ = x.shape
    y = torch.empty((m, m), device=x.device, dtype=x.dtype)
    matmul_transpose_assign(x, y)
    return y
