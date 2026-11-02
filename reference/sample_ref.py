"""The sampler of llama/main.bend (sample, uniform) in Python, operation for operation, so a test can ask for the same
token from the same F32 logits and seed.

Every value is an np.float32 and every operation one F32 operation in Bend's order: F32 + - * / round to nearest as
in C, and Bend's F32.exp is C's double exp rounded to F32 (the emitted C is `(f32)exp(x)`), which math.exp (the same
libm) followed by np.float32 repeats.
"""
import math
import numpy as np

M32 = 0xFFFFFFFF
F = np.float32


def hash32(x):
    """Chris Wellons' lowbias32, as hash32 in llama/main.bend"""
    x &= M32
    x ^= x >> 16
    x = (x * 2146121005) & M32
    x ^= x >> 15
    x = (x * 2221713035) & M32
    x ^= x >> 16
    return x


def uniform(seed, n):
    """draw n of the seed: the top 24 bits of hash32(hash32(seed) + n) times 2^-24, an F32 in [0, 1)"""
    return F(hash32((hash32(seed) + n) & M32) >> 8) * F(2.0 ** -24)


def f32_exp(x):
    return F(math.exp(float(x)))


def weight(v, temp, top):
    return f32_exp(F(v / temp) - F(top / temp))


def topk(xs, k):
    """the k largest values with their ids, largest first; equal values in id order (a stable sort)"""
    order = sorted(range(len(xs)), key=lambda i: (-float(xs[i]), i))[:k]
    return [(F(xs[i]), i) for i in order]


def sample(xs, temp, k, p, u):
    """(id, logit) drawn from the F32 logits xs, with temperature temp, top-k k, top-p p and the uniform draw u"""
    return sample_cands(topk(xs, k), temp, p, u)


def sample_cands(cs, temp, p, u):
    """the draw from the top-k candidates cs [(logit, id)], largest first: llama/main.bend's sample.cs"""
    temp, p, u = F(temp), F(p), F(u)
    if not cs:
        return 0, F(0.0)
    top = cs[0][0]
    z = F(0.0)
    for v, _ in cs:
        z = F(z + weight(v, temp, top))
    lim = F(p * z)
    n, acc = 0, F(0.0)
    for j, (v, _) in enumerate(cs):
        if j > 0 and not acc < lim:
            break
        acc = F(acc + weight(v, temp, top))
        n += 1
    target = F(u * acc)
    acc = F(0.0)
    for j in range(n):
        v, i = cs[j]
        acc = F(acc + weight(v, temp, top))
        if j == n - 1 or target < acc:
            return i, v


def bits(x):
    return int(np.array(x, dtype=np.float32).view(np.uint32))


def from_bits(b):
    return np.array(b, dtype=np.uint32).view(np.float32)[()]
