#!/usr/bin/env python3
"""Generates `src/median_net.rs`: the min/max networks of the 7 x 7 median (rank 24 of 49).

Per output row the 7 window rows are sorted per column (`sort7`); with `c(x)` the sorted column
at `x`, `t3(x)` = the sorted merge of columns x..x+2 (Batcher odd-even merges); the window at `o`
is columns o..o+6 = `t3(o)` + `c(o+3)` + `t3(o+4)`, whose 25th smallest value is the median,
taken as min over i of max(L_i, U_(25 - i)) of the sorted lists L = t3(o) and U = merge(t3(o+4),
c(o+3)). Each stage is a straight-line function over 16-lane `u8` vectors (one lane per output
pixel); constants are folded, equal nodes shared and unused ones dropped. The 0-1 principle
(exhaustive over all sorted 0-1 inputs) and random 8-bit windows check every network before it is
written.

    python3 tools/gen_median_net.py > src/median_net.rs && cargo fmt
"""
import random
import sys

INF, NEG = -1, -2


class Net:
    def __init__(self):
        self.nodes = []  # (kind, a, b); kind: in / min / max
        self.cse = {}

    def inp(self, name):
        self.nodes.append(("in", name, None))
        return len(self.nodes) - 1

    def _op(self, kind, a, b):
        key = (kind, min(a, b), max(a, b))
        if key not in self.cse:
            self.nodes.append(key)
            self.cse[key] = len(self.nodes) - 1
        return self.cse[key]

    def mn(self, a, b):
        if a == INF or a == b:
            return b
        if b == INF:
            return a
        if NEG in (a, b):
            return NEG
        return self._op("min", a, b)

    def mx(self, a, b):
        if a == NEG or a == b:
            return b
        if b == NEG:
            return a
        if INF in (a, b):
            return INF
        return self._op("max", a, b)

    def ce(self, a, b):
        return self.mn(a, b), self.mx(a, b)

    def live(self, outs):
        seen, stack = set(), [o for o in outs if o >= 0]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if self.nodes[n][0] != "in":
                stack += [c for c in self.nodes[n][1:] if c >= 0]
        return seen

    def count(self, outs):
        return sum(1 for n in self.live(outs) if self.nodes[n][0] != "in")

    def evaluate(self, outs, env):
        val = {}
        for n, (k, a, b) in enumerate(self.nodes):
            val[n] = env[a] if k == "in" else (min if k == "min" else max)(val[a], val[b])
        return [val[o] for o in outs]


def merge(net, a, b):
    """Batcher's odd-even merge of two sorted lists of any lengths."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return list(a or b)
    if m == 1 and n == 1:
        return list(net.ce(a[0], b[0]))
    v = merge(net, a[0::2], b[0::2])
    w = merge(net, a[1::2], b[1::2])
    res = [v[0]]
    for i in range(len(w)):
        res += list(net.ce(v[i + 1], w[i])) if i + 1 < len(v) else [w[i]]
    return res + v[len(w) + 1 :]


SORT7 = [
    [(0, 6), (2, 3), (4, 5)],
    [(0, 2), (1, 4), (3, 6)],
    [(0, 1), (2, 5), (3, 4)],
    [(1, 2), (4, 6)],
    [(2, 3), (4, 5)],
    [(1, 2), (3, 4), (5, 6)],
]


def sort7(net, w):
    w = list(w)
    for layer in SORT7:
        for i, j in layer:
            w[i], w[j] = net.ce(w[i], w[j])
    return w


def kth(net, t, u, k):
    """k-th smallest (1-based) of the union of the sorted lists t and u."""
    best = None
    for i in range(len(t) + 1):
        j = k - i
        if 0 <= j <= len(u):
            term = net.mx(NEG if i == 0 else t[i - 1], NEG if j == 0 else u[j - 1])
            best = term if best is None else net.mn(best, term)
    return best


def stage_sort7():
    net = Net()
    r = [net.inp(f"r{i}") for i in range(7)]
    return net, [r], sort7(net, r)


def stage_t3():
    net = Net()
    c = [[net.inp(f"c{x}_{i}") for i in range(7)] for x in range(3)]
    return net, c, merge(net, merge(net, c[0], c[1]), c[2])


def stage_select():
    net = Net()
    t = [[net.inp(f"t{x}_{i}") for i in range(21)] for x in range(2)]
    c = [net.inp(f"c_{i}") for i in range(7)]
    u = merge(net, t[1], c)
    return net, [t[0], t[1], c], [kth(net, t[0], u, 25)]


def check(net, outs, ins, kind):
    """Exhaustive 0-1 inputs (sorted lists of 0s then 1s; all 7-bit patterns for `sort7`), then
    random 8-bit values with many ties."""
    import itertools

    def run(lists):
        env = {net.nodes[i][1]: v for g, vals in zip(ins, lists) for i, v in zip(g, vals)}
        got = net.evaluate(outs, env)
        allv = sorted(sum(lists, []))
        want = [allv[24]] if kind == "select" else allv
        assert got == want, (kind, lists, got, want)

    if kind == "sort7":
        for bits in itertools.product((0, 1), repeat=7):
            run([list(bits)])
    else:
        for cuts in itertools.product(*[range(len(g) + 1) for g in ins]):
            run([[0] * c + [1] * (len(g) - c) for g, c in zip(ins, cuts)])
    rnd = random.Random(1)
    for _ in range(4000):
        lists = []
        for g in ins:
            vals = [rnd.choice([rnd.randrange(256), rnd.randrange(4)]) for _ in g]
            lists.append(vals if kind == "sort7" else sorted(vals))
        run(lists)


def emit(name, doc, net, ins, outs, sig):
    live = net.live(outs)
    lines = [f"/// {doc}", "#[inline(always)]", f"pub fn {name}({sig}) -> {outs_type(len(outs))} {{"]
    for g in ins:
        pass
    ids = {}
    for g in ins:
        for i in g if isinstance(g, list) else [g]:
            ids[i] = net.nodes[i][1]
    body = []
    for n, (k, a, b) in enumerate(net.nodes):
        if n not in live or k == "in":
            continue
        ids[n] = f"n{n}"
        body.append(f"    let n{n} = {k}({ids[a]}, {ids[b]});")
    lines += body
    lines.append("    " + (ids[outs[0]] if len(outs) == 1 else "[" + ", ".join(ids[o] for o in outs) + "]"))
    lines.append("}")
    return "\n".join(lines), net.count(outs)


def outs_type(n):
    return "V" if n == 1 else f"[V; {n}]"


def destructure(prefix, names, count):
    return f"let [{', '.join(f'{prefix}{i}' for i in range(count))}] = {names};"


def main():
    out = [
        "// @generated by tools/gen_median_net.py: do not edit.",
        "//! Min/max networks of the 7 x 7 median, see `tools/gen_median_net.py`.",
        "",
        "#![allow(clippy::all)]",
        "use crate::simd::{max, min, V};",
        "",
    ]
    counts = {}
    # sort7
    net, ins, outs = stage_sort7()
    check(net, outs, ins, "sort7")
    code, cnt = emit("sort7", "Sorts the 7 values of a window column (rows top to bottom).", net, ins, outs, "r: [V; 7]")
    code = code.replace("{\n", "{\n    let [r0, r1, r2, r3, r4, r5, r6] = r;\n", 1)
    out.append(code), out.append("")
    counts["sort7"] = cnt
    # t3
    net, ins, outs = stage_t3()
    check(net, outs, ins, "t3")
    code, cnt = emit("t3", "Sorted merge of three sorted columns (21 values).", net, ins, outs, "c0: [V; 7], c1: [V; 7], c2: [V; 7]")
    pre = "\n".join(
        f"    let [{', '.join(f'c{x}_{i}' for i in range(7))}] = c{x};" for x in range(3)
    )
    code = code.replace("{\n", "{\n" + pre + "\n", 1)
    out.append(code), out.append("")
    counts["t3"] = cnt
    # select
    net, ins, outs = stage_select()
    check(net, outs, ins, "select")
    code, cnt = emit("select", "Median of the window at `o`: `t0` = t3(o), `t1` = t3(o + 4), `c` = c(o + 3).", net, ins, outs, "t0: [V; 21], t1: [V; 21], c: [V; 7]")
    pre = (
        f"    let [{', '.join(f't0_{i}' for i in range(21))}] = t0;\n"
        f"    let [{', '.join(f't1_{i}' for i in range(21))}] = t1;\n"
        f"    let [{', '.join(f'c_{i}' for i in range(7))}] = c;"
    )
    code = code.replace("{\n", "{\n" + pre + "\n", 1)
    out.append(code), out.append("")
    counts["select"] = cnt
    print("\n".join(out))
    print(f"// min/max ops per 16 pixels: {counts} (total {sum(counts.values())})")


main()
