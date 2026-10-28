"""Wave assignment of one pure-map layer: which dolls may be released together.

A layer's dolls wait at a sync barrier and a clock doll opens one barrier per wave, `S` map units apart. Two dolls of
one wave cross their effect lines in the same tic, and the engine drops the second of two plane movers aimed at one
sector in one tic. The dolls of a wave must therefore not share a *plane* target (a floor or ceiling that a later layer
reads as a gate, or an action special) unless they can never both reach an effect line.

`assign` colours the conflict graph of a layer: dolls are vertices, two dolls are adjacent when they share a plane
target and are not mutually exclusive. Colour classes are waves. The relaxations, each with its reason:

  exclusive  two dolls whose gates demand disjoint intervals of one value can never both pass (one-hot lookups, the
             plus and minus dolls of a signed read, equality dolls), so they need no separation.
  lights     a light-level target takes every same-tic hit (test_light_hits_in_one_tic_are_not_lost), so only planes
             and actions are targets that conflict.
  colour     DSATUR on the conflict graph instead of first fit; a layer whose conflict graph is the line graph of a
             complete graph (a round-robin tournament: every pair of k scores compared once) is coloured by the
             circle method, which uses k - 1 colours for even k (k for odd k) where first fit needs about 1.5 k.
"""
from __future__ import annotations

import heapq
from collections.abc import Sequence

Box = dict[str, tuple[int, int]]       # value name -> [lo, hi] the value must lie in for a doll to reach an effect


def gate_interval(port_value: str, kind: str, limit: int) -> tuple[str, int, int]:
    """(value, lo, hi): the values of `value` for which a gate on the port carrying `port_value` (possibly '-v') opens.
    A floor gate opens iff port <= limit, a ceiling gate iff port >= limit (tile.Gate)."""
    neg = port_value.startswith("-")
    var = port_value.lstrip("-")
    big = 10 ** 9
    if kind == "floor":
        return (var, -limit, big) if neg else (var, -big, limit)
    return (var, -big, -limit) if neg else (var, limit, big)


def reach_box(spec, lane_gate, block, skip: set[str]) -> Box:
    """Necessary conditions for a doll to reach ANY effect line: its lane gate and every stage gate up to and including
    the first stage that has a nonzero effect. Values in `skip` (written by this very layer) are left out."""
    box: Box = {}

    def add(g):
        var, lo, hi = gate_interval(spec.ports[g.port].split(":", 1)[1], g.kind, g.limit)
        if var in skip:
            return
        a, b = box.get(var, (-10 ** 9, 10 ** 9))
        box[var] = (max(a, lo), min(b, hi))

    add(lane_gate)
    for st in block.stages:
        if st.gate is not None:
            add(st.gate)
        if any(e.w for e in st.effects):
            break
    return box


def exclusive(a: Box, b: Box) -> bool:
    """True when some value has disjoint intervals in the two boxes (no input makes both dolls pass)."""
    if len(b) < len(a):
        a, b = b, a
    for var, (lo, hi) in a.items():
        other = b.get(var)
        if other is not None and (lo > other[1] or other[0] > hi):
            return True
    return False


def _components(n: int, groups: list[tuple[int, ...]]) -> list[list[int]]:
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for g in groups:
        r = find(g[0])
        for i in g[1:]:
            parent[find(i)] = r
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return [c for c in comps.values()]


def _adjacency(members: list[int], groups: list[tuple[int, ...]], boxes: Sequence[Box | None]) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {i: set() for i in members}
    for g in groups:
        for x, u in enumerate(g):
            bu = boxes[u]
            for v in g[x + 1:]:
                if v in adj[u]:
                    continue
                if bu is not None and boxes[v] is not None and exclusive(bu, boxes[v]):
                    continue
                adj[u].add(v)
                adj[v].add(u)
    return adj


def dsatur(adj: dict[int, set[int]]) -> dict[int, int]:
    """DSATUR (Brelaz 1979): colour next the vertex with the most distinct neighbour colours (ties: most uncoloured
    neighbours, then lowest index), with the lowest colour no neighbour has. Exact on cliques, bipartite graphs, cycles
    and wheels; a heuristic elsewhere."""
    colour: dict[int, int] = {}
    seen: dict[int, set[int]] = {v: set() for v in adj}
    free = {v: len(a) for v, a in adj.items()}
    heap = [(0, -free[v], v) for v in adj]
    heapq.heapify(heap)
    while heap:
        neg_sat, _, v = heapq.heappop(heap)            # heapq pops the smallest tuple: keys are pushed negated
        if v in colour or -neg_sat != len(seen[v]):
            continue
        c = 0
        while c in seen[v]:
            c += 1
        colour[v] = c
        for u in adj[v]:
            if u in colour:
                continue
            free[u] -= 1
            seen[u].add(c)
            heapq.heappush(heap, (-len(seen[u]), -free[u], u))
    return colour


def circle_colours(k: int) -> dict[tuple[int, int], int]:
    """Proper edge colouring of the complete graph K_k by the circle method (a round-robin tournament): k - 1 colours
    (rounds) for even k, k for odd k, the minimum possible. Vertex k' = k (odd k) is a bye."""
    n = k + (k % 2)
    m = n - 1
    out = {}
    for i in range(k):
        for j in range(i + 1, k):
            if j == n - 1:
                out[i, j] = (2 * i) % m
            else:
                out[i, j] = (i + j) % m
    return out


def _round_robin(members: list[int], hits: Sequence[frozenset]) -> dict[int, int] | None:
    """Circle-method colours when the members' conflict structure is exactly the line graph of a complete graph: every
    member has two targets, each pair of targets occurs once, and there are k (k - 1) / 2 members for k targets."""
    targets: dict = {}
    pair_of = {}
    for i in members:
        h = hits[i]
        if len(h) != 2:
            return None
        a, b = sorted(h, key=repr)
        pair_of[i] = (a, b)
        for t in (a, b):
            targets.setdefault(t, len(targets))
    k = len(targets)
    if len(members) != k * (k - 1) // 2 or len(set(pair_of.values())) != len(members):
        return None
    colours = circle_colours(k)
    out = {}
    for i, (a, b) in pair_of.items():
        x, y = sorted((targets[a], targets[b]))
        out[i] = colours[x, y]
    return out


def first_fit(hits: Sequence[frozenset], boxes: Sequence[Box | None]) -> list[int]:
    """Doll order, lowest wave whose dolls (on every target of this doll) are all exclusive with it."""
    used: dict[tuple, dict[int, list[int]]] = {}
    out = []
    for i, mine in enumerate(hits):
        slot = 0
        while True:
            bad = False
            for t in mine:
                for j in used.get(t, {}).get(slot, ()):
                    if boxes[i] is None or boxes[j] is None or not exclusive(boxes[i], boxes[j]):
                        bad = True
                        break
                if bad:
                    break
            if not bad:
                break
            slot += 1
        out.append(slot)
        for t in mine:
            used.setdefault(t, {}).setdefault(slot, []).append(i)
    return out


def lower_bound(adj: dict[int, set[int]]) -> int:
    """Size of a clique found greedily (vertices by decreasing conflict degree): the layer needs at least this many
    waves. It is a bound, not the chromatic number: a graph can need more."""
    clique: list[int] = []
    for u in sorted(adj, key=lambda v: (-len(adj[v]), v)):
        if all(u in adj[w] for w in clique):
            clique.append(u)
    return max(1, len(clique))


def assign(hits: Sequence[frozenset], boxes: Sequence[Box | None], colour: bool) -> tuple[list[int], dict]:
    """Wave of every doll (list aligned with `hits`) and a report.

    hits[i]: the conflicting targets of doll i (planes, actions; not lights when the lights relaxation is on).
    boxes[i]: reach_box of doll i, or None to disable exclusivity for it.
    colour: False = first fit in doll order (the greedy scheduler, with exclusion); True = DSATUR per connected
    component, replaced by the circle method on a component that is a complete round robin.
    Components share no target, so each is coloured from 0 and the layer needs the maximum of their colour counts.
    report: waves, lower_bound (clique bound), conflicts (edges), components, round_robin (components coloured by the
    circle method)."""
    n = len(hits)
    by_target: dict = {}
    for i, h in enumerate(hits):
        for t in h:
            by_target.setdefault(t, []).append(i)
    groups = sorted({tuple(v) for v in by_target.values() if len(v) > 1})
    if not colour:
        slot = first_fit(hits, boxes)
        return slot, dict(waves=max(slot, default=-1) + 1, lower_bound=None, conflicts=None, components=None,
                          round_robin=0)
    slot = [0] * n
    comps = [c for c in _components(n, groups) if len(c) > 1]
    comp_groups: dict[int, list[tuple[int, ...]]] = {}
    index = {i: k for k, c in enumerate(comps) for i in c}
    for g in groups:
        comp_groups.setdefault(index[g[0]], []).append(g)
    edges, bound, circle = 0, 1, 0
    for k, members in enumerate(comps):
        adj = _adjacency(members, comp_groups.get(k, []), boxes)
        edges += sum(len(a) for a in adj.values()) // 2
        col = dsatur(adj)
        rr = _round_robin(members, hits)
        if rr is not None and max(rr.values()) < max(col.values()):
            col, circle = rr, circle + 1
        for i, c in col.items():
            slot[i] = c
        bound = max(bound, lower_bound(adj))
    return slot, dict(waves=max(slot, default=-1) + 1, lower_bound=bound, conflicts=edges, components=len(comps),
                      round_robin=circle)
