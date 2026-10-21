"""Pure DOOM: several layers chained inside ONE map. No bus, no Python between layers.

In tile mode (tile.py) every tile ends in light-level accumulators that Python copies into the next tile's gate
heights. Here a layer's weight lines move LATER layers' gate floors/ceilings directly:

  * value as plane height: each consumed value is a port (a tag on gate sectors). A floor port's floor, or a
    ceiling port's ceiling, IS the value. Producing dolls cross lines running Floor_/Ceiling_Raise/LowerByValue
    (port tag, MOVE_SPEED, |w|); the plane starts at the value's bias (baked into the map) and ends at
    bias + sum.  Ceiling-gate sectors have a pit floor (FLOOR_PIT, bridged by the 32-wide dolls) so a moving
    ceiling can never meet its own floor.
  * serialisation: DOOM drops same-tic moves into the same sector. Greedy colouring assigns dolls to
    nonconflicting waves. Their effect cells are aligned within each connected crossbar; gated corridors
    use wave spacing longer than the entire corridor.
  * sync barriers: gate arrivals jitter by ~8 tics. Passing dolls wait at a second barrier until the clock
    releases their wave. They restart from rest; delay is in the clock, NOT in each arithmetic lane.
  * clock dolls: one per layer crosses one lowering line per wave, then freezes the finished layer and
    starts the next. Clock corridors fold via Teleport_NoFog, with 64-unit clearance after every special
    before a teleport so same-tic teleport truncation cannot discard releases. No ACS arithmetic or host
    scheduling occurs between layers.
  * state: values listed in `state` start at their initial height, may be read by a layer BEFORE the layer that
    updates them, and survive the per-run reset. That is how a map learns: weights are ceiling heights and a
    learning layer's dolls raise and lower them.
  * outputs nobody consumes accumulate into light levels; `readout` names expose plane heights as well.
  * layout: individual lanes fold into rows in x=[X_ORIGIN, X_END] = [-16000, 16000], stacked north from y=0 up to
    Y_END; output lights sit below y=-512 and the scroller closets at CLOSET. The whole map, including the world,
    must span <= MAX_SPAN units per axis: ViZDoom's 16.16 coordinate differences wrap beyond 32,767 units, and
    then sector lookups (point-in-subsector) go wrong (docs/engine_limits.md). A model
    whose layout exceeds that fails (fits=False; building raises). Independent crossbars omit one another's unused
    effect cells. Reset destinations are disjoint from all doll TIDs.
    `layout_only` runs the same geometry emitter into a counting sink, reporting exact sector/doll counts
    and hypothetical bounds even if the layout would overflow. It emits no WAD and is not an engine test.
Python writes the input ports, drops layer 0's barriers (TileRunner.start) and reads the results after
everything has settled; nothing in between. Between runs, ACS script 3 rewinds the machine (dolls home,
barriers up, non-state values back to their biases); state values are left alone.

Blocks: lanes are floor gates; a block's stages are a corridor of optional gates (floor or ceiling) each followed
by effect lines, or (crossbars) one ungated stage.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .tile import (CEIL, CLOSED, CONV_TAG, CONVEYOR, DOLL_TID0, FX_LEN, GATE_LEN, HOME_TID0, LANE_W, MAX_SPAN,
                   MOTION_SCRIPT, MOVE_SPEED, OUT_TAG0, PAGE, POST_LEN, PORT_TAG0, STAIR_LEN, START_TAG, STEP_UP,
                   Block, Gate, TileMeta, TileSpec, motion_acs)
from .udmf import MapBuilder, compile_acs, write_wad

SERIAL = 64            # clock distance between crossbar wave releases (>= 4 conveyor tics)
SYNC_WAIT = 400        # clock line that drops the sync barriers, this far past them (~25 tics)
CLOCK_MARGIN = 512     # next-layer line beyond the last possible effect cell (dolls restart from rest)
STAGE_TAG0 = 60000     # start barriers of layer s >= 1 (layer 0 uses START_TAG, dropped by ACS script 7)
FREEZE_TAG0 = 64000    # freeze barriers (one tag per layer and walk-floor height)
FLOOR_PIT = -15000     # floor of ceiling-gate sectors
FLOOR_RAISE, FLOOR_LOWER, LOWER_TO_LOWEST, LIGHT_RAISE, LIGHT_LOWER = 23, 20, 21, 110, 111
CEIL_RAISE, CEIL_LOWER, EXEC_ALWAYS, FLOOR_MOVE_TO = 41, 40, 226, 37
# lane geometry (x offsets inside a layer):
#   home | home2 | start barrier | pad | lane gate | post | transition | wave sync barrier | corridor
X_HOME2, X_BARRIER, X_PAD, X_GATE, X_POST, X_TRANS = 24, 48, 56, 96, 104, 160
X_ORIGIN, Y_ORIGIN = -16000, 0  # layer rows run east from X_ORIGIN to X_END, stacked north from Y_ORIGIN up to Y_END;
X_END = X_ORIGIN + MAX_SPAN     # rooms, lights and closets live at y < 0
CLOSET = (-4096, -4096)  # conveyor-scroller closets (MapBuilder.static_specials), 96 units apart eastwards
Y_END = CLOSET[1] + MAX_SPAN  # rows stop here, so the brain and its closets span <= MAX_SPAN
WORLD_TOP = -2048      # the world builds at y <= WORLD_TOP (south of the brain and its lights)
LAND_LEN = 128         # extra post after a floor gate so the doll lands before the next gate
FREEZE = True          # freeze finished layers (False only for the negative control in verify_learn.py)


def stage_tag(s: int) -> int:
    return START_TAG if s == 0 else STAGE_TAG0 + s


def _stairs(floor: int, target: int) -> int:
    """Number of 24-unit risers needed to climb from floor to target (the last step may be shorter)."""
    return max(0, -(-(target - floor) // STEP_UP) - 1)


def map_bounds(m: MapBuilder) -> list[int]:
    """[x0, y0, x1, y1] of m's sectors plus the conveyor closet compile_pure adds after the world."""
    cx, cy = m.closet_origin
    k = len(m.wall_props)
    return [min(min(r.x0 for r in m.rects), cx + 96 * k), min(min(r.y0 for r in m.rects), cy),
            max(max(r.x1 for r in m.rects), cx + 96 * k + 64), max(max(r.y1 for r in m.rects), cy + 64)]


def world_area(bounds: list[int]) -> list[int] | None:
    """[x0, y0, x1, y1]: where world sectors may go (y <= WORLD_TOP) so that a map whose other sectors have these
    bounds stays within MAX_SPAN per axis; the MAX_SPAN-wide window nearest [X_ORIGIN, X_END]. None if no room."""
    bx0, by0, bx1, by1 = bounds
    x0 = min(max(X_ORIGIN, bx1 - MAX_SPAN), bx0)
    y0, y1 = by1 - MAX_SPAN, min(WORLD_TOP, by0 + MAX_SPAN)
    if bx1 - bx0 > MAX_SPAN or y1 <= y0:
        return None
    return [x0, y0, x0 + MAX_SPAN, y1]


class _GeometryCounter(MapBuilder):
    """Count the actual emitter's geometry without storing millions of rectangles."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sector_count = self.line_count = self.thing_count = 0

    def rect(self, x0, y0, x1, y1, **kwargs):
        self.sector_count += 1
        if not self.rects:
            super().rect(x0, y0, x1, y1, **kwargs)
        else:
            r = self.rects[0]
            r.x0, r.y0 = min(r.x0, x0), min(r.y0, y0)
            r.x1, r.y1 = max(r.x1, x1), max(r.y1, y1)
        return self.sector_count - 1

    def link(self, *args, **kwargs):
        self.line_count += 1

    def thing(self, *args, **kwargs):
        self.thing_count += 1


def compile_pure(stages: list[TileSpec], wad: Path, name: str = "pure", state: dict[str, int] | None = None,
                 readout: list[str] | None = None, serial: dict[int, int] | None = None,
                 world: Callable[[MapBuilder], str] | None = None,
                 actions: dict[str, list[dict]] | None = None, *, layout_only: bool = False,
                 extra_lumps: list[tuple[str, bytes]] | None = None) -> TileMeta:
    """Compile layers into one map.

    stages: TileSpecs; a port may name an earlier layer's output, a state value, or (otherwise) an input.
    state: value name -> initial value; persistent across runs (pass both "v" and "-v" if both are used).
    readout: value names whose plane height Python reads (besides the light outputs).
    serial: per-layer override of clock wave spacing S (0 releases all dolls together; negative control).
    world: builds the real player's surroundings inside layout["world_area"] (y <= WORLD_TOP; see world_area) and
    must place the LAST player-1 start, so the dolls are his voodoo dolls; returns extra ACS (script numbers >= 200).
    Default: an empty closet. A map (world included) wider or taller than MAX_SPAN raises ValueError.
    actions: value name -> extra line specials (dicts with special/arg0..arg4) crossed by every doll that adds a
    positive weight to that value, e.g. ThrustThing on the player's TID: an output neuron's spike moves his body.
    layout_only: count actual emitted geometry without allocating the map or compiling ACS; may exceed bounds
    (then layout["fits"] is False)."""
    # Optional global resources (e.g. a script-triggered weapon) precede MAP01.
    state = dict(state or {})
    serial = dict(serial or {})
    produced: dict[str, tuple[int, int]] = {}            # output name -> (stage, bias)
    for s, spec in enumerate(stages):
        for oname, bias in spec.outputs:
            if oname not in state:
                produced[oname] = (s, bias)
    tags: dict[str, int] = {}                             # port key "kind:value" -> tag
    consumers: dict[str, list[tuple[str, int]]] = {}      # value name -> [(kind, tag)]
    inputs: dict[str, int] = {}
    init: dict[str, int] = {}                             # port key -> initial plane height (internal ports)
    for s, spec in enumerate(stages):
        for key in spec.ports:
            kind, val = key.split(":", 1)
            if key in tags:
                continue
            tags[key] = PORT_TAG0 + len(tags)
            if val in state:
                init[key] = state[val]
            elif val in produced:
                if produced[val][0] >= s:
                    raise ValueError(f"{key} in layer {s} is produced by layer {produced[val][0]}")
                init[key] = produced[val][1]
            else:
                inputs[key] = tags[key]
                continue
            consumers.setdefault(val, []).append((kind, tags[key]))
    lights = [o for spec in stages for o, _ in spec.outputs if o not in consumers and o not in state]
    out_start = max(OUT_TAG0, PORT_TAG0 + len(tags))
    if out_start <= STAGE_TAG0 < out_start + len(lights):
        out_start = STAGE_TAG0 + len(stages) + 1
    out_tag = {o: out_start + i for i, o in enumerate(lights)}
    light_bias = {o: b for spec in stages for o, b in spec.outputs if o in out_tag}

    action_list: list[dict] = []
    action_of: dict[str, list[int]] = {}
    for oname, specials in (actions or {}).items():
        action_of[oname] = list(range(len(action_list), len(action_list) + len(specials)))
        action_list.extend(specials)

    def targets(oname: str) -> list[tuple[str, int]]:
        return (consumers.get(oname, []) + ([("light", out_tag[oname])] if oname in out_tag else [])
                + [("action", k) for k in action_of.get(oname, [])])

    def effect_line(kind: str, tag: int, w: int) -> dict:
        common = dict(playercross=True, repeatspecial=True)
        if kind == "action":                     # only positive contributions trigger the action
            return {**action_list[tag], **common} if w > 0 else {}
        if kind == "light":
            return dict(special=LIGHT_RAISE if w > 0 else LIGHT_LOWER, arg0=tag, arg1=abs(w), **common)
        if kind == "floor":
            return dict(special=FLOOR_RAISE if w > 0 else FLOOR_LOWER, arg0=tag, arg1=MOVE_SPEED, arg2=abs(w),
                        **common)
        return dict(special=CEIL_RAISE if w > 0 else CEIL_LOWER, arg0=tag, arg1=MOVE_SPEED, arg2=abs(w), **common)

    freeze_tags: dict[tuple[int, int], int] = {}
    next_tag = max(FREEZE_TAG0, PORT_TAG0 + len(tags), out_start + len(lights)) + len(stages) + 1

    def fresh_tag():
        nonlocal next_tag
        tag = next_tag
        next_tag += 1
        return tag

    def freeze_rect(s: int, floor: int, x0, y0, x1, y1, group):
        """Flush walkway until the layer is frozen; blocks late gate crossings."""
        if (s, floor) not in freeze_tags:
            freeze_tags[s, floor] = fresh_tag()
        return m.rect(x0, y0, x1, y1, floor=floor, ceil=CEIL, group=group,
                      tag=freeze_tags[s, floor], light=176, ftex="FLAT23")

    def gate_rect(gate: Gate, key: str, x0, y0, x1, y1, group):
        h = init.get(key)
        if gate.kind == "floor":
            return m.rect(x0, y0, x1, y1, floor=CLOSED if h is None else h, ceil=CEIL, group=group,
                          tag=tags[key], light=224, ftex="FLAT23", wall="METAL")
        return m.rect(x0, y0, x1, y1, floor=FLOOR_PIT, ceil=gate.walk_floor if h is None else h, group=group,
                      tag=tags[key], light=224, ftex="FLAT23", ctex="FLAT23", wall="METAL")

    def post_len(gate: Gate) -> int:
        # a doll crosses a floor gate ON the port, up to 24 above its walkway, then falls back down, and DOOM's step
        # check uses its current height. Measured (build/measure_fall.py): off a port at z=-1 a doll falls -3, -6,
        # -10, -15, -21, -25 (6 tics, ~70 units); with an 80-unit gap it met the next gate at z=-15 and stepped 17 up
        # onto a port that blocks from the walkway (26 up). LAND_LEN lets it land first.
        return POST_LEN + (LAND_LEN if gate.kind == "floor" else 0)

    def corridor_len(block: Block, floor: int) -> int:
        x = 0
        for st in block.stages:
            if st.gate is not None:
                target = st.gate.walk_floor
                if target < floor:
                    raise ValueError(f"{name}: corridor would descend {floor}->{target}")
                x += _stairs(floor, target) * STAIR_LEN + STAIR_LEN + 2 * GATE_LEN + post_len(st.gate)
                floor = target
            x += FX_LEN * sum(len(targets(spec_out[e.out])) for e in st.effects if e.w)
        return x

    m = (_GeometryCounter if layout_only else MapBuilder)(closet_origin=CLOSET)
    group, n_dolls = 0, 0
    pack_x, pack_y = X_ORIGIN, Y_ORIGIN
    extents = []
    stage_stats = []

    def place(width):
        nonlocal pack_x, pack_y
        if width > X_END - X_ORIGIN:
            raise ValueError(f"{name}: corridor width {width} exceeds {X_END - X_ORIGIN} (map span limit "
                             f"{MAX_SPAN} per axis)")
        if pack_x + width > X_END:
            pack_x, pack_y = X_ORIGIN, pack_y + LANE_W + 8
        if pack_y + LANE_W > Y_END and not layout_only:
            raise ValueError(f"{name}: folded layout exceeds y={Y_END} in layer {s} ({spec.name}), i.e. the map "
                             f"would span more than {MAX_SPAN} units per axis; emitted sectors={len(m.rects)}, "
                             f"planned dolls={total_dolls}")
        xy = pack_x, pack_y
        pack_x += width + 8
        return xy
    total_dolls = sum(spec.n_dolls + 1 for spec in stages)
    home_tid0 = max(HOME_TID0, DOLL_TID0 + total_dolls + 1)
    clock_tid = home_tid0 + total_dolls + 1
    resets: list[str] = []
    barrier_tags: list[int] = []
    for s, spec in enumerate(stages):
        spec_out = [o for o, _ in spec.outputs]
        crossbar = all(len(b.stages) == 1 and b.stages[0].gate is None for b in spec.blocks if b.lanes)
        # every (output, target) pair a crossbar layer feeds gets one cell slot, identical for all dolls
        slots = [(o, k, t) for o in spec_out for k, t in targets(o)] if crossbar else []
        dolls = []
        for block in spec.blocks:
            for gate in block.lanes:
                if gate.kind != "floor":
                    raise NotImplementedError("lanes must be floor gates")
                dolls.append((gate, spec.ports[gate.port], block))
        if not dolls:
            raise ValueError(f"{name}: layer {s} has no dolls")

        def first_floor(gate, block):
            st0 = block.stages[0]
            return st0.gate.walk_floor if (not crossbar and st0.gate is not None) else gate.walk_floor

        trans = max(_stairs(g.walk_floor, first_floor(g, b)) * STAIR_LEN for g, _, b in dolls) + 64
        x_sync = X_TRANS + trans
        x_run = x_sync + GATE_LEN
        # offsets only have to separate dolls whose effects can hit the same sector: greedy colouring
        hits = []
        for _, _, block in dolls:
            outs = {spec_out[e.out] for st in block.stages for e in st.effects if e.w}
            hits.append({t for o in outs for _, t in targets(o)})
        # Independent crossbars need only their own aligned effect cells, not
        # empty cells for every other token position in a transformer layer.
        parents = list(range(len(dolls)))

        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        owners = {}
        if crossbar:
            for i, mine in enumerate(hits):
                for target in mine:
                    if target in owners:
                        parents[root(i)] = root(owners[target])
                    else:
                        owners[target] = i
            component_slots = {}
            for item in slots:
                if item[2] in owners:
                    component_slots.setdefault(root(owners[item[2]]), []).append(item)
            lane_slots = [component_slots.get(root(i), []) for i in range(len(dolls))]
            spans = [len(items) * FX_LEN for items in lane_slots]
        else:
            spans = [corridor_len(b, first_floor(g, b)) for g, _, b in dolls]
        span = max(spans)
        S = serial.get(s, SERIAL if crossbar else span + 64)
        slot_of: list[int] = []
        used_slots: dict[int, set[int]] = {}
        for mine in hits:
            taken = set().union(*(used_slots.get(t, set()) for t in mine))
            slot = 0
            while slot in taken:
                slot += 1
            slot_of.append(slot)
            for t in mine:
                used_slots.setdefault(t, set()).add(slot)
        # S=0 intentionally releases conflicting dolls together (negative control).
        if S == 0:
            slot_of = [0] * len(dolls)
        wave_tags = [fresh_tag() for _ in range(max(slot_of) + 1)]
        btag = stage_tag(s)
        barrier_tags += [btag, *wave_tags]
        widths = [x_run + 1 + length + 64 for length in spans]
        locations = [place(width) for width in widths]
        extents.append([min(x for x, _ in locations), min(y for _, y in locations),
                        max(x + width for (x, _), width in zip(locations, widths)),
                        max(y + LANE_W for _, y in locations)])
        stage_stats.append(dict(name=spec.name, dolls=len(dolls), waves=len(wave_tags),
                                corridor_width=max(widths), wave_spacing=S,
                                lane_area=sum(widths) * LANE_W))
        for g, (gate, port_key, block) in enumerate(dolls):
            group += 1
            x, y0 = locations[g]
            y1 = y0 + LANE_W
            f = gate.walk_floor
            stag = wave_tags[slot_of[g]]
            strip_end = widths[g]
            if crossbar:
                slots = lane_slots[g]
            kw = dict(ceil=CEIL, group=group, tag=CONV_TAG)
            home = m.rect(x, y0, x + X_HOME2, y1, floor=f, light=192, ftex="FLOOR4_8", **kw)
            home2 = m.rect(x + X_HOME2, y0, x + X_BARRIER, y1, floor=f, light=192, ftex="FLOOR4_8", **kw)
            m.link(home, home2, special=EXEC_ALWAYS, arg0=4, arg1=0, arg2=DOLL_TID0 + n_dolls, playercross=True,
                   repeatspecial=True)                                      # the doll tags itself (for resets)
            m.rect(x + X_BARRIER, y0, x + X_PAD, y1, floor=CLOSED, ceil=CEIL, group=group, tag=btag, light=224,
                   ftex="FLAT23", wall="SUPPORT3")
            m.rect(x + X_PAD, y0, x + X_GATE, y1, floor=f, light=192, ftex="FLOOR4_8", **kw)
            gate_rect(gate, port_key, x + X_GATE, y0, x + X_POST, y1, group)
            freeze_rect(s, f, x + X_POST, y0, x + X_POST + GATE_LEN, y1, group)
            m.rect(x + X_POST + GATE_LEN, y0, x + X_TRANS, y1, floor=f, light=176, **kw)
            # transition: climb (or drop) to the corridor floor before the sync barrier, where timing is free
            c0 = first_floor(gate, block)
            cx, floor = x + X_TRANS, f
            for _ in range(_stairs(f, c0)):
                floor += STEP_UP
                m.rect(cx, y0, cx + STAIR_LEN, y1, floor=floor, light=176, **kw)
                cx += STAIR_LEN
            m.rect(cx, y0, x + x_sync, y1, floor=c0, light=176, **kw)
            m.rect(x + x_sync, y0, x + x_run, y1, floor=CLOSED, ceil=CEIL, group=group, tag=stag, light=224,
                   ftex="FLAT23", wall="SUPPORT3")
            prev = m.rect(x + x_run, y0, x + x_run + 1, y1, floor=c0, light=176, **kw)
            cx, floor = x + x_run + 1, c0
            covered = cx                                  # floor is built up to here; empty cells are merged

            def cover():
                nonlocal prev, covered
                if covered < cx:
                    prev = m.rect(covered, y0, cx, y1, floor=floor, light=200, ftex="FLAT1_1", **kw)
                    covered = cx

            def cell(kind, tag, w):
                nonlocal prev, cx, covered
                if not w:                                 # crossbar slot without a line: merged into one sector
                    cx += FX_LEN
                    return
                if crossbar:
                    cover()
                cur = m.rect(cx, y0, cx + FX_LEN, y1, floor=floor, light=200, ftex="FLAT1_1", **kw)
                m.link(prev, cur, **effect_line(kind, tag, w))
                prev, cx, covered = cur, cx + FX_LEN, cx + FX_LEN

            if crossbar:
                effects = {spec_out[e.out]: e.w for e in block.stages[0].effects if e.w}
                for oname, tkind, ttag in slots:
                    cell(tkind, ttag, effects.get(oname, 0))
                cover()
            else:
                for st in block.stages:
                    if st.gate is not None:
                        target = st.gate.walk_floor
                        for _ in range(_stairs(floor, target)):
                            floor += STEP_UP
                            prev = m.rect(cx, y0, cx + STAIR_LEN, y1, floor=floor, light=176, **kw)
                            cx += STAIR_LEN
                        floor = target
                        m.rect(cx, y0, cx + STAIR_LEN, y1, floor=floor, light=176, **kw)
                        gate_rect(st.gate, spec.ports[st.gate.port], cx + STAIR_LEN, y0,
                                  cx + STAIR_LEN + GATE_LEN, y1, group)
                        prev = freeze_rect(s, floor, cx + STAIR_LEN + GATE_LEN, y0, cx + STAIR_LEN + 2 * GATE_LEN,
                                           y1, group)
                        cx += STAIR_LEN + 2 * GATE_LEN
                    for e in st.effects:
                        if e.w:
                            for tkind, ttag in targets(spec_out[e.out]):
                                cell(tkind, ttag, e.w)
                    if st.gate is not None:
                        prev = m.rect(cx, y0, cx + post_len(st.gate), y1, floor=floor, light=176, **kw)
                        cx += post_len(st.gate)
            m.rect(cx, y0, max(cx + 64, x + strip_end), y1, floor=floor, light=160, **kw)     # runout
            m.thing(x + 16, (y0 + y1) // 2, 1)                                # voodoo doll
            m.thing(x + 16, (y0 + y1) // 2, 14, id=home_tid0 + n_dolls)        # reset destination
            resets.append(f"TeleportOther({DOLL_TID0 + n_dolls}, {home_tid0 + n_dolls}, 0);")
            n_dolls += 1
        # One clock doll visits a folded chain of short corridors. Every wave
        # opens a dedicated sync barrier, so passing dolls start from rest;
        # the ADC-gate arrival jitter cannot erode their separation.
        # A teleport is at least 64 units AFTER a release/freeze line, never
        # sharing its tic (Teleport_NoFog would truncate same-tic specials).
        chunk_size = max(1, 12000 // max(S, SERIAL))
        chunks = [wave_tags[i:i + chunk_size] for i in range(0, len(wave_tags), chunk_size)]
        destinations = list(range(clock_tid, clock_tid + len(chunks)))
        clock_tid += len(chunks)
        for chunk, releases in enumerate(chunks):
            group += 1
            first, last = chunk == 0, chunk == len(chunks) - 1
            release = x_sync + SYNC_WAIT if first else max(S, SERIAL) + 16
            final_release = release + (len(releases) - 1) * S
            freeze = [(t, fl) for (fs, fl), t in freeze_tags.items() if fs == s and FREEZE]
            finish = final_release + span + CLOCK_MARGIN if last else final_release + 64
            width = finish + (len(freeze) * FX_LEN + 128 if last else 64)
            x, y0 = place(width)
            y1 = y0 + LANE_W
            kw = dict(ceil=CEIL, group=group, tag=CONV_TAG)
            if first:
                home = m.rect(x, y0, x + X_HOME2, y1, floor=0, light=255, **kw)
                home2 = m.rect(x + X_HOME2, y0, x + X_BARRIER, y1, floor=0, light=255, **kw)
                m.link(home, home2, special=EXEC_ALWAYS, arg0=4, arg1=0, arg2=DOLL_TID0 + n_dolls,
                       playercross=True, repeatspecial=True)
                m.rect(x + X_BARRIER, y0, x + X_PAD, y1, floor=CLOSED, ceil=CEIL,
                       group=group, tag=btag, light=224)
                start = X_PAD
                m.thing(x + 16, (y0 + y1) // 2, 1)
                m.thing(x + 16, (y0 + y1) // 2, 14, id=home_tid0 + n_dolls)
                resets.append(f"TeleportOther({DOLL_TID0 + n_dolls}, {home_tid0 + n_dolls}, 0);")
                n_dolls += 1
            else:
                start = 0
                m.thing(x + 16, (y0 + y1) // 2, 14, id=destinations[chunk])
            before = m.rect(x + start, y0, x + release, y1, floor=0, light=255, **kw)
            for i, wave_tag in enumerate(releases):
                end_x = release + S if i + 1 < len(releases) else finish
                after = m.rect(x + release, y0, x + end_x, y1, floor=0, light=255, **kw)
                m.link(before, after, special=LOWER_TO_LOWEST, arg0=wave_tag, arg1=MOVE_SPEED,
                       playercross=True, repeatspecial=True)
                before, release = after, end_x
            if not last:
                end = m.rect(x + finish, y0, x + width, y1, floor=0, light=255, **kw)
                m.link(after, end, special=71, arg0=destinations[chunk + 1],
                       playercross=True, repeatspecial=True)
            else:
                prev, cx = after, x + finish
                if s + 1 < len(stages):
                    for t, _ in freeze:
                        cur = m.rect(cx, y0, cx + FX_LEN, y1, floor=0, light=255, **kw)
                        m.link(prev, cur, special=FLOOR_MOVE_TO, arg0=t, arg1=MOVE_SPEED,
                               arg2=CLOSED, arg3=0, playercross=True, repeatspecial=True)
                        prev, cx = cur, cx + FX_LEN
                    end = m.rect(cx, y0, x + width, y1, floor=0, light=255, **kw)
                    m.link(prev, end, special=LOWER_TO_LOWEST, arg0=stage_tag(s + 1), arg1=MOVE_SPEED,
                           playercross=True, repeatspecial=True)
                else:
                    m.rect(cx, y0, x + width, y1, floor=0, light=255, **kw)
    for oi, oname in enumerate(lights):
        ox, oy = (oi % 64) * 72, -512 - (oi // 64) * 72
        m.rect(ox, oy, ox + 64, oy + 64, floor=0, ceil=128, light=light_bias[oname], tag=out_tag[oname],
               ftex="FLAT14", wall="COMPBLUE")
    area = world_area(map_bounds(m))
    if world is None:
        m.rect(-768, -512, -512, -256, floor=0, ceil=128)
        m.thing(-640, -384, 1)                                                # last player-1 start = player
        world_acs = ""
    else:
        world_acs = world(m)
    m.scroll_carry(CONV_TAG, **CONVEYOR)
    bounds = [min(r.x0 for r in m.rects), min(r.y0 for r in m.rects),
              max(r.x1 for r in m.rects), max(r.y1 for r in m.rects)]
    span = [bounds[2] - bounds[0], bounds[3] - bounds[1]]
    fits = max(span) <= MAX_SPAN
    if not fits and not layout_only:
        raise ValueError(f"{name}: the map spans {span[0]} x {span[1]} units (bounds {bounds}), beyond the "
                         f"{MAX_SPAN}-unit limit per axis where ViZDoom misplaces things (docs/engine_limits.md); "
                         f"world area for this brain: {area}")

    # rewind: barriers up, non-state values back to their biases, lights back to theirs (state left alone)
    resets += [f"Floor_MoveToValue({t}, {MOVE_SPEED}, {CLOSED}, 0);" for t in barrier_tags]
    resets += [f"Floor_MoveToValue({t}, {MOVE_SPEED}, {abs(fl)}, {int(fl < 0)});"
               for (_, fl), t in freeze_tags.items()]
    for key, h in init.items():
        kind, val = key.split(":", 1)
        if val in state:
            continue
        fn = "Floor_MoveToValue" if kind == "floor" else "Ceiling_MoveToValue"
        resets.append(f"{fn}({tags[key]}, {MOVE_SPEED}, {abs(h)}, {int(h < 0)});")
    resets += [f"Light_ChangeToValue({out_tag[o]}, {light_bias[o]});" for o in lights]
    # readout: lights, then requested plane heights
    reads: list[tuple[str, str]] = [(o, f"GetSectorLightLevel({out_tag[o]})") for o in lights]
    read_keys = {}
    for key in tags:
        if key not in inputs:
            read_keys.setdefault(key.split(":", 1)[1], key)
    for val in readout or []:
        key = read_keys.get(val)
        if key is None:
            raise ValueError(f"readout {val!r}: no internal port carries it")
        fn = "GetSectorFloorZ" if key.startswith("floor:") else "GetSectorCeilingZ"
        reads.append((val, f"({fn}({tags[key]}, 0, 0) >> 16)"))
    pages = [[n for n, _ in reads[i:i + PAGE]] for i in range(0, len(reads), PAGE)]
    decl = "\n".join(f"global int {i}:o{i};" for i in range(1, PAGE + 1))
    page_scripts = []
    for pi in range(len(pages)):
        chunk = reads[pi * PAGE:(pi + 1) * PAGE]
        body = " ".join(f"o{i + 1} = {expr};" for i, (_, expr) in enumerate(chunk))
        page_scripts.append(f"script {100 + pi} (void) {{ {body} }}")
    acs = f"""#include "zcommon.acs"
{decl}
script 1 (int tag, int h, int neg) {{ Floor_MoveToValue(tag, {MOVE_SPEED}, h, neg); }}
script 2 (int tag, int h, int neg) {{ Ceiling_MoveToValue(tag, {MOVE_SPEED}, h, neg); }}
script 3 (void) {{
  {chr(10).join('  ' + r for r in resets).strip()}
}}
script 4 (int tid) {{ Thing_ChangeTID(0, tid); }}
script 7 (void) {{ Floor_LowerToLowest({START_TAG}, {MOVE_SPEED}); }}
{chr(10).join(page_scripts)}
{motion_acs(n_dolls)}
{world_acs}
"""
    wad = Path(wad)
    if not layout_only:
        write_wad(wad, m.textmap(), compile_acs(acs), extra_lumps=extra_lumps)
    # settle estimate in game tics: per layer the clock walks every wave, the longest corridor, the sync wait and the
    # clock margin at ~16 units per tic; plan() routes on it and TileRunner gives up after twice it (>= 20,000)
    estimate = sum((s["waves"] * s["wave_spacing"] + s["corridor_width"] + SYNC_WAIT + CLOCK_MARGIN + 15) // 16 + 16
                   for s in stage_stats)
    meta = TileMeta(name=name, ports=inputs, outputs={n: 0 for n, _ in reads}, pages=pages, n_dolls=n_dolls,
                    n_sectors=m.sector_count if layout_only else len(m.rects), layout=dict(
                        width=span[0], bounds=bounds, span=span, max_span=MAX_SPAN, world_area=area, fits=fits,
                        stages=extents, stage_stats=stage_stats, tags=tags, estimated_settle_tics=estimate,
                        motion_script=MOTION_SCRIPT, read_expressions=dict(reads)))
    if not layout_only:
        meta.save(wad.with_suffix(".json"))
    return meta
