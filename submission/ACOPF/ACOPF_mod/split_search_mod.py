import itertools
import math
import random

import numpy as np

from ACOPF.ACOPF_mod import obs_network

_STATIC = {}

FULL_ENUMERATION_LIMIT = 2048
DEFAULT_MAX_CANDIDATES = 3000
DC_CHUNK = 1024


def _static_susceptance(env, obs):
    key = id(env)
    if key in _STATIC:
        return _STATIC[key]
    grid = obs_network.model_grid(obs)
    lines, trafos = grid.get_lines(), grid.get_trafos()
    cls = type(env)
    b = np.zeros(cls.n_line, dtype=float)
    for line_id in range(cls.n_line):
        element = (
            lines[line_id] if line_id < len(lines) else trafos[line_id - len(lines)]
        )
        x = float(element.x_pu)
        b[line_id] = 1.0 / x if abs(x) > 1e-9 else 1.0 / 1e-9
    _STATIC[key] = b
    return b


def _sub_positions(cls, sub):
    start = int(np.sum(cls.sub_info[:sub]))
    return list(range(start, start + int(cls.sub_info[sub])))


def _position_kind(cls, position):
    line_or = np.flatnonzero(cls.line_or_pos_topo_vect == position)
    if line_or.size:
        return "line_or", int(line_or[0])
    line_ex = np.flatnonzero(cls.line_ex_pos_topo_vect == position)
    if line_ex.size:
        return "line_ex", int(line_ex[0])
    gen = np.flatnonzero(cls.gen_pos_topo_vect == position)
    if gen.size:
        return "gen", int(gen[0])
    load = np.flatnonzero(cls.load_pos_topo_vect == position)
    if load.size:
        return "load", int(load[0])
    storage = np.flatnonzero(cls.storage_pos_topo_vect == position)
    if storage.size:
        return "storage", int(storage[0])
    raise ValueError(position)


def substation_elements(env, obs, sub, exclude_lines=()):
    cls = type(env)
    status = np.asarray(obs.line_status, dtype=bool)
    topo = np.asarray(obs.topo_vect, dtype=int)
    excluded = {int(l) for l in exclude_lines}
    out = []
    for position in _sub_positions(cls, sub):
        kind, ident = _position_kind(cls, position)
        if kind in ("line_or", "line_ex") and (not status[ident] or ident in excluded):
            continue
        if topo[position] <= 0 and kind not in ("line_or", "line_ex"):
            continue
        out.append((position, kind, ident))
    return out


def _mw_limits(env, obs):
    cls = type(env)
    rho = np.asarray(obs.rho, dtype=float)
    p_or = np.asarray(obs.p_or, dtype=float)
    q_or = np.asarray(obs.q_or, dtype=float)
    s_or = np.hypot(p_or, q_or)
    limits = np.full(cls.n_line, np.nan)
    ok = np.isfinite(rho) & (rho > 0.02) & np.isfinite(s_or) & (s_or > 1e-3)
    limits[ok] = s_or[ok] / rho[ok]
    thermal = np.asarray(obs.thermal_limit, dtype=float)
    v_or = np.asarray(obs.v_or, dtype=float)
    fallback = thermal * np.sqrt(3.0) * np.where(v_or > 1.0, v_or, 100.0) / 1000.0
    limits[~ok] = fallback[~ok]
    limits = np.where(np.isfinite(limits) & (limits > 1e-3), limits, fallback)
    return limits


def _base_topology(env, obs):
    cls = type(env)
    topo = np.asarray(obs.topo_vect, dtype=int)
    status = np.asarray(obs.line_status, dtype=bool)
    n_nodes = 2 * int(cls.n_sub)

    def node(position, sub):
        bus = int(topo[position])
        return obs_network.global_bus(cls, sub, bus) if bus > 0 else -1

    line_from = np.array(
        [
            node(int(cls.line_or_pos_topo_vect[l]), int(cls.line_or_to_subid[l]))
            if status[l] else -1
            for l in range(cls.n_line)
        ],
        dtype=int,
    )
    line_to = np.array(
        [
            node(int(cls.line_ex_pos_topo_vect[l]), int(cls.line_ex_to_subid[l]))
            if status[l] else -1
            for l in range(cls.n_line)
        ],
        dtype=int,
    )
    injection = np.zeros(n_nodes, dtype=float)
    for g in range(cls.n_gen):
        n = node(int(cls.gen_pos_topo_vect[g]), int(cls.gen_to_subid[g]))
        if n >= 0:
            injection[n] += float(obs.gen_p[g])
    for d in range(cls.n_load):
        n = node(int(cls.load_pos_topo_vect[d]), int(cls.load_to_subid[d]))
        if n >= 0:
            injection[n] -= float(obs.load_p[d])
    for k in range(cls.n_storage):
        n = node(int(cls.storage_pos_topo_vect[k]), int(cls.storage_to_subid[k]))
        if n >= 0:
            injection[n] -= float(obs.storage_power[k])
    return line_from, line_to, injection


def enumerate_assignments(
    n_elements,
    fixed_index=0,
    focus_indices=(),
    max_candidates=DEFAULT_MAX_CANDIDATES,
    rng=None,
):
    if n_elements <= 1:
        return []
    others = [i for i in range(n_elements) if i != fixed_index]
    total = 2 ** len(others)
    if total <= max_candidates:
        out = []
        for bits in itertools.product((1, 2), repeat=len(others)):
            assignment = [1] * n_elements
            for index, bit in zip(others, bits):
                assignment[index] = bit
            out.append(tuple(assignment))
        return out

    rng = rng or random.Random(0)
    focus = [i for i in focus_indices if i != fixed_index]
    seen = set()
    out = []
    if focus:
        anchor = focus[0]
        companions = [i for i in others if i != anchor]
        k = 0
        while k <= len(companions):
            count = sum(math.comb(len(companions), j) for j in range(k + 1))
            if count * 2 > max_candidates * 0.7:
                break
            k += 1
        k = max(0, k - 1)
        for size in range(k + 1):
            for combo in itertools.combinations(companions, size):
                for anchor_bus in (1, 2):
                    assignment = [1] * n_elements
                    side = anchor_bus
                    assignment[anchor] = side
                    for c in combo:
                        assignment[c] = side
                    if anchor_bus == 1:
                        for i in others:
                            if i != anchor and i not in combo:
                                assignment[i] = 2
                    key = tuple(assignment)
                    if key not in seen and key[fixed_index] == 1:
                        seen.add(key)
                        out.append(key)
    while len(out) < max_candidates:
        assignment = [1] * n_elements
        for i in others:
            assignment[i] = rng.choice((1, 2))
        key = tuple(assignment)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _connected_after_split(elements, assignment, component_of_line, n_components):
    touched = {1: set(), 2: set()}
    has_line = {1: False, 2: False}
    n_on_bus = {1: 0, 2: 0}
    for (position, kind, ident), bus in zip(elements, assignment):
        n_on_bus[bus] += 1
        if kind in ("line_or", "line_ex"):
            comp = component_of_line.get(ident)
            if comp is None:
                continue
            has_line[bus] = True
            touched[bus].add(comp)
    for bus in (1, 2):
        if n_on_bus[bus] and not has_line[bus]:
            return False
    if n_on_bus[1] and n_on_bus[2]:
        if not (touched[1] & touched[2]) and n_components > 1:
            return False
        if not (touched[1] & touched[2]):
            return bool(touched[1]) and bool(touched[2])
    covered = touched[1] | touched[2]
    return len(covered) == n_components


def _components_without_sub(cls, line_from, line_to, sub):
    n_nodes = 2 * int(cls.n_sub)
    parent = list(range(n_nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    sub_nodes = {int(sub), int(sub) + int(cls.n_sub)}
    for l in range(cls.n_line):
        u, v = int(line_from[l]), int(line_to[l])
        if u < 0 or v < 0 or u in sub_nodes or v in sub_nodes:
            continue
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    component_of_line = {}
    labels = {}
    for l in range(cls.n_line):
        u, v = int(line_from[l]), int(line_to[l])
        if u < 0 or v < 0:
            continue
        other = None
        if u in sub_nodes and v not in sub_nodes:
            other = v
        elif v in sub_nodes and u not in sub_nodes:
            other = u
        elif u in sub_nodes and v in sub_nodes:
            other = None
        if other is not None:
            root = find(other)
            labels.setdefault(root, len(labels))
            component_of_line[l] = labels[root]
    roots = set()
    for l in range(cls.n_line):
        u, v = int(line_from[l]), int(line_to[l])
        if u < 0 or v < 0 or u in sub_nodes or v in sub_nodes:
            continue
        roots.add(find(u))
    return component_of_line, max(1, len(roots))


def dc_rank_splits(
    env,
    obs,
    sub,
    monitored_lines=(),
    outage_lines=(),
    max_candidates=DEFAULT_MAX_CANDIDATES,
    focus_lines=(),
    rng=None,
):
    cls = type(env)
    elements = substation_elements(env, obs, sub, exclude_lines=outage_lines)
    if len(elements) < 3:
        return []
    b = _static_susceptance(env, obs)
    line_from, line_to, injection = _base_topology(env, obs)
    line_from = line_from.copy()
    line_to = line_to.copy()
    for l in outage_lines:
        line_from[int(l)] = -1
        line_to[int(l)] = -1
    component_of_line, n_components = _components_without_sub(
        cls, line_from, line_to, sub
    )

    positions = [p for p, _k, _i in elements]
    focus_indices = [
        i for i, (_p, kind, ident) in enumerate(elements)
        if kind in ("line_or", "line_ex") and ident in set(map(int, focus_lines))
    ]
    fixed_index = 0
    assignments = enumerate_assignments(
        len(elements),
        fixed_index=fixed_index,
        focus_indices=focus_indices,
        max_candidates=max_candidates,
        rng=rng,
    )
    topo = np.asarray(obs.topo_vect, dtype=int)
    currently_split = any(int(topo[p]) == 2 for p, _k, _i in elements)
    assignments = [
        a for a in assignments
        if (len(set(a)) == 2 or currently_split)
        and _connected_after_split(elements, a, component_of_line, n_components)
    ]
    if not assignments:
        return []

    configs = [dict(zip(positions, assignment)) for assignment in assignments]
    scores = dc_evaluate_configs(
        env, obs, configs, monitored_lines=monitored_lines, outage_lines=outage_lines
    )
    results = [
        {
            "sub": int(sub),
            "assignment": tuple(int(x) for x in assignment),
            "positions": positions,
            "monitored_rho": float(m),
            "global_rho": float(g),
        }
        for assignment, (m, g) in zip(assignments, scores)
        if np.isfinite(m) and np.isfinite(g)
    ]
    results.sort(key=lambda r: (round(r["monitored_rho"], 3), r["global_rho"]))
    return results


def _position_table(cls):
    key = ("positions", id(cls))
    if key in _STATIC:
        return _STATIC[key]
    table = {}
    cumulative = np.cumsum(cls.sub_info)
    for position in range(int(cls.dim_topo)):
        kind, ident = _position_kind(cls, position)
        sub = int(np.searchsorted(cumulative, position, side="right"))
        table[position] = (kind, ident, sub)
    _STATIC[key] = table
    return table


def dc_evaluate_configs(env, obs, configs, monitored_lines=(), outage_lines=()):
    cls = type(env)
    if not configs:
        return []
    b = _static_susceptance(env, obs)
    table = _position_table(cls)
    line_from, line_to, injection = _base_topology(env, obs)
    line_from = line_from.copy()
    line_to = line_to.copy()
    for l in outage_lines:
        line_from[int(l)] = -1
        line_to[int(l)] = -1
    live = (line_from >= 0) & (line_to >= 0)
    positions = sorted({int(p) for config in configs for p in config})
    touched_subs = {table[p][2] for p in positions}
    active = set(int(n) for n in np.concatenate([line_from[live], line_to[live]]))
    for sub in touched_subs:
        active.add(int(sub))
        active.add(int(sub) + int(cls.n_sub))
    active = sorted(active)
    index_of = {node: i for i, node in enumerate(active)}
    n_nodes = len(active)
    topo = np.asarray(obs.topo_vect, dtype=int)
    limits = _mw_limits(env, obs)

    forbidden = {int(s) for s in touched_subs} | {int(s) + int(cls.n_sub) for s in touched_subs}
    reference = None
    for g in np.argsort(-np.asarray(obs.gen_p)):
        n = obs_network.global_bus(
            cls, int(cls.gen_to_subid[g]), int(topo[int(cls.gen_pos_topo_vect[g])])
        )
        if n in index_of and n not in forbidden:
            reference = index_of[n]
            break
    if reference is None:
        reference = 0

    C, P = len(configs), len(positions)
    column = {p: j for j, p in enumerate(positions)}
    bus = np.zeros((C, P), dtype=int)
    for c, config in enumerate(configs):
        for p, value in config.items():
            bus[c, column[int(p)]] = int(value)
    node1 = np.array([index_of[obs_network.global_bus(cls, table[p][2], 1)] for p in positions])
    node2 = np.array([index_of[obs_network.global_bus(cls, table[p][2], 2)] for p in positions])
    new_node = np.where(bus == 2, node2[None, :], node1[None, :])
    changed = bus > 0

    base_from = np.array([index_of.get(int(n), -1) for n in line_from])
    base_to = np.array([index_of.get(int(n), -1) for n in line_to])
    base_inj = np.zeros(n_nodes)
    for node, value in enumerate(injection):
        if node in index_of:
            base_inj[index_of[node]] = value

    or_cols = [j for j, p in enumerate(positions) if table[p][0] == "line_or" and live[table[p][1]]]
    ex_cols = [j for j, p in enumerate(positions) if table[p][0] == "line_ex" and live[table[p][1]]]
    inj_cols = [j for j, p in enumerate(positions) if table[p][0] in ("gen", "load", "storage")]
    or_lines = np.array([table[positions[j]][1] for j in or_cols], dtype=int)
    ex_lines = np.array([table[positions[j]][1] for j in ex_cols], dtype=int)

    def element_injection(p):
        kind, ident, _sub = table[p]
        if kind == "gen":
            return float(obs.gen_p[ident])
        if kind == "load":
            return -float(obs.load_p[ident])
        return -float(obs.storage_power[ident])

    inj_values = np.array([element_injection(positions[j]) for j in inj_cols])
    inj_current = np.array([
        index_of.get(obs_network.global_bus(cls, table[positions[j]][2], int(topo[positions[j]])), -1)
        for j in inj_cols
    ], dtype=int)

    monitored = [int(l) for l in monitored_lines if live[int(l)]]
    keep = np.array([i for i in range(n_nodes) if i != reference])
    movable = np.zeros(cls.n_line, dtype=bool)
    movable[or_lines] = True
    movable[ex_lines] = True
    fixed = live & ~movable
    B0 = np.zeros((n_nodes, n_nodes))
    lf0, lt0, bb0 = base_from[fixed], base_to[fixed], b[fixed]
    np.add.at(B0, (lf0, lf0), bb0)
    np.add.at(B0, (lt0, lt0), bb0)
    np.add.at(B0, (lf0, lt0), -bb0)
    np.add.at(B0, (lt0, lf0), -bb0)
    moving = np.flatnonzero(live & movable)
    out = []
    for start in range(0, C, DC_CHUNK):
        sl = slice(start, start + DC_CHUNK)
        Cc = bus[sl].shape[0]
        f_idx = np.tile(base_from, (Cc, 1))
        t_idx = np.tile(base_to, (Cc, 1))
        if or_cols:
            sub_new = new_node[sl][:, or_cols]
            sub_changed = changed[sl][:, or_cols]
            f_idx[:, or_lines] = np.where(sub_changed, sub_new, f_idx[:, or_lines])
        if ex_cols:
            sub_new = new_node[sl][:, ex_cols]
            sub_changed = changed[sl][:, ex_cols]
            t_idx[:, ex_lines] = np.where(sub_changed, sub_new, t_idx[:, ex_lines])
        inj = np.tile(base_inj, (Cc, 1))
        if inj_cols:
            sub_new = new_node[sl][:, inj_cols]
            sub_changed = changed[sl][:, inj_cols]
            rows = np.repeat(np.arange(Cc), len(inj_cols)).reshape(Cc, -1)
            values = np.where(sub_changed, inj_values[None, :], 0.0)
            valid_current = inj_current >= 0
            np.add.at(
                inj,
                (rows[:, valid_current].ravel(), np.tile(inj_current[valid_current], Cc)),
                -values[:, valid_current].ravel(),
            )
            np.add.at(inj, (rows.ravel(), sub_new.ravel()), values.ravel())
        mismatch = inj.sum(axis=1, keepdims=True)
        loads = np.where(inj < 0, -inj, 0.0)
        weight = loads / np.maximum(loads.sum(axis=1, keepdims=True), 1e-9)
        inj = inj - mismatch * weight
        B = np.broadcast_to(B0, (Cc, n_nodes, n_nodes)).copy()
        if moving.size:
            rows = np.repeat(np.arange(Cc), moving.size)
            lf = f_idx[:, moving].ravel()
            lt = t_idx[:, moving].ravel()
            bb = np.tile(b[moving], Cc)
            np.add.at(B, (rows, lf, lf), bb)
            np.add.at(B, (rows, lt, lt), bb)
            np.add.at(B, (rows, lf, lt), -bb)
            np.add.at(B, (rows, lt, lf), -bb)
        degree = np.count_nonzero(B, axis=2)
        Bk = B[:, keep][:, :, keep]
        Pk = inj[:, keep]
        diag_zero = degree[:, keep] == 0
        idx = np.nonzero(diag_zero)
        Bk[idx[0], idx[1], idx[1]] = 1.0
        Pk = np.where(diag_zero, 0.0, Pk)
        theta = np.zeros((Cc, n_nodes))
        try:
            theta[:, keep] = np.linalg.solve(Bk, Pk[..., None])[..., 0]
        except np.linalg.LinAlgError:
            out.extend([(float("inf"), float("inf"))] * Cc)
            continue
        flows = b[live][None, :] * (
            np.take_along_axis(theta, f_idx[:, live], axis=1)
            - np.take_along_axis(theta, t_idx[:, live], axis=1)
        )
        rho = np.zeros((Cc, cls.n_line))
        rho[:, live] = np.abs(flows) / limits[live][None, :]
        rho = np.where(np.isfinite(rho), rho, np.inf)
        global_rho = rho.max(axis=1)
        monitored_rho = rho[:, monitored].max(axis=1) if monitored else global_rho
        out.extend(zip(monitored_rho.tolist(), global_rho.tolist()))
    return out


def config_action(env, obs, config):
    cls = type(env)
    topo = np.asarray(obs.topo_vect, dtype=int)
    set_bus = np.zeros(cls.dim_topo, dtype=int)
    changed = False
    for position, bus in config.items():
        set_bus[int(position)] = int(bus)
        if int(topo[int(position)]) != int(bus):
            changed = True
    if not changed:
        return None
    return env.action_space({"set_bus": set_bus})


def assignment_action(env, obs, ranked):
    cls = type(env)
    topo = np.asarray(obs.topo_vect, dtype=int)
    set_bus = np.zeros(cls.dim_topo, dtype=int)
    changed = False
    for position, bus in zip(ranked["positions"], ranked["assignment"]):
        set_bus[position] = int(bus)
        if int(topo[position]) != int(bus):
            changed = True
    if not changed:
        return None
    return env.action_space({"set_bus": set_bus})


def candidate_substations(env, obs, lines, hops=0):
    cls = type(env)
    subs = set()
    for l in lines:
        subs.add(int(cls.line_or_to_subid[int(l)]))
        subs.add(int(cls.line_ex_to_subid[int(l)]))
    frontier = set(subs)
    status = np.asarray(obs.line_status, dtype=bool)
    for _ in range(int(hops)):
        nxt = set()
        for l in range(cls.n_line):
            if not status[l]:
                continue
            a, b_ = int(cls.line_or_to_subid[l]), int(cls.line_ex_to_subid[l])
            if a in frontier:
                nxt.add(b_)
            if b_ in frontier:
                nxt.add(a)
        frontier = nxt - subs
        subs |= nxt
    return sorted(subs)


def search_splits(
    env,
    obs,
    substations,
    monitored_lines,
    outage_lines=(),
    focus_lines=None,
    top_k=16,
    max_candidates=DEFAULT_MAX_CANDIDATES,
    min_dc_improvement=0.0,
    baseline_monitored=None,
):
    cls = type(env)
    focus = list(monitored_lines) if focus_lines is None else list(focus_lines)
    pooled = []
    for sub in substations:
        sub = int(sub)
        if int(obs.time_before_cooldown_sub[sub]) > 0:
            continue
        if int(cls.sub_info[sub]) < 4:
            continue
        ranked = dc_rank_splits(
            env,
            obs,
            sub,
            monitored_lines=monitored_lines,
            outage_lines=outage_lines,
            max_candidates=max_candidates,
            focus_lines=list(focus) + list(outage_lines),
        )
        for r in ranked[: max(top_k * 3, 24)]:
            if (
                baseline_monitored is not None
                and r["monitored_rho"] > baseline_monitored - min_dc_improvement
            ):
                continue
            action = assignment_action(env, obs, r)
            if action is None:
                continue
            pooled.append((r["monitored_rho"], r["global_rho"], sub, action, r))
    pooled.sort(key=lambda item: (round(item[0], 3), item[1]))
    out = []
    for monitored_rho, global_rho, sub, action, r in pooled[:top_k]:
        label = f"split @ sub {sub} (dc {monitored_rho:.2f}/{global_rho:.2f})"
        out.append((action, label, r))
    return out
