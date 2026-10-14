
import math
from collections import defaultdict

import networkx as nx
import numpy as np

SQRT3 = math.sqrt(3.0)
PMAX_PMIN_RATIO = 4.0


def _bus_graph(env, obs, exclude=()):
    cls = type(env)
    topo = np.asarray(obs.topo_vect, dtype=int)
    status = np.asarray(obs.line_status, dtype=bool)
    g = nx.MultiGraph()
    for line in range(cls.n_line):
        if not status[line] or line in exclude:
            continue
        bo, be = int(topo[cls.line_or_pos_topo_vect[line]]), int(topo[cls.line_ex_pos_topo_vect[line]])
        if bo <= 0 or be <= 0:
            continue
        g.add_edge((int(cls.line_or_to_subid[line]), bo), (int(cls.line_ex_to_subid[line]), be), key=int(line))
    return g


def _node_of(cls, topo, kind, idx):
    pos = {"load": cls.load_pos_topo_vect, "gen": cls.gen_pos_topo_vect, "storage": cls.storage_pos_topo_vect}[kind][idx]
    sub = {"load": cls.load_to_subid, "gen": cls.gen_to_subid, "storage": cls.storage_to_subid}[kind][idx]
    bus = int(topo[pos])
    return (int(sub), bus) if bus > 0 else None


def line_capacity_mva(env, obs, line, v_min_kv=None):
    cls = type(env)
    i_max = float(env.get_thermal_limit()[line])
    if v_min_kv is None:
        v = np.asarray(obs.v_or, dtype=float)
        finite = v[np.isfinite(v) & (v > 0)]
        v_min_kv = float(np.percentile(finite, 5)) if finite.size else 138.0
    return SQRT3 * v_min_kv * i_max / 1000.0


_POCKET_MEMO = {}
_POCKET_MEMO_MAX = 4096


def _cuts_up_to_two(adj):
    parent = {}; parent_edge = {}; depth = {}
    tree_edges = set(); non_tree = []; seen_non_tree = set()
    for root in adj:
        if root in depth:
            continue
        depth[root] = 0; parent[root] = None; parent_edge[root] = None
        stack = [root]
        while stack:
            u = stack.pop()
            for v, e in adj[u].items():
                if v not in depth:
                    depth[v] = depth[u] + 1; parent[v] = u; parent_edge[v] = e
                    tree_edges.add(e); stack.append(v)
                elif e != parent_edge[u] and e != parent_edge[v] and e not in seen_non_tree:
                    seen_non_tree.add(e)
                    non_tree.append((u, v, e))
    mask = {e: 0 for e in tree_edges}
    for bit, (u, v, e) in enumerate(non_tree):
        b = 1 << bit
        a, c = u, v
        while a != c:
            if depth[a] < depth[c]:
                a, c = c, a
            mask[parent_edge[a]] |= b
            a = parent[a]
    cuts = []
    by_mask = {}
    bridges = []
    for e, m in mask.items():
        if m == 0:
            cuts.append((e,)); bridges.append(e)
        else:
            by_mask.setdefault(m, []).append(e)
    for i in range(len(bridges)):
        for j in range(i + 1, len(bridges)):
            cuts.append((bridges[i], bridges[j]))
    for m, group in by_mask.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                cuts.append((group[i], group[j]))
        if m & (m - 1) == 0:
            bit = m.bit_length() - 1
            for e in group:
                cuts.append((e, non_tree[bit][2]))
    return cuts


def _components_without(adj, removed):
    seen = set(); comps = []
    for root in adj:
        if root in seen:
            continue
        comp = {root}; stack = [root]; seen.add(root)
        while stack:
            u = stack.pop()
            for v, e in adj[u].items():
                if e in removed or v in seen:
                    continue
                seen.add(v); comp.add(v); stack.append(v)
        comps.append(comp)
    return comps


def pockets_after_loss(env, obs, line, max_cut=2):
    cls = type(env)
    topo = np.asarray(obs.topo_vect, dtype=int)
    status = np.asarray(obs.line_status, dtype=bool)
    key = (topo.tobytes(), status.tobytes(), int(line), int(max_cut))
    hit = _POCKET_MEMO.get(key)
    if hit is not None:
        return [dict(pk) for pk in hit]
    adj = {}
    line_of = defaultdict(list)
    edge_of = {}
    for li in range(cls.n_line):
        if not status[li] or li == line:
            continue
        bo, be = int(topo[cls.line_or_pos_topo_vect[li]]), int(topo[cls.line_ex_pos_topo_vect[li]])
        if bo <= 0 or be <= 0:
            continue
        u, v = (int(cls.line_or_to_subid[li]), bo), (int(cls.line_ex_to_subid[li]), be)
        if u == v:
            continue
        eid = edge_of.setdefault(frozenset((u, v)), len(edge_of))
        line_of[eid].append(int(li))
        adj.setdefault(u, {})[v] = eid
        adj.setdefault(v, {})[u] = eid
    if not adj:
        return []
    gen_p = np.asarray(obs.gen_p, dtype=float)
    node_gen = defaultdict(float)
    node_loads = defaultdict(list); node_gens = defaultdict(list); node_storages = defaultdict(list)
    for gi in range(cls.n_gen):
        n = _node_of(cls, topo, "gen", gi)
        if n is not None:
            node_gen[n] += float(gen_p[gi]); node_gens[n].append(gi)
    for li in range(cls.n_load):
        n = _node_of(cls, topo, "load", li)
        if n is not None:
            node_loads[n].append(li)
    for si in range(cls.n_storage):
        n = _node_of(cls, topo, "storage", si)
        if n is not None:
            node_storages[n].append(si)
    cuts = [c for c in _cuts_up_to_two(adj) if len(c) <= max_cut]
    pockets = {}
    for cut in sorted(cuts, key=len):
        comps = _components_without(adj, set(cut))
        if len(comps) < 2:
            continue
        comps.sort(key=lambda c: (-sum(node_gen[n] for n in c), -len(c)))
        cut_lines = tuple(sorted(l for e in cut for l in line_of[e]))
        for comp in comps[1:]:
            k2 = frozenset(comp)
            if k2 in pockets and len(pockets[k2]["cut"]) <= len(cut_lines):
                continue
            loads = sorted(li for n in comp for li in node_loads[n])
            if not loads:
                continue
            pockets[k2] = {
                "nodes": sorted(comp), "cut": cut_lines, "loads": loads,
                "gens": sorted(gi for n in comp for gi in node_gens[n]),
                "storages": sorted(si for n in comp for si in node_storages[n]),
            }
    result = list(pockets.values())
    if len(_POCKET_MEMO) >= _POCKET_MEMO_MAX:
        _POCKET_MEMO.clear()
    _POCKET_MEMO[key] = [dict(pk) for pk in result]
    return result


def certificate(env, obs, line, v_min_kv=None, max_cut=2, only_new=True, demand_scale=1.0):
    cls = type(env)
    out = []
    before = {}
    if only_new:
        for pk in pockets_after_loss(env, obs, line=-1, max_cut=max_cut):
            before[frozenset(pk["nodes"])] = len(pk["cut"])
    load_p = np.asarray(obs.load_p, dtype=float); load_q = np.asarray(obs.load_q, dtype=float)
    gen_p = np.asarray(obs.gen_p, dtype=float)
    pmax = np.asarray(cls.gen_pmax, dtype=float)
    redisp = np.asarray(cls.gen_redispatchable, dtype=bool)
    for pocket in pockets_after_loss(env, obs, line, max_cut=max_cut):
        if only_new and before.get(frozenset(pocket["nodes"]), 99) <= len(pocket["cut"]):
            continue
        p = float(load_p[pocket["loads"]].sum()) * float(demand_scale)
        q = float(load_q[pocket["loads"]].sum()) * float(demand_scale)
        g_now = float(gen_p[pocket["gens"]].sum())
        g_max = float(sum(pmax[gi] if redisp[gi] else gen_p[gi] for gi in pocket["gens"]))
        s_dis = float(sum(cls.storage_max_p_prod[si] for si in pocket["storages"]))
        c_b = float(sum(line_capacity_mva(env, obs, b, v_min_kv) for b in pocket["cut"]))
        demand_now = math.hypot(p, q) - g_now
        demand_best = math.hypot(max(p - (g_max + s_dis), 0.0), q)
        out.append({
            "line": int(line), "cut": pocket["cut"], "nodes": pocket["nodes"], "loads": pocket["loads"],
            "load_mw": p, "load_mvar": q, "gen_now_mw": g_now, "gen_max_mw": g_max, "storage_mw": s_dis,
            "capacity_mva": c_b, "margin_now_mva": c_b - demand_now, "margin_best_mva": c_b - demand_best,
            "fatal_now": c_b < demand_now, "fatal_even_with_local_control": c_b < demand_best,
        })
    return out


def fatal_lines(env, obs, lines, demand_scale=1.0, v_min_kv=None):
    out = set()
    for line in lines:
        try:
            certs = certificate(env, obs, int(line), v_min_kv=v_min_kv, demand_scale=demand_scale)
        except Exception:
            continue
        if any(c["fatal_even_with_local_control"] for c in certs):
            out.add(int(line))
    return out


def attack_probability(env, obs):
    lines = [int(l) for l in getattr(env, "alertable_line_ids", ())]
    status = np.asarray(obs.line_status, dtype=bool)
    live = [l for l in lines if status[l]]
    if not live:
        return {}
    rho = np.asarray(obs.rho, dtype=float)[live]
    ranks = np.empty_like(np.argsort(rho)); ranks[np.argsort(rho)] = np.arange(len(live))
    beta = math.log(PMAX_PMIN_RATIO) / max(len(live) - 1, 1)
    w = np.exp(beta * ranks); w /= w.sum()
    return {l: float(pw) for l, pw in zip(live, w)}


def expected_loss(env, obs, v_min_kv=None):
    probs = attack_probability(env, obs)
    fatal = {}
    for line in probs:
        certs = certificate(env, obs, line, v_min_kv=v_min_kv)
        fatal[line] = any(c["fatal_even_with_local_control"] for c in certs)
    return sum(p for l, p in probs.items() if fatal[l]), fatal, probs
