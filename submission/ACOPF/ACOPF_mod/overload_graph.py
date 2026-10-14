import os
import itertools

import numpy as np
import networkx as nx
from ACOPF.ACOPF_mod.ac_lodf import ac_lodf, _line_ac_blocks
from ACOPF.ACOPF_mod.n1_check import N_1_check
from ACOPF.ACOPF_mod.live_grid import live_grid
from ACOPF.ACOPF_mod import obs_network
from ACOPF.ACOPF_mod.data_extraction import OBSERVATION_NETWORK

try:
    import matplotlib
    try:
        matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
    except ImportError:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception:
    plt = None
    Line2D = None


def _line_s_pre(env, line_id, obs=None):
    grid = env.backend._grid
    if obs is not None and OBSERVATION_NETWORK:
        V = obs_network.observation_network(env, obs, grid)["voltage"]
    else:
        V = np.asarray(grid.get_V())
    b = _line_ac_blocks(env, obs)[line_id]
    i_glob = b["bus1"]
    I_pre = b["y11"] * V[b["bus1"]] + b["y12"] * V[b["bus2"]]
    return float(abs(V[i_glob]) * abs(I_pre))


def build_gov(env, obs, congested_line, threshold=0.05, lodf=None):
    if OBSERVATION_NETWORK:
        return _build_gov(env, obs, congested_line, threshold, lodf)
    with live_grid(env, obs):
        return _build_gov(env, obs, congested_line, threshold, lodf)


def _build_gov(env, obs, congested_line, threshold=0.05, lodf=None):
    if lodf is None:
        lodf = ac_lodf(env, obs, congested_line)

    cutoff = threshold * _line_s_pre(env, congested_line, obs)
    cls = type(env)

    G = nx.MultiGraph()
    sub1 = int(cls.line_or_to_subid[congested_line])
    sub2 = int(cls.line_ex_to_subid[congested_line])
    G.add_edge(sub1, sub2, key=int(congested_line), line=congested_line,
               delta_mw=None, kind="congested")

    for line_id, d in lodf.items():
        if abs(d["delta_mw"]) < cutoff:
            continue
        s1 = int(cls.line_or_to_subid[line_id])
        s2 = int(cls.line_ex_to_subid[line_id])
        kind = "increase" if d["delta_mw"] > 0 else "decrease"
        G.add_edge(s1, s2, key=int(line_id), line=line_id,
                   delta_mw=d["delta_mw"], kind=kind)

    return G


def identify_paths(G):
    congested_edges = [
        (u, v, d)
        for u, v, d in G.edges(data=True)
        if d.get("kind") == "congested"
    ]
    if not congested_edges:
        raise ValueError("overload-distribution graph has no congested edge")
    c_sub1, _, _ = congested_edges[0]
    Gc = nx.Graph()
    Gc.add_edges_from((u, v) for u, v, d in G.edges(data=True) if d["kind"] in ("decrease", "congested"))
    series_nodes = nx.node_connected_component(Gc, c_sub1)
    series_path = {(u, v): d["line"] for u, v, d in G.edges(data=True)
                   if d["kind"] in ("decrease", "congested") and u in series_nodes and v in series_nodes}

    parallel_path = {(u, v): d["line"] for u, v, d in G.edges(data=True)
                     if d["kind"] == "increase" and (u in series_nodes or v in series_nodes)}

    hubs = sorted({n for u, v, d in G.edges(data=True) if d["kind"] == "increase"
                  for n in (u, v) if n in series_nodes})

    return {"series_path": series_path, "parallel_path": parallel_path, "hubs": hubs}


def label_upstream_downstream(env, obs, G):
    cls = type(env)
    series_path = identify_paths(G)["series_path"]

    c_line = next(d["line"] for _, _, d in G.edges(data=True) if d["kind"] == "congested")
    or_sub, ex_sub = int(cls.line_or_to_subid[c_line]), int(cls.line_ex_to_subid[c_line])
    ua_seed, da_seed = (or_sub, ex_sub) if obs.p_or[c_line] >= 0 else (ex_sub, or_sub)

    DiG = nx.DiGraph()
    for line in series_path.values():
        l_or, l_ex = int(cls.line_or_to_subid[line]), int(cls.line_ex_to_subid[line])
        DiG.add_edge(l_or, l_ex) if obs.p_or[line] >= 0 else DiG.add_edge(l_ex, l_or)

    Ua = {ua_seed} | nx.ancestors(DiG, ua_seed)
    Da = {da_seed} | nx.descendants(DiG, da_seed)
    return sorted(Ua), sorted(Da)


def substation_ranking(G, hubs, obs):
    c_line = next(d["line"] for _, _, d in G.edges(data=True) if d["kind"] == "congested")
    reference = abs(obs.p_or[c_line])

    pct = {}
    for hub in hubs:
        touching = sum(abs(d["delta_mw"]) for u, v, d in G.edges(data=True)
                      if d["kind"] == "increase" and hub in (u, v))
        pct[hub] = 100.0 * touching / reference

    return dict(sorted(pct.items(), key=lambda kv: -kv[1]))


def split(env, obs, G, sub, Ua, Da):
    cls = type(env)
    paths = identify_paths(G)
    series_lines = set(paths["series_path"].values())
    parallel_lines = set(paths["parallel_path"].values())
    if sub not in Ua and sub not in Da:
        raise ValueError(f"substation {sub} is not on the directed series path")
    downstream = sub in Da

    def direction(line):
        is_or = int(cls.line_or_to_subid[line]) == sub
        leaves = (is_or and obs.p_or[line] >= 0) or (not is_or and obs.p_or[line] < 0)
        return "out" if leaves else "in"

    bus1, bus2 = [], []
    corridor_side = "in" if downstream else "out"

    for l in range(cls.n_line):
        if not obs.line_status[l]:
            continue
        if int(cls.line_or_to_subid[l]) != sub and int(cls.line_ex_to_subid[l]) != sub:
            continue
        if l in series_lines:
            (bus1 if direction(l) == corridor_side else bus2).append(("line", l))
        elif l in parallel_lines:
            bus2.append(("line", l))
        else:
            bus1.append(("line", l))

    for g in range(cls.n_gen):
        if int(cls.gen_to_subid[g]) == sub:
            (bus1 if downstream else bus2).append(("gen", g))

    for ld in range(cls.n_load):
        if int(cls.load_to_subid[ld]) == sub:
            (bus2 if downstream else bus1).append(("load", ld))

    for k in range(cls.n_storage):
        if int(cls.storage_to_subid[k]) == sub:
            (bus2 if downstream else bus1).append(("storage", k))

    return {"bus1": bus1, "bus2": bus2}


def action_from_split(env, sub, bus_split):
    cls = type(env)
    set_bus = np.zeros(cls.dim_topo, dtype=int)

    def pos_of(kind, id_):
        if kind == "line":
            return int(cls.line_or_pos_topo_vect[id_]) if int(cls.line_or_to_subid[id_]) == sub \
                else int(cls.line_ex_pos_topo_vect[id_])
        if kind == "gen":
            return int(cls.gen_pos_topo_vect[id_])
        if kind == "load":
            return int(cls.load_pos_topo_vect[id_])
        if kind == "storage":
            return int(cls.storage_pos_topo_vect[id_])
        raise ValueError(f"unknown element kind {kind!r}")

    for kind, id_ in bus_split["bus1"]:
        set_bus[pos_of(kind, id_)] = 1
    for kind, id_ in bus_split["bus2"]:
        set_bus[pos_of(kind, id_)] = 2

    return env.action_space({"set_bus": set_bus})


def _sub_slice(env, sub):
    cls = type(env)
    cum = np.cumsum(cls.sub_info)
    end = int(cum[sub])
    return end - int(cls.sub_info[sub]), end


def classify_gov(env, obs, G, paths, Ua):
    series_nodes = {n for pair in paths["series_path"] for n in pair}
    parallel_nodes = {n for pair in paths["parallel_path"] for n in pair}

    if not parallel_nodes:
        n_multi = multi_node_substations(env, obs, series_nodes)
        return ("multi_nodes" if n_multi else "unmeshed"), n_multi

    increase_sub = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d["kind"] == "increase":
            increase_sub.add_edge(u, v)
    ua_present = [n for n in Ua if increase_sub.has_node(n)]
    is_looped = any(
        any(nx.has_path(increase_sub, n, ua) for ua in ua_present)
        for n in parallel_nodes if increase_sub.has_node(n)
    )
    n_multi = multi_node_substations(env, obs, parallel_nodes)
    return ("looped" if is_looped else "parallel"), n_multi


def multi_node_substations(env, obs, node_set):
    out = []
    for sub in {int(s) for s in node_set}:
        start, end = _sub_slice(env, sub)
        if len({int(b) for b in obs.topo_vect[start:end] if b > 0}) > 1:
            out.append(sub)
    return out


def path_c_branch_switch_candidates(env, obs, paths):
    return [int(l) for l in set(paths["series_path"].values())
            if obs.line_status[l] and obs.time_before_cooldown_line[l] == 0]


def relief_candidates(env, obs, node_set):
    cls = type(env)
    node_set = {int(s) for s in node_set}
    out = []

    for sub in multi_node_substations(env, obs, node_set):
        if obs.time_before_cooldown_sub[sub] > 0:
            continue
        start, end = _sub_slice(env, sub)
        set_bus = np.zeros(cls.dim_topo, dtype=int)
        set_bus[start:end] = np.where(obs.topo_vect[start:end] > 0, 1, 0)
        out.append((f"merge sub {sub}", env.action_space({"set_bus": set_bus})))

    for line in range(cls.n_line):
        if obs.line_status[line] or obs.time_before_cooldown_line[line] > 0:
            continue
        or_sub, ex_sub = int(cls.line_or_to_subid[line]), int(cls.line_ex_to_subid[line])
        if or_sub in node_set or ex_sub in node_set:
            act = env.action_space({"set_line_status": [(int(line), 1)],
                                    "set_bus": {"lines_or_id": [(int(line), 1)],
                                               "lines_ex_id": [(int(line), 1)]}})
            out.append((f"reconnect line {line}", act))

    return out


MAX_LINES_OUT_FOR_DISCONNECT = 4
BRIDGE_PENALTY = 0.5


def _disconnects_a_line(act):
    return bool(np.any(np.asarray(act.set_line_status) < 0))


def _select_best(env, obs, candidates, time_step=0, verbose=True):
    severity_before = float(np.sum(np.maximum(obs.rho - 1.0, 0.0)))
    rho_before = float(obs.rho.max())
    n_out = int((~np.asarray(obs.line_status, dtype=bool)).sum())
    bridges_before = len(N_1_check(env, obs, None)["bridges"])

    best = None
    for label, act in candidates:
        sim_obs, _, sim_done, sim_info = obs.simulate(act, time_step=time_step)
        if sim_done or sim_info.get("is_illegal") or sim_info.get("is_ambiguous"):
            continue

        severity_after = float(np.sum(np.maximum(sim_obs.rho - 1.0, 0.0)))
        score = severity_before - severity_after
        rho_after = float(sim_obs.rho.max())
        if score < 0:
            continue
        if score == 0 and rho_after >= rho_before:
            continue

        n1 = N_1_check(env, obs, act)
        if not n1["passed"]:
            continue

        penalty = 0.0
        if _disconnects_a_line(act):
            if n_out >= MAX_LINES_OUT_FOR_DISCONNECT and severity_after > 0:
                continue
            penalty = BRIDGE_PENALTY * max(0, len(n1["bridges"]) - bridges_before)
            if score - penalty < 0:
                continue
            if score - penalty == 0 and rho_after >= rho_before:
                continue

        key = (score - penalty, -rho_after)
        if best is None or key > best[0]:
            best = (key, act, label, rho_after)

        if severity_before > 0 and severity_after == 0:
            break

    if best is None:
        return None, None, None
    _, act, label, rho_after = best
    return act, label, rho_after


def find_remedial_action(env, obs, gov, paths, ranking, Ua, Da, time_step=0):
    eligible = set(Ua) | set(Da)

    def hub_and_path_c_splits():
        tried = set()
        for sub in ranking:
            if sub not in eligible or sub in tried or obs.time_before_cooldown_sub[sub] > 0:
                continue
            tried.add(sub)
            try:
                yield f"hub split @ sub {sub}", action_from_split(env, sub, split(env, obs, gov, sub, Ua, Da))
            except ValueError:
                continue

        series_nodes = {n for pair in paths["series_path"] for n in pair}
        remaining = sorted((series_nodes & eligible) - tried,
                           key=lambda s: -int(type(env).sub_info[s]))
        for sub in remaining:
            if obs.time_before_cooldown_sub[sub] > 0:
                continue
            try:
                yield f"Path_c split @ sub {sub}", action_from_split(env, sub, split(env, obs, gov, sub, Ua, Da))
            except ValueError:
                continue

    def path_c_switches():
        for line in path_c_branch_switch_candidates(env, obs, paths):
            yield f"branch switch line {line}", env.action_space({"set_line_status": [(line, -1)]})

    def relief(node_set, label_prefix):
        for label, act in relief_candidates(env, obs, node_set):
            yield f"{label_prefix}: {label}", act

    parallel_nodes = {n for pair in paths["parallel_path"] for n in pair}
    da_ua_beyond = (set(Ua) | set(Da)) - parallel_nodes

    all_candidates = itertools.chain(
        hub_and_path_c_splits(),
        path_c_switches(),
        relief(parallel_nodes, "Path_// relief"),
        relief(da_ua_beyond, "Da/Ua relief"),
    )
    return _select_best(env, obs, all_candidates, time_step=time_step)


def plot_gov(env, G, congested_line):
    pos = {int(name.split("_")[1]): xy for name, xy in type(env).grid_layout.items()
          if int(name.split("_")[1]) in G.nodes}

    color = {"decrease": "tab:blue", "increase": "tab:red", "congested": "black"}
    edges = list(G.edges(data=True))
    widths = [4.0 if d["kind"] == "congested" else
             1.0 + 3.0 * abs(d["delta_mw"]) / max(abs(dd["delta_mw"]) for _, _, dd in edges
                                                  if dd["delta_mw"] is not None)
             for _, _, d in edges]

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v) for u, v, _ in edges],
                           edge_color=[color[d["kind"]] for _, _, d in edges], width=widths)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightgray", edgecolors="black",
                           node_size=350)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    legend = [Line2D([0], [0], color="black", lw=4, label=f"congested (line {congested_line})"),
             Line2D([0], [0], color="tab:blue", lw=2, label="decrease (Path_c-like)"),
             Line2D([0], [0], color="tab:red", lw=2, label="increase (Path_//-like)")]
    ax.legend(handles=legend, loc="best")
    ax.set_title(f"Overload Distribution Graph -- congested line {congested_line}")
    ax.axis("off")
    plt.show()


def summarize_gov(G, congested_line):
    print(f"GOv: {G.number_of_nodes()} substations, {G.number_of_edges()} edges "
          f"(threshold survivors + the congested line itself)")
    for u, v, d in G.edges(data=True):
        tag = "CONGESTED" if d["kind"] == "congested" else d["kind"].upper()
        delta = f"{d['delta_mw']:+.2f} MW" if d["delta_mw"] is not None else ""
        print(f"  sub {u:3d} -- sub {v:3d}   line {d['line']:3d}   {tag:9s} {delta}")
