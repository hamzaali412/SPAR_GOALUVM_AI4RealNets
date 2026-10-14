
from collections import defaultdict
import numpy as np
import networkx as nx


def N_1_check(env, obs, split_act, attackable_lines=None):
    cls = type(env)
    if attackable_lines is None:
        attackable_lines = [int(l) for l in env.alertable_line_ids]

    candidate_bus = np.asarray(obs.topo_vect, dtype=int).copy()
    candidate_status = np.asarray(obs.line_status, dtype=bool).copy()
    if split_act is not None:
        set_bus = np.asarray(split_act.set_bus)
        touched = set_bus != 0
        candidate_bus[touched] = set_bus[touched]
        set_status = np.asarray(split_act.set_line_status)
        candidate_status[set_status == 1] = True
        candidate_status[set_status == -1] = False

    edge_lines = defaultdict(list)
    G_simple = nx.Graph()
    G_full = nx.MultiGraph()

    for l in range(cls.n_line):
        if not candidate_status[l]:
            continue
        pos_or = int(cls.line_or_pos_topo_vect[l])
        pos_ex = int(cls.line_ex_pos_topo_vect[l])
        bus_or, bus_ex = int(candidate_bus[pos_or]), int(candidate_bus[pos_ex])
        if bus_or <= 0 or bus_ex <= 0:
            continue
        node_or = (int(cls.line_or_to_subid[l]), bus_or)
        node_ex = (int(cls.line_ex_to_subid[l]), bus_ex)

        G_simple.add_edge(node_or, node_ex)
        G_full.add_edge(node_or, node_ex, line=l)
        edge_lines[frozenset((node_or, node_ex))].append(l)

    connected = nx.is_connected(G_simple) if G_simple.number_of_nodes() else True

    critical_lines = []
    if connected:
        for u, v in nx.bridges(G_simple):
            lines = edge_lines[frozenset((u, v))]
            if len(lines) == 1:
                critical_lines.append(lines[0])

    vulnerable_to = sorted(set(critical_lines) & set(attackable_lines))

    return {
        "connected": connected,
        "bridges": sorted(critical_lines),
        "vulnerable_to": vulnerable_to,
        "passed": connected and not vulnerable_to,
    }
