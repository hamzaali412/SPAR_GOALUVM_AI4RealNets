
import numpy as np


def record_intervention(env, obs_before, act, label, step):
    cls = type(env)
    topo_positions = np.where(np.asarray(act.set_bus) != 0)[0]
    lines = np.where(np.asarray(act.set_line_status) != 0)[0]

    cum = np.cumsum(cls.sub_info)
    subs = {int(np.searchsorted(cum, pos, side="right")) for pos in topo_positions}

    return {
        "label": label,
        "step": step,
        "topo_positions": topo_positions.tolist(),
        "prior_topo": obs_before.topo_vect[topo_positions].astype(int).tolist(),
        "lines": lines.tolist(),
        "prior_connected": obs_before.line_status[lines].astype(bool).tolist(),
        "subs": sorted(subs),
    }


def cooldown_cleared(obs, record):
    return (all(obs.time_before_cooldown_sub[s] == 0 for s in record["subs"]) and
            all(obs.time_before_cooldown_line[l] == 0 for l in record["lines"]))


def _mask_offline_line_ends(env, obs, set_bus, own_lines=()):
    if obs is None:
        return set_bus
    cls = type(env)
    status = np.asarray(obs.line_status, dtype=bool)
    own = {int(l) for l in own_lines}
    for line in np.flatnonzero(~status):
        if int(line) in own:
            continue
        set_bus[int(cls.line_or_pos_topo_vect[line])] = 0
        set_bus[int(cls.line_ex_pos_topo_vect[line])] = 0
    return set_bus


def build_restore_action(env, records, obs=None):
    if isinstance(records, dict):
        records = [records]
    cls = type(env)
    set_bus = np.zeros(cls.dim_topo, dtype=int)
    set_line_status = []
    reconnected = []
    for record in records:
        for pos, prior in zip(record["topo_positions"], record["prior_topo"]):
            set_bus[pos] = prior if prior > 0 else 0
        for line, was_connected in zip(record["lines"], record["prior_connected"]):
            if was_connected:
                set_line_status.append((int(line), 1))
                set_bus[int(cls.line_or_pos_topo_vect[line])] = 1
                set_bus[int(cls.line_ex_pos_topo_vect[line])] = 1
                reconnected.append(int(line))
            else:
                set_line_status.append((int(line), -1))
    _mask_offline_line_ends(env, obs, set_bus, reconnected)

    act_dict = {"set_bus": set_bus}
    if set_line_status:
        act_dict["set_line_status"] = set_line_status
    return env.action_space(act_dict)


def build_partial_restore_action(env, record, include_lines=True, include_topology=True, obs=None):
    cls = type(env)
    set_bus = np.zeros(cls.dim_topo, dtype=int)
    set_line_status = []
    reconnected = []
    if include_topology:
        for pos, prior in zip(record["topo_positions"], record["prior_topo"]):
            set_bus[pos] = prior if prior > 0 else 0
    if include_lines:
        for line, was_connected in zip(record["lines"], record["prior_connected"]):
            if was_connected:
                set_line_status.append((int(line), 1))
                set_bus[int(cls.line_or_pos_topo_vect[line])] = 1
                set_bus[int(cls.line_ex_pos_topo_vect[line])] = 1
                reconnected.append(int(line))
            else:
                set_line_status.append((int(line), -1))
    _mask_offline_line_ends(env, obs, set_bus, reconnected)
    act_dict = {"set_bus": set_bus}
    if set_line_status:
        act_dict["set_line_status"] = set_line_status
    return env.action_space(act_dict)


def trim_record(record, drop_topology=False, drop_lines=False):
    if drop_topology:
        record["topo_positions"] = []
        record["prior_topo"] = []
        record["subs"] = []
    if drop_lines:
        record["lines"] = []
        record["prior_connected"] = []
    return not record["topo_positions"] and not record["lines"]
