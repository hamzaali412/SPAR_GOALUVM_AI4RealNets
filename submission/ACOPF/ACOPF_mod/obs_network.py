
import numpy as np

from ACOPF.ACOPF_mod.build_ybus import Branch, _branch_admittances


def global_bus(cls, substation, busbar):
    if busbar <= 0:
        return -1
    return int(substation) + (0 if int(busbar) == 1 else int(cls.n_sub))


def _element_bus(cls, obs, position, substation):
    return global_bus(cls, substation, int(np.asarray(obs.topo_vect)[position]))


def observation_branches(env, obs, grid):
    cls = type(env)
    status = np.asarray(obs.line_status, dtype=bool)
    containers = ((grid.get_lines(), 0, "line"),
                  (grid.get_trafos(), len(grid.get_lines()), "transformer"))
    branches = []
    for container, first_id, kind in containers:
        for local_id in range(len(container)):
            line_id = first_id + local_id
            if line_id >= cls.n_line or not status[line_id]:
                continue
            from_bus = _element_bus(
                cls, obs, int(cls.line_or_pos_topo_vect[line_id]),
                int(cls.line_or_to_subid[line_id]),
            )
            to_bus = _element_bus(
                cls, obs, int(cls.line_ex_pos_topo_vect[line_id]),
                int(cls.line_ex_to_subid[line_id]),
            )
            if from_bus < 0 or to_bus < 0:
                continue
            element = container[local_id]
            yff, yft, ytf, ytt, ratio, shift, tap_on_from = _branch_admittances(
                element
            )
            branches.append(
                Branch(
                    line_id=line_id,
                    kind=kind,
                    from_bus=from_bus,
                    to_bus=to_bus,
                    r_pu=float(element.r_pu),
                    x_pu=float(element.x_pu),
                    h_from_pu=complex(element.h1_pu),
                    h_to_pu=complex(element.h2_pu),
                    tap_ratio=ratio,
                    phase_shift_rad=shift,
                    tap_on_from_side=tap_on_from,
                    yff=yff,
                    yft=yft,
                    ytf=ytf,
                    ytt=ytt,
                )
            )
    return branches


def observation_bus_state(env, obs, grid, branches):
    cls = type(env)
    n_bus = len(grid.get_V())
    nominal_kv = np.asarray(grid.get_bus_vn_kv(), dtype=float)
    magnitude = np.zeros(n_bus, dtype=float)
    angle = np.zeros(n_bus, dtype=float)
    seen = np.zeros(n_bus, dtype=bool)

    sides = (
        ("line_or_pos_topo_vect", "line_or_to_subid", "v_or", "theta_or"),
        ("line_ex_pos_topo_vect", "line_ex_to_subid", "v_ex", "theta_ex"),
    )
    for branch in branches:
        line_id = branch.line_id
        for position_attribute, sub_attribute, v_attr, theta_attr in sides:
            bus = _element_bus(
                cls, obs, int(getattr(cls, position_attribute)[line_id]),
                int(getattr(cls, sub_attribute)[line_id]),
            )
            if bus < 0 or seen[bus]:
                continue
            kv = float(np.asarray(getattr(obs, v_attr))[line_id])
            base_kv = float(nominal_kv[bus])
            if base_kv <= 0.0 or not np.isfinite(kv) or kv <= 0.0:
                continue
            magnitude[bus] = kv / base_kv
            angle[bus] = np.deg2rad(
                float(np.asarray(getattr(obs, theta_attr))[line_id])
            )
            seen[bus] = True
    return seen, magnitude * np.exp(1j * angle)


def observation_slack_bus(env, obs, grid, active):
    cls = type(env)
    generators = grid.get_generators()
    for generator_id in range(len(generators)):
        if not bool(generators[generator_id].is_slack):
            continue
        bus = _element_bus(
            cls, obs, int(cls.gen_pos_topo_vect[generator_id]),
            int(cls.gen_to_subid[generator_id]),
        )
        if bus >= 0 and active[bus]:
            return bus
    return None


def ybus_from_branches(branches, n_bus):
    from scipy.sparse import csr_matrix

    count = len(branches)
    if count == 0:
        return csr_matrix((n_bus, n_bus), dtype=complex)
    from_bus = np.asarray([b.from_bus for b in branches], dtype=int)
    to_bus = np.asarray([b.to_bus for b in branches], dtype=int)
    row = np.arange(count, dtype=int)
    both_rows = np.concatenate((row, row))
    both_cols = np.concatenate((from_bus, to_bus))

    connection_from = csr_matrix(
        (np.ones(count), (row, from_bus)), shape=(count, n_bus)
    )
    connection_to = csr_matrix(
        (np.ones(count), (row, to_bus)), shape=(count, n_bus)
    )
    y_from = csr_matrix(
        (
            np.concatenate(
                ([b.yff for b in branches], [b.yft for b in branches])
            ),
            (both_rows, both_cols),
        ),
        shape=(count, n_bus),
    )
    y_to = csr_matrix(
        (
            np.concatenate(
                ([b.ytf for b in branches], [b.ytt for b in branches])
            ),
            (both_rows, both_cols),
        ),
        shape=(count, n_bus),
    )
    return (connection_from.T @ y_from + connection_to.T @ y_to).tocsr()


def observation_network(env, obs, grid):
    branches = observation_branches(env, obs, grid)
    n_bus = len(grid.get_V())
    active, voltage = observation_bus_state(env, obs, grid, branches)
    return {
        "branches": branches,
        "ybus": ybus_from_branches(branches, n_bus),
        "active": active,
        "voltage": voltage,
        "slack_bus": observation_slack_bus(env, obs, grid, active),
    }
