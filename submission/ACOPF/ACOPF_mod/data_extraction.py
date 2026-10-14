from dataclasses import dataclass
import numpy as np
from ACOPF.ACOPF_mod.build_ybus import Branch, build_ybus
import os

from ACOPF.ACOPF_mod.live_grid import live_grid
from ACOPF.ACOPF_mod import obs_network

OBSERVATION_NETWORK = os.environ.get("ACOPF_OBS_NETWORK", "1") == "1"


@dataclass
class ACOPFData:
    buses: list
    slack_bus: int
    pv_buses: list
    pq_buses: list
    voltage_0: dict
    angle_0: dict
    ybus: object
    ybus_error: float
    neighbors: dict
    conductance: dict
    susceptance: dict
    branches: dict
    bus_nominal_kv: dict
    branch_i_max_from_a: dict
    branch_i_max_to_a: dict
    branch_s_max_mva: dict
    branch_i_max_from_pu: dict
    branch_i_max_to_pu: dict
    base_mva: float
    generators: list
    redispatchable_generators: list
    renewable_generators: list
    slack_generators: list
    generator_bus: dict
    generator_p_0: dict
    generator_q_0: dict
    generator_p_reference: dict
    generator_target_dispatch: dict
    generator_p_min: dict
    generator_p_max: dict
    generator_q_min: dict
    generator_q_max: dict
    generator_ramp_up: dict
    generator_ramp_down: dict
    generator_cost: dict
    renewable_available_p: dict
    loads: list
    load_bus: dict
    load_p: dict
    load_q: dict
    storages: list
    storage_bus: dict
    storage_power_0: dict
    storage_energy: dict
    storage_energy_min: dict
    storage_energy_max: dict
    storage_p_charge_max: dict
    storage_p_discharge_max: dict
    storage_efficiency_charge: dict
    storage_efficiency_discharge: dict
    storage_loss: dict
    delta_time_hours: float


def _objects_by_index(container):
    return [container[index] for index in range(len(container))]


def _sparse_rows(ybus, active_buses):
    active = set(active_buses)
    conductance = {bus: {} for bus in active_buses}
    susceptance = {bus: {} for bus in active_buses}
    neighbors = {bus: [] for bus in active_buses}

    coo = ybus.tocoo()
    for row, column, value in zip(coo.row, coo.col, coo.data):
        row = int(row)
        column = int(column)
        if row not in active or column not in active or abs(value) <= 1e-14:
            continue
        conductance[row][column] = float(value.real)
        susceptance[row][column] = float(value.imag)
        neighbors[row].append(column)
    return neighbors, conductance, susceptance


def extract_grid2op_data(env, obs, validate_ybus=True, grid_obs=None):
    if OBSERVATION_NETWORK:
        return _extract_grid2op_data(
            env, obs, env.backend._grid, validate_ybus, from_observation=True
        )
    with live_grid(env, obs if grid_obs is None else grid_obs) as live:
        return _extract_grid2op_data(env, obs, live, validate_ybus)


def _extract_grid2op_data(
    env, obs, grid, validate_ybus=True, from_observation=False
):
    cls = type(env)

    if from_observation:
        network = obs_network.observation_network(env, obs, grid)
        ybus, branch_list, ybus_error = network["ybus"], network["branches"], 0.0
        bus_status = network["active"]
        voltage_complex = network["voltage"]
        slack_bus = network["slack_bus"]
    else:
        ybus, branch_list, ybus_error = build_ybus(grid, validate=validate_ybus)
        bus_status = np.asarray(grid.get_bus_status(), dtype=bool)
        voltage_complex = np.asarray(grid.get_V(), dtype=complex)
        slack_bus = None

    buses = [int(bus) for bus in np.flatnonzero(bus_status)]
    active = set(buses)
    neighbors, conductance, susceptance = _sparse_rows(ybus, buses)

    voltage_0 = {bus: float(abs(voltage_complex[bus])) for bus in buses}
    angle_0 = {bus: float(np.angle(voltage_complex[bus])) for bus in buses}

    if slack_bus is None:
        slack_ids = [
            int(bus) for bus in grid.get_slack_ids() if int(bus) in active
        ]
        if not slack_ids:
            raise ValueError(
                "The solved backend does not expose an active slack bus"
            )
        slack_bus = slack_ids[0]
    if slack_bus not in active:
        raise ValueError("No active slack bus in the observed topology")

    generator_objects = _objects_by_index(grid.get_generators())
    load_objects = _objects_by_index(grid.get_loads())
    storage_objects = _objects_by_index(grid.get_storages())

    def observed_bus(position_attribute, sub_attribute, index):
        return obs_network.global_bus(
            cls,
            int(getattr(cls, sub_attribute)[index]),
            int(np.asarray(obs.topo_vect)[
                int(getattr(cls, position_attribute)[index])
            ]),
        )

    def element_bus(objects, position_attribute, sub_attribute, index):
        if from_observation:
            return observed_bus(position_attribute, sub_attribute, index)
        return int(objects[index].bus_id) if bool(objects[index].connected) else -1

    generator_bus_all = {
        g: element_bus(generator_objects, "gen_pos_topo_vect", "gen_to_subid", g)
        for g in range(len(generator_objects))
    }
    load_bus_all = {
        d: element_bus(load_objects, "load_pos_topo_vect", "load_to_subid", d)
        for d in range(len(load_objects))
    }
    storage_bus_all = {
        k: element_bus(
            storage_objects, "storage_pos_topo_vect", "storage_to_subid", k
        )
        for k in range(len(storage_objects))
    }

    generators = [g for g, bus in generator_bus_all.items() if bus in active]
    loads = [d for d, bus in load_bus_all.items() if bus in active]
    storages = [k for k, bus in storage_bus_all.items() if bus in active]

    generator_bus = {g: generator_bus_all[g] for g in generators}
    load_bus = {d: load_bus_all[d] for d in loads}
    storage_bus = {k: storage_bus_all[k] for k in storages}

    slack_generators = [
        generator
        for generator in generators
        if bool(generator_objects[generator].is_slack)
    ]
    redispatchable = [
        generator
        for generator in generators
        if bool(cls.gen_redispatchable[generator])
    ]
    renewables = [
        generator
        for generator in generators
        if bool(cls.gen_renewable[generator])
    ]

    pv_buses = sorted(
        {
            generator_bus[generator]
            for generator in generators
            if bool(generator_objects[generator].voltage_regulator_on)
        }
        - {slack_bus}
    )
    pq_buses = [
        bus for bus in buses if bus != slack_bus and bus not in set(pv_buses)
    ]

    base_mva = float(grid.get_sn_mva())
    bus_nominal_kv = np.asarray(grid.get_bus_vn_kv(), dtype=float)
    thermal_limit_a = np.asarray(env.get_thermal_limit(), dtype=float)
    branches = {branch.line_id: branch for branch in branch_list}
    i_max_from_a = {}
    i_max_to_a = {}
    s_max_mva = {}
    i_max_from = {}
    i_max_to = {}
    for branch in branch_list:
        from_nominal_kv = float(bus_nominal_kv[branch.from_bus])
        to_nominal_kv = float(bus_nominal_kv[branch.to_bus])
        monitored_limit_a = float(thermal_limit_a[branch.line_id])

        rating_mva = (
            np.sqrt(3.0) * from_nominal_kv * monitored_limit_a / 1000.0
        )
        if branch.kind == "transformer":
            receiving_limit_a = (
                rating_mva * 1000.0
                / (np.sqrt(3.0) * to_nominal_kv)
            )
        else:
            receiving_limit_a = monitored_limit_a

        from_base_a = (
            base_mva / (np.sqrt(3.0) * from_nominal_kv) * 1000.0
        )
        to_base_a = (
            base_mva / (np.sqrt(3.0) * to_nominal_kv) * 1000.0
        )
        i_max_from_a[branch.line_id] = monitored_limit_a
        i_max_to_a[branch.line_id] = receiving_limit_a
        s_max_mva[branch.line_id] = rating_mva
        i_max_from[branch.line_id] = monitored_limit_a / from_base_a
        i_max_to[branch.line_id] = receiving_limit_a / to_base_a

    gen_p = np.asarray(obs.gen_p, dtype=float)
    gen_q = np.asarray(obs.gen_q, dtype=float)
    actual_dispatch = np.asarray(obs.actual_dispatch, dtype=float)
    target_dispatch = np.asarray(obs.target_dispatch, dtype=float)
    available_p = np.asarray(obs.gen_p_before_curtail, dtype=float)
    generator_cost_array = np.asarray(cls.gen_cost_per_MW, dtype=float)

    storage_power = np.asarray(obs.storage_power, dtype=float)

    data = ACOPFData(
        buses=buses,
        slack_bus=slack_bus,
        pv_buses=pv_buses,
        pq_buses=pq_buses,
        voltage_0=voltage_0,
        angle_0=angle_0,
        ybus=ybus,
        ybus_error=ybus_error,
        neighbors=neighbors,
        conductance=conductance,
        susceptance=susceptance,
        branches=branches,
        bus_nominal_kv={
            bus: float(bus_nominal_kv[bus]) for bus in buses
        },
        branch_i_max_from_a=i_max_from_a,
        branch_i_max_to_a=i_max_to_a,
        branch_s_max_mva=s_max_mva,
        branch_i_max_from_pu=i_max_from,
        branch_i_max_to_pu=i_max_to,
        base_mva=base_mva,
        generators=generators,
        redispatchable_generators=redispatchable,
        renewable_generators=renewables,
        slack_generators=slack_generators,
        generator_bus=generator_bus,
        generator_p_0={generator: float(gen_p[generator]) for generator in generators},
        generator_q_0={generator: float(gen_q[generator]) for generator in generators},
        generator_p_reference={
            generator: float(gen_p[generator] - actual_dispatch[generator])
            for generator in generators
        },
        generator_target_dispatch={
            generator: float(target_dispatch[generator]) for generator in generators
        },
        generator_p_min={generator: float(cls.gen_pmin[generator]) for generator in generators},
        generator_p_max={generator: float(cls.gen_pmax[generator]) for generator in generators},
        generator_q_min={
            generator: float(generator_objects[generator].min_q_mvar)
            for generator in generators
        },
        generator_q_max={
            generator: float(generator_objects[generator].max_q_mvar)
            for generator in generators
        },
        generator_ramp_up={
            generator: float(cls.gen_max_ramp_up[generator]) for generator in generators
        },
        generator_ramp_down={
            generator: float(cls.gen_max_ramp_down[generator]) for generator in generators
        },
        generator_cost={
            generator: float(generator_cost_array[generator]) for generator in generators
        },
        renewable_available_p={
            generator: float(max(0.0, available_p[generator])) for generator in renewables
        },
        loads=loads,
        load_bus=load_bus,
        load_p={load: float(obs.load_p[load]) for load in loads},
        load_q={load: float(obs.load_q[load]) for load in loads},
        storages=storages,
        storage_bus=storage_bus,
        storage_power_0={
            storage: float(storage_power[storage]) for storage in storages
        },
        storage_energy={
            storage: float(obs.storage_charge[storage]) for storage in storages
        },
        storage_energy_min={
            storage: float(cls.storage_Emin[storage]) for storage in storages
        },
        storage_energy_max={
            storage: float(cls.storage_Emax[storage]) for storage in storages
        },
        storage_p_charge_max={
            storage: float(cls.storage_max_p_absorb[storage]) for storage in storages
        },
        storage_p_discharge_max={
            storage: float(cls.storage_max_p_prod[storage]) for storage in storages
        },
        storage_efficiency_charge={
            storage: float(cls.storage_charging_efficiency[storage])
            for storage in storages
        },
        storage_efficiency_discharge={
            storage: float(cls.storage_discharging_efficiency[storage])
            for storage in storages
        },
        storage_loss={
            storage: float(cls.storage_loss[storage]) for storage in storages
        },
        delta_time_hours=float(env.delta_time_seconds) / 3600.0,
    )
    return data
