import time
import os as _os
from types import SimpleNamespace

from pyomo.opt import SolverResults

import numpy as np
import pyomo.environ as pyo

from ACOPF.ACOPF_mod.acopf_setup import DEFAULT_IPOPT_OPTIONS, initialization


ACCEPTABLE_TERMINATIONS = {
    pyo.TerminationCondition.optimal,
    pyo.TerminationCondition.locallyOptimal,
    pyo.TerminationCondition.feasible,
}

RESERVE_DOWN_FROM_CURTAIL = float(_os.environ.get("ACOPF_RESERVE_DOWN_FROM_CURTAIL", "0.0"))


RESERVE_ACTION_OFFSET = _os.environ.get("ACOPF_RESERVE_ACTION_OFFSET", "1") == "1"
RESERVE_NET_OF_REDISPATCH = _os.environ.get("ACOPF_RESERVE_NET_OF_REDISPATCH", "1") == "1"
RESERVE_SLACK_WEIGHT = float(_os.environ.get("ACOPF_RESERVE_SLACK_WEIGHT", "100.0"))
PREVIEW_SLACK_WEIGHT = float(_os.environ.get("ACOPF_PREVIEW_SLACK_WEIGHT", "20.0"))


def _add_preview_requirements(model, preview_up, preview_down):
    G = model.REDISPATCHABLE_GEN
    model.preview_slack_up = pyo.Var(within=pyo.NonNegativeReals, initialize=0.0)
    model.preview_slack_down = pyo.Var(within=pyo.NonNegativeReals, initialize=0.0)
    charge = sum(model.storage_charge[k] for k in model.STORAGE)
    discharge = sum(model.storage_discharge[k] for k in model.STORAGE)
    model.preview_up_requirement = pyo.Constraint(
        expr=sum(model.reserve_up[g] for g in G) + discharge - charge + model.preview_slack_up >= preview_up)
    model.preview_down_requirement = pyo.Constraint(
        expr=sum(model.reserve_down[g] for g in G) + charge - discharge + model.preview_slack_down >= preview_down)
    model.preview_slack_penalty = pyo.Expression(expr=PREVIEW_SLACK_WEIGHT * (model.preview_slack_up + model.preview_slack_down))
MODEL_STORAGE_SUSTAIN_H = float(_os.environ.get("ACOPF_STORAGE_SUSTAIN_H", "0.5"))


def _storage_power_bounds(data, k):
    c_max = float(data.storage_p_charge_max[k]); d_max = float(data.storage_p_discharge_max[k])
    if MODEL_STORAGE_SUSTAIN_H <= 0.0:
        return c_max, d_max
    hours = max(MODEL_STORAGE_SUSTAIN_H, float(data.delta_time_hours))
    e0 = float(data.storage_energy[k]); e_max = float(data.storage_energy_max[k]); e_min = float(data.storage_energy_min[k])
    eta_c = max(float(data.storage_efficiency_charge[k]), 1e-6); eta_d = max(float(data.storage_efficiency_discharge[k]), 1e-6)
    head_up = max(0.98 * e_max - e0, 0.0); head_dn = max(e0 - max(e_min, 0.02 * e_max), 0.0)
    return min(c_max, head_up / (eta_c * hours)), min(d_max, head_dn * eta_d / hours)


def _add_reserve_ramp_limits(model, data):
    G = model.REDISPATCHABLE_GEN
    if RESERVE_NET_OF_REDISPATCH:
        model.reserve_up_ramp = pyo.Constraint(
            G, rule=lambda m, g: m.reserve_up[g] <= data.generator_ramp_up[g] - m.redispatch_up[g] + m.redispatch_down[g])
        model.reserve_down_ramp = pyo.Constraint(
            G, rule=lambda m, g: m.reserve_down[g] <= data.generator_ramp_down[g] + m.redispatch_up[g] - m.redispatch_down[g])
    else:
        model.reserve_up_ramp = pyo.Constraint(G, rule=lambda m, g: m.reserve_up[g] <= data.generator_ramp_up[g])
        model.reserve_down_ramp = pyo.Constraint(G, rule=lambda m, g: m.reserve_down[g] <= data.generator_ramp_down[g])
    model.reserve_up_headroom = pyo.Constraint(G, rule=lambda m, g: m.reserve_up[g] <= data.generator_p_max[g] - m.generator_p[g])
    model.reserve_down_headroom = pyo.Constraint(G, rule=lambda m, g: m.reserve_down[g] <= m.generator_p[g] - data.generator_p_min[g])


def _add_reserve_requirements(model, data, forecast_up, forecast_down, margin, slack_weight):
    G = model.REDISPATCHABLE_GEN
    model.reserve_slack_up = pyo.Var(within=pyo.NonNegativeReals, initialize=0.0)
    model.reserve_slack_down = pyo.Var(within=pyo.NonNegativeReals, initialize=0.0)
    elastic = RESERVE_SLACK_WEIGHT > 0.0
    if not elastic:
        model.reserve_slack_up.fix(0.0)
        model.reserve_slack_down.fix(0.0)
    up = sum(model.reserve_up[g] for g in G)
    down = sum(model.reserve_down[g] for g in G)
    if RESERVE_ACTION_OFFSET:
        curtail = sum(model.curtailment[g] for g in model.RENEWABLE_GEN)
        charge = sum(model.storage_charge[k] for k in model.STORAGE)
        discharge = sum(model.storage_discharge[k] for k in model.STORAGE)
        model.reserve_up_requirement = pyo.Constraint(
            expr=up + discharge - charge - curtail + model.reserve_slack_up >= forecast_up + margin)
        model.reserve_down_requirement = pyo.Constraint(
            expr=down + charge + curtail - discharge + model.reserve_slack_down >= forecast_down + margin)
    else:
        model.reserve_up_requirement = pyo.Constraint(expr=up + model.reserve_slack_up >= forecast_up + margin)
        model.reserve_down_requirement = pyo.Constraint(
            expr=down + _down_reserve_from_renewables(model, data) + model.reserve_slack_down >= forecast_down + margin)
    model.reserve_slack_penalty = pyo.Expression(
        expr=(slack_weight * (model.reserve_slack_up + model.reserve_slack_down)) if elastic else 0.0)

def _down_reserve_from_renewables(model, data):
    if RESERVE_DOWN_FROM_CURTAIL <= 0.0:
        return 0.0
    return RESERVE_DOWN_FROM_CURTAIL * sum(
        float(data.generator_p_0[g]) - model.curtailment[g] for g in model.RENEWABLE_GEN
    )


def acopf_setup_twostage_mod(
    data,
    rho_target=0.95,
    voltage_min=0.90,
    voltage_max=1.10,
    enforce_generator_q_limits=False,
    voltage_trust_region=0.10,
    angle_trust_region=np.deg2rad(30.0),
    forecast_up_requirement=0.0,
    forecast_down_requirement=0.0,
    reserve_margin_mw=5.0,
    recovery_weight=0.35,
    reserve_reward=1e-3,
    balancing_cap_mw=None,
    attack_lines=None,
    attack_weight=200.0,
    curtailment_weight=5.0,
    contingency_terms=None,
    contingency_weight=200.0,
    storage_weight=0.25,
):
    model = pyo.ConcreteModel()

    buses = list(data.buses)
    non_slack = [bus for bus in buses if bus != data.slack_bus]
    pq_buses = list(data.pq_buses)
    pq_set = set(pq_buses)
    generators = list(data.generators)
    redispatchable = list(data.redispatchable_generators)
    redispatchable_set = set(redispatchable)
    fixed_generators = [g for g in generators if g not in redispatchable_set]
    renewables = list(data.renewable_generators)
    renewable_set = set(renewables)
    storages = list(data.storages)
    branches = sorted(data.branches)

    model.BUS = pyo.Set(initialize=buses, ordered=True)
    model.NON_SLACK = pyo.Set(initialize=non_slack, ordered=True)
    model.PQ_BUS = pyo.Set(initialize=pq_buses, ordered=True)
    model.GEN = pyo.Set(initialize=generators, ordered=True)
    model.REDISPATCHABLE_GEN = pyo.Set(
        initialize=redispatchable, ordered=True
    )
    model.FIXED_GEN = pyo.Set(initialize=fixed_generators, ordered=True)
    model.RENEWABLE_GEN = pyo.Set(initialize=renewables, ordered=True)
    model.STORAGE = pyo.Set(initialize=storages, ordered=True)
    model.LINE = pyo.Set(initialize=branches, ordered=True)

    model.theta = pyo.Var(
        model.NON_SLACK,
        bounds=lambda _m, bus: (
            max(-np.pi, data.angle_0[bus] - angle_trust_region),
            min(np.pi, data.angle_0[bus] + angle_trust_region),
        ),
        initialize={bus: data.angle_0[bus] for bus in non_slack},
    )

    def voltage_bounds(_model, bus):
        observed = data.voltage_0[bus]
        return (
            max(min(voltage_min, observed), observed - voltage_trust_region),
            min(max(voltage_max, observed), observed + voltage_trust_region),
        )

    model.voltage = pyo.Var(
        model.PQ_BUS,
        within=pyo.NonNegativeReals,
        bounds=voltage_bounds,
        initialize={bus: data.voltage_0[bus] for bus in pq_buses},
    )

    def voltage(model_, bus):
        return model_.voltage[bus] if bus in pq_set else data.voltage_0[bus]

    def angle(model_, bus):
        return (
            model_.theta[bus]
            if bus != data.slack_bus
            else data.angle_0[data.slack_bus]
        )

    model.generator_p = pyo.Var(
        model.GEN,
        within=pyo.NonNegativeReals,
        bounds=lambda _m, g: (
            data.generator_p_min[g], data.generator_p_max[g]
        ),
        initialize=lambda _m, g: float(data.generator_p_0[g]),
    )
    model.redispatch_up = pyo.Var(
        model.REDISPATCHABLE_GEN,
        within=pyo.NonNegativeReals,
        bounds=lambda _m, g: (0.0, data.generator_ramp_up[g]),
        initialize=0.0,
    )
    model.redispatch_down = pyo.Var(
        model.REDISPATCHABLE_GEN,
        within=pyo.NonNegativeReals,
        bounds=lambda _m, g: (0.0, data.generator_ramp_down[g]),
        initialize=0.0,
    )
    model.redispatch_definition = pyo.Constraint(
        model.REDISPATCHABLE_GEN,
        rule=lambda m, g: m.generator_p[g]
        == data.generator_p_0[g] + m.redispatch_up[g] - m.redispatch_down[g],
    )

    model.reserve_up = pyo.Var(
        model.REDISPATCHABLE_GEN, within=pyo.NonNegativeReals
    )
    model.reserve_down = pyo.Var(
        model.REDISPATCHABLE_GEN, within=pyo.NonNegativeReals
    )
    _add_reserve_ramp_limits(model, data)
    required_up = max(0.0, float(forecast_up_requirement)) + float(reserve_margin_mw)
    required_down = max(0.0, float(forecast_down_requirement)) + float(reserve_margin_mw)

    model.dispatch_target_abs = pyo.Var(
        model.REDISPATCHABLE_GEN, within=pyo.NonNegativeReals
    )
    model.dispatch_target_abs_positive = pyo.Constraint(
        model.REDISPATCHABLE_GEN,
        rule=lambda m, g: m.dispatch_target_abs[g]
        >= m.generator_p[g] - data.generator_p_reference[g],
    )
    model.dispatch_target_abs_negative = pyo.Constraint(
        model.REDISPATCHABLE_GEN,
        rule=lambda m, g: m.dispatch_target_abs[g]
        >= -(m.generator_p[g] - data.generator_p_reference[g]),
    )
    model.fixed_generator_output = pyo.Constraint(
        model.FIXED_GEN,
        rule=lambda m, g: m.generator_p[g] == data.generator_p_0[g],
    )

    def generator_q_bounds(_model, generator):
        if enforce_generator_q_limits:
            return data.generator_q_min[generator], data.generator_q_max[generator]
        return -1e4, 1e4

    model.generator_q = pyo.Var(
        model.GEN,
        bounds=generator_q_bounds,
        initialize=lambda _m, g: data.generator_q_0[g],
    )

    model.curtailment = pyo.Var(
        model.RENEWABLE_GEN,
        within=pyo.NonNegativeReals,
        bounds=lambda _m, g: (
            0.0,
            min(data.generator_p_0[g], data.renewable_available_p[g]),
        ),
        initialize=0.0,
    )

    model.storage_charge = pyo.Var(
        model.STORAGE,
        within=pyo.NonNegativeReals,
        bounds=lambda _m, k: (0.0, _storage_power_bounds(data, k)[0]),
        initialize=lambda _m, k: min(max(0.0, data.storage_power_0[k]), _storage_power_bounds(data, k)[0]),
    )
    model.storage_discharge = pyo.Var(
        model.STORAGE,
        within=pyo.NonNegativeReals,
        bounds=lambda _m, k: (0.0, _storage_power_bounds(data, k)[1]),
        initialize=lambda _m, k: min(max(0.0, -data.storage_power_0[k]), _storage_power_bounds(data, k)[1]),
    )

    def energy_after(model_, storage):
        dt = data.delta_time_hours
        return (
            data.storage_energy[storage]
            - data.storage_loss[storage] * dt
            + data.storage_efficiency_charge[storage]
            * model_.storage_charge[storage]
            * dt
            - model_.storage_discharge[storage]
            * dt
            / data.storage_efficiency_discharge[storage]
        )

    if balancing_cap_mw is not None and (renewables or storages):
        model.balancing_cap = pyo.Constraint(
            expr=sum(model.curtailment[g] for g in model.RENEWABLE_GEN)
            + sum(model.storage_charge[k] for k in model.STORAGE)
            <= float(balancing_cap_mw)
        )

    model.storage_energy_minimum = pyo.Constraint(
        model.STORAGE,
        rule=lambda m, k: energy_after(m, k) >= data.storage_energy_min[k],
    )
    model.storage_energy_maximum = pyo.Constraint(
        model.STORAGE,
        rule=lambda m, k: energy_after(m, k) <= data.storage_energy_max[k],
    )
    _add_reserve_requirements(
        model, data, max(0.0, float(forecast_up_requirement)), max(0.0, float(forecast_down_requirement)),
        float(reserve_margin_mw), RESERVE_SLACK_WEIGHT)
    model.preview_slack_penalty = pyo.Expression(expr=0.0)

    generators_at_bus = {bus: [] for bus in buses}
    renewables_at_bus = {bus: [] for bus in buses}
    loads_at_bus = {bus: [] for bus in buses}
    storages_at_bus = {bus: [] for bus in buses}
    for generator, bus in data.generator_bus.items():
        if bus in generators_at_bus:
            generators_at_bus[bus].append(generator)
            if generator in renewable_set:
                renewables_at_bus[bus].append(generator)
    for load, bus in data.load_bus.items():
        if bus in loads_at_bus:
            loads_at_bus[bus].append(load)
    for storage, bus in data.storage_bus.items():
        if bus in storages_at_bus:
            storages_at_bus[bus].append(storage)

    def p_network(model_, bus):
        return data.base_mva * sum(
            voltage(model_, bus)
            * voltage(model_, neighbor)
            * (
                data.conductance[bus][neighbor]
                * pyo.cos(angle(model_, bus) - angle(model_, neighbor))
                + data.susceptance[bus][neighbor]
                * pyo.sin(angle(model_, bus) - angle(model_, neighbor))
            )
            for neighbor in data.neighbors[bus]
        )

    def q_network(model_, bus):
        return data.base_mva * sum(
            voltage(model_, bus)
            * voltage(model_, neighbor)
            * (
                data.conductance[bus][neighbor]
                * pyo.sin(angle(model_, bus) - angle(model_, neighbor))
                - data.susceptance[bus][neighbor]
                * pyo.cos(angle(model_, bus) - angle(model_, neighbor))
            )
            for neighbor in data.neighbors[bus]
        )

    def p_balance_rule(model_, bus):
        injection = (
            sum(model_.generator_p[g] for g in generators_at_bus[bus])
            - sum(model_.curtailment[g] for g in renewables_at_bus[bus])
            + sum(
                model_.storage_discharge[k] - model_.storage_charge[k]
                for k in storages_at_bus[bus]
            )
            - sum(data.load_p[d] for d in loads_at_bus[bus])
        )
        return injection == p_network(model_, bus)

    def q_balance_rule(model_, bus):
        injection = (
            sum(model_.generator_q[g] for g in generators_at_bus[bus])
            - sum(data.load_q[d] for d in loads_at_bus[bus])
        )
        return injection == q_network(model_, bus)

    model.p_balance = pyo.Constraint(model.BUS, rule=p_balance_rule)
    model.q_balance = pyo.Constraint(model.BUS, rule=q_balance_rule)

    def voltage_real(model_, bus):
        return voltage(model_, bus) * pyo.cos(angle(model_, bus))

    def voltage_imaginary(model_, bus):
        return voltage(model_, bus) * pyo.sin(angle(model_, bus))

    def current_squared(model_, line, from_side):
        branch = data.branches[line]
        e_self = voltage_real(model_, branch.from_bus)
        f_self = voltage_imaginary(model_, branch.from_bus)
        e_other = voltage_real(model_, branch.to_bus)
        f_other = voltage_imaginary(model_, branch.to_bus)
        if from_side:
            y_self, y_other = branch.yff, branch.yft
        else:
            y_self, y_other = branch.ytt, branch.ytf
            e_self, e_other = e_other, e_self
            f_self, f_other = f_other, f_self
        current_real = (
            y_self.real * e_self - y_self.imag * f_self
            + y_other.real * e_other - y_other.imag * f_other
        )
        current_imaginary = (
            y_self.real * f_self + y_self.imag * e_self
            + y_other.real * f_other + y_other.imag * e_other
        )
        return current_real**2 + current_imaginary**2

    model.thermal_slack_from = pyo.Var(
        model.LINE, within=pyo.NonNegativeReals, initialize=0.0
    )
    model.thermal_slack_to = pyo.Var(
        model.LINE, within=pyo.NonNegativeReals, initialize=0.0
    )
    model.worst_thermal_slack = pyo.Var(
        within=pyo.NonNegativeReals, initialize=max(0.0, 1.0 - rho_target)
    )
    model.worst_from = pyo.Constraint(
        model.LINE,
        rule=lambda m, line: m.thermal_slack_from[line]
        <= m.worst_thermal_slack,
    )
    model.worst_to = pyo.Constraint(
        model.LINE,
        rule=lambda m, line: m.thermal_slack_to[line]
        <= m.worst_thermal_slack,
    )
    model.thermal_from = pyo.Constraint(
        model.LINE,
        rule=lambda m, line: current_squared(m, line, True)
        <= (
            (rho_target + m.thermal_slack_from[line])
            * data.branch_i_max_from_pu[line]
        ) ** 2,
    )
    model.thermal_to = pyo.Constraint(
        model.LINE,
        rule=lambda m, line: current_squared(m, line, False)
        <= (
            (rho_target + m.thermal_slack_to[line])
            * data.branch_i_max_to_pu[line]
        ) ** 2,
    )

    def active_flow_from(model_, line):
        branch = data.branches[line]
        e_f = voltage_real(model_, branch.from_bus)
        f_f = voltage_imaginary(model_, branch.from_bus)
        e_t = voltage_real(model_, branch.to_bus)
        f_t = voltage_imaginary(model_, branch.to_bus)
        i_re = (
            branch.yff.real * e_f - branch.yff.imag * f_f
            + branch.yft.real * e_t - branch.yft.imag * f_t
        )
        i_im = (
            branch.yff.real * f_f + branch.yff.imag * e_f
            + branch.yft.real * f_t + branch.yft.imag * e_t
        )
        return data.base_mva * (e_f * i_re + f_f * i_im)

    model.active_flow_from = active_flow_from

    contingency_terms = [
        t for t in (contingency_terms or [])
        if int(t["outage"]) in data.branches and t.get("monitored")
    ]
    ct_index = [
        (i, int(k)) for i, t in enumerate(contingency_terms)
        for k in t["monitored"] if int(k) in data.branches and int(k) != int(t["outage"])
    ]
    model.CT = pyo.Set(initialize=ct_index, dimen=2, ordered=True)
    model.ct_slack = pyo.Var(model.CT, within=pyo.NonNegativeReals, initialize=0.0)

    def _ct_flow(m, i, k):
        t = contingency_terms[i]
        return active_flow_from(m, k) + float(t["monitored"][k]["lodf"]) * active_flow_from(m, int(t["outage"]))

    model.ct_upper = pyo.Constraint(
        model.CT,
        rule=lambda m, i, k: _ct_flow(m, i, k)
        <= (1.0 + m.ct_slack[i, k]) * float(contingency_terms[i]["monitored"][k]["pmax_mw"]),
    )
    model.ct_lower = pyo.Constraint(
        model.CT,
        rule=lambda m, i, k: _ct_flow(m, i, k)
        >= -(1.0 + m.ct_slack[i, k]) * float(contingency_terms[i]["monitored"][k]["pmax_mw"]),
    )
    contingency_term = float(contingency_weight) * sum(
        float(contingency_terms[i]["weight"]) * model.ct_slack[i, k] for (i, k) in model.CT
    )

    attack_lines = dict(attack_lines or {})
    attack_lines = {int(l): float(t) for l, t in attack_lines.items() if int(l) in data.branches}
    model.ATTACK = pyo.Set(initialize=sorted(attack_lines), ordered=True)
    model.attack_slack = pyo.Var(model.ATTACK, within=pyo.NonNegativeReals, initialize=0.0)
    model.attack_from = pyo.Constraint(
        model.ATTACK,
        rule=lambda m, line: current_squared(m, line, True)
        <= ((attack_lines[line] + m.attack_slack[line]) * data.branch_i_max_from_pu[line]) ** 2,
    )
    attack_term = float(attack_weight) * sum(model.attack_slack[l] for l in model.ATTACK)

    model.stage1_objective = pyo.Objective(
        expr=model.worst_thermal_slack
        + 1e-4
        * sum(
            model.thermal_slack_from[line] + model.thermal_slack_to[line]
            for line in model.LINE
        ),
        sense=pyo.minimize,
    )

    model.stage2_objective = pyo.Objective(
        expr=sum(
            model.redispatch_up[g] + model.redispatch_down[g]
            for g in model.REDISPATCHABLE_GEN
        )
        + float(storage_weight)
        * sum(
            model.storage_charge[k] + model.storage_discharge[k]
            for k in model.STORAGE
        )
        + float(curtailment_weight) * sum(model.curtailment[g] for g in model.RENEWABLE_GEN)
        + float(recovery_weight)
        * sum(
            model.dispatch_target_abs[g]
            for g in model.REDISPATCHABLE_GEN
        )
        - float(reserve_reward)
        * sum(
            model.reserve_up[g] + model.reserve_down[g]
            for g in model.REDISPATCHABLE_GEN
        )
        + 1e-3
        * sum(
            (model.voltage[bus] - data.voltage_0[bus]) ** 2
            for bus in model.PQ_BUS
        )
        + attack_term
        + contingency_term
        + model.reserve_slack_penalty
        + model.preview_slack_penalty,
        sense=pyo.minimize,
    )
    model.stage2_objective.deactivate()

    model._acopf_data = data
    model._rho_target = float(rho_target)
    model._reserve_requirements = (required_up, required_down)
    return model


def initialization_twostage_mod(model, data=None):
    if data is None:
        data = model._acopf_data
    initialization(model, data)

    for generator in model.REDISPATCHABLE_GEN:
        model.redispatch_up[generator].value = 0.0
        model.redispatch_down[generator].value = 0.0
        model.generator_p[generator].value = data.generator_p_0[int(generator)]
        generator = int(generator)
        model.dispatch_target_abs[generator].value = abs(
            data.generator_p_0[generator]
            - data.generator_p_reference[generator]
        )
        model.reserve_up[generator].value = max(
            0.0,
            min(
                data.generator_ramp_up[generator],
                data.generator_p_max[generator] - data.generator_p_0[generator],
            ),
        )
        model.reserve_down[generator].value = max(
            0.0,
            min(
                data.generator_ramp_down[generator],
                data.generator_p_0[generator] - data.generator_p_min[generator],
            ),
        )
    for generator in model.RENEWABLE_GEN:
        model.curtailment[generator].value = 0.0
    for storage in model.STORAGE:
        power = data.storage_power_0[int(storage)]
        model.storage_charge[storage].value = max(0.0, power)
        model.storage_discharge[storage].value = max(0.0, -power)

    for line in model.LINE:
        model.thermal_slack_from[line].value = 0.0
        model.thermal_slack_to[line].value = 0.0
    model.worst_thermal_slack.value = max(0.0, 1.0 - model._rho_target)
    if hasattr(model, "reserve_slack_up") and not model.reserve_slack_up.fixed:
        model.reserve_slack_up.value = 0.0
        model.reserve_slack_down.value = 0.0
    return model


def _solver(solver_name, options):
    solver = pyo.SolverFactory(solver_name)
    if not solver.available(exception_flag=False):
        raise RuntimeError(f"The requested solver is not available: {solver_name}")
    solver_options = dict(DEFAULT_IPOPT_OPTIONS)
    if options:
        solver_options.update(options)
    for key, value in solver_options.items():
        solver.options[key] = value
    return solver


import tempfile as _tempfile
from pyomo.common.tempfiles import TempfileManager as _TFM


_SCRATCH = {"dir": None}


def _scratch_dir():
    d = _SCRATCH["dir"]
    if d is not None and _os.path.isdir(d):
        return d
    base = _TFM.tempdir or _tempfile.gettempdir()
    if _os.path.basename(base).startswith("acopf_"):
        base = _os.path.dirname(base)
    d = _os.path.join(base, f"acopf_{_os.getpid()}")
    try:
        _os.makedirs(d, exist_ok=True)
        _TFM.tempdir = d
        _SCRATCH["dir"] = d
        import atexit, shutil
        atexit.register(lambda: shutil.rmtree(d, ignore_errors=True))
        return d
    except Exception:
        return base


def _scratch_snapshot():
    try:
        d = _scratch_dir()
        return d, {n for n in _os.listdir(d) if ".pyomo." in n}
    except Exception:
        return None, set()


def _scratch_release(snapshot):
    d, before = snapshot
    if d is None:
        return
    try:
        for n in _os.listdir(d):
            if ".pyomo." in n and n not in before:
                try:
                    _os.remove(_os.path.join(d, n))
                except OSError:
                    pass
    except Exception:
        pass


def _solve_without_raising(solver, model, tee):
    snap = _scratch_snapshot()
    try:
        return _solve_without_raising_inner(solver, model, tee)
    finally:
        _scratch_release(snap)


def _solve_without_raising_inner(solver, model, tee):
    try:
        results = solver.solve(model, tee=tee, load_solutions=False)
    except Exception:
        results = SolverResults()
        results.solver.status = pyo.SolverStatus.error
        results.solver.termination_condition = pyo.TerminationCondition.error
        return results
    if results.solver.termination_condition in ACCEPTABLE_TERMINATIONS:
        try:
            model.solutions.load_from(results)
        except Exception:
            results.solver.status = pyo.SolverStatus.error
            results.solver.termination_condition = pyo.TerminationCondition.error
    return results


import os as _os

SINGLE_PASS = _os.environ.get("ACOPF_SINGLE_PASS", "1") == "1"
SINGLE_PASS_WEIGHT = float(_os.environ.get("ACOPF_SINGLE_PASS_WEIGHT", "1e5"))


def _solve_single_pass(model, solver, tee, weight):
    model.stage1_objective.deactivate()
    model.stage2_objective.deactivate()
    if hasattr(model, "single_pass_objective"):
        model.del_component(model.single_pass_objective)
    model.single_pass_objective = pyo.Objective(
        expr=float(weight) * model.stage1_objective.expr + model.stage2_objective.expr,
        sense=pyo.minimize,
    )
    results = _solve_without_raising(solver, model, tee)
    status = results.solver.termination_condition
    model._stage1_results = results
    model._stage1_status = status
    model._stage2_results = results
    model._stage2_status = status if status in ACCEPTABLE_TERMINATIONS else None
    if status in ACCEPTABLE_TERMINATIONS:
        model._stage1_worst_slack = max(0.0, float(pyo.value(model.worst_thermal_slack)))
    return results


def solve_acopf_twostage_mod(
    model,
    tee=True,
    solver_name="ipopt",
    options=None,
    stage_tolerance=1e-5,
):
    solver = _solver(solver_name, options)
    if SINGLE_PASS:
        return _solve_single_pass(model, solver, tee, SINGLE_PASS_WEIGHT)
    stage1 = _solve_without_raising(solver, model, tee)
    stage1_status = stage1.solver.termination_condition
    model._stage1_results = stage1
    model._stage1_status = stage1_status
    if stage1_status not in ACCEPTABLE_TERMINATIONS:
        model._stage2_status = None
        return stage1

    best_slack = max(0.0, float(pyo.value(model.worst_thermal_slack)))
    model._stage1_worst_slack = best_slack
    model.stage1_objective.deactivate()
    model.stage2_feasibility = pyo.Constraint(
        expr=model.worst_thermal_slack <= best_slack + stage_tolerance
    )
    model.stage2_objective.activate()
    stage2 = _solve_without_raising(solver, model, tee)
    if stage2.solver.termination_condition not in ACCEPTABLE_TERMINATIONS:
        model._stage2_results = stage2
        model._stage2_status = stage2.solver.termination_condition
        return stage1
    model._stage2_results = stage2
    model._stage2_status = stage2.solver.termination_condition
    return stage2


CURTAIL_RELEASE_STEP = float(_os.environ.get("ACOPF_CURTAIL_RELEASE_STEP", "0.25"))
DECODE_RELEASE = _os.environ.get("ACOPF_DECODE_RELEASE", "0") == "1"
CURTAIL_RELEASE_MAX_RHO = float(_os.environ.get("ACOPF_CURTAIL_RELEASE_MAX_RHO", "1.0"))


def decode_action_twostage_mod(env, obs, model, tolerance_mw=0.05):
    data = model._acopf_data
    action_dictionary = {}

    redispatch = []
    for generator in model.REDISPATCHABLE_GEN:
        generator = int(generator)
        desired_generation = float(pyo.value(model.generator_p[generator]))
        desired_target = desired_generation - data.generator_p_reference[generator]
        command = desired_target - data.generator_target_dispatch[generator]
        command = float(
            np.clip(
                command,
                -0.999 * data.generator_ramp_down[generator],
                0.999 * data.generator_ramp_up[generator],
            )
        )
        if abs(command) > tolerance_mw:
            redispatch.append((generator, command))
    if redispatch:
        action_dictionary["redispatch"] = redispatch

    curtailment = []
    limits = np.asarray(getattr(obs, "curtailment_limit", np.ones(0)), dtype=float)
    rho_now = float(np.nanmax(np.asarray(getattr(obs, "rho", [0.0]), dtype=float)))
    may_release = DECODE_RELEASE and rho_now <= CURTAIL_RELEASE_MAX_RHO
    for generator in model.RENEWABLE_GEN:
        generator = int(generator)
        curtailed_mw = float(pyo.value(model.curtailment[generator]))
        if curtailed_mw <= tolerance_mw:
            if may_release and generator < limits.size and limits[generator] < 1.0 - 1e-6:
                curtailment.append((generator, float(min(1.0, limits[generator] + CURTAIL_RELEASE_STEP))))
            continue
        desired_output = max(0.0, data.generator_p_0[generator] - curtailed_mw)
        pmax = data.generator_p_max[generator]
        if pmax > 0.0:
            curtailment.append((generator, min(1.0, desired_output / pmax)))
    if curtailment:
        action_dictionary["curtail"] = curtailment

    storage_commands = []
    for storage in model.STORAGE:
        storage = int(storage)
        command = float(
            pyo.value(model.storage_charge[storage])
            - pyo.value(model.storage_discharge[storage])
        )
        if abs(command) > tolerance_mw:
            storage_commands.append((storage, command))
    if storage_commands:
        action_dictionary["set_storage"] = storage_commands

    return env.action_space(action_dictionary), action_dictionary


def twostage_diagnostics_mod(model):
    return SimpleNamespace(
        rho_target=model._rho_target,
        stage1_status=getattr(model, "_stage1_status", None),
        stage2_status=getattr(model, "_stage2_status", None),
        worst_slack=float(
            getattr(
                model,
                "_stage1_worst_slack",
                pyo.value(model.worst_thermal_slack),
            )
        ),
        reserve_requirements=getattr(model, "_reserve_requirements", (0.0, 0.0)),
        reserve_up=sum(float(pyo.value(model.reserve_up[g])) for g in model.REDISPATCHABLE_GEN),
        reserve_down=sum(float(pyo.value(model.reserve_down[g])) for g in model.REDISPATCHABLE_GEN),
    )


MODEL_REUSE = _os.environ.get("ACOPF_MODEL_REUSE", "1") == "1"
WARM_START = _os.environ.get("ACOPF_WARM_START", "1") == "1"
WARM_START_DUALS = _os.environ.get("ACOPF_WARM_START_DUALS", "1") == "1"
WARM_START_MU_INIT = float(_os.environ.get("ACOPF_WARM_START_MU_INIT", "1e-3"))
WARM_START_OPTIONS = {
    "warm_start_init_point": "yes",
    "warm_start_bound_push": 1e-6,
    "warm_start_bound_frac": 1e-6,
    "warm_start_slack_bound_push": 1e-6,
    "warm_start_slack_bound_frac": 1e-6,
    "warm_start_mult_bound_push": 1e-6,
    "mu_strategy": "monotone",
    "mu_init": WARM_START_MU_INIT,
}

_CACHE = {"key": None, "model": None, "warm": False, "hits": 0, "builds": 0}
_SOLVERS = {}


def _r(value, digits=9):
    return round(float(value), digits)


def _structure_key(data, ct_index, attack_lines, contingency_terms, enforce_generator_q_limits):
    branches = tuple(
        (int(l), int(b.from_bus), int(b.to_bus), _r(b.yff.real), _r(b.yff.imag), _r(b.yft.real), _r(b.yft.imag),
         _r(b.ytt.real), _r(b.ytt.imag), _r(b.ytf.real), _r(b.ytf.imag),
         _r(data.branch_i_max_from_pu[l]), _r(data.branch_i_max_to_pu[l]))
        for l, b in sorted(data.branches.items())
    )
    admittance = tuple(
        (int(bus), tuple((int(n), _r(data.conductance[bus][n]), _r(data.susceptance[bus][n])) for n in data.neighbors[bus]))
        for bus in data.buses
    )
    gens = tuple(
        (int(g), int(data.generator_bus[g]), _r(data.generator_p_min[g]), _r(data.generator_p_max[g]),
         _r(data.generator_ramp_up[g]), _r(data.generator_ramp_down[g]), _r(data.generator_q_min[g]), _r(data.generator_q_max[g]))
        for g in data.generators
    )
    storages = tuple(
        (int(k), int(data.storage_bus[k]), _r(data.storage_energy_min[k]), _r(data.storage_energy_max[k]),
         _r(data.storage_p_charge_max[k]), _r(data.storage_p_discharge_max[k]), _r(data.storage_efficiency_charge[k]),
         _r(data.storage_efficiency_discharge[k]), _r(data.storage_loss[k]))
        for k in data.storages
    )
    return (
        tuple(int(b) for b in data.buses), int(data.slack_bus), tuple(int(b) for b in data.pq_buses),
        tuple(int(g) for g in data.redispatchable_generators), tuple(int(g) for g in data.renewable_generators),
        tuple((int(d), int(data.load_bus[d])) for d in data.loads), _r(data.base_mva), _r(data.delta_time_hours),
        branches, admittance, gens, storages, tuple(ct_index), tuple(sorted(int(l) for l in (attack_lines or {}))),
        tuple(
            (int(t["outage"]), tuple(sorted(int(k) for k in t["monitored"])))
            for t in (contingency_terms or []) if int(t["outage"]) in data.branches and t.get("monitored")
        ),
        bool(enforce_generator_q_limits),
    )


def _filtered_contingency_terms(data, contingency_terms):
    terms = [
        t for t in (contingency_terms or [])
        if int(t["outage"]) in data.branches and t.get("monitored")
    ]
    index = [
        (i, int(k)) for i, t in enumerate(terms)
        for k in t["monitored"] if int(k) in data.branches and int(k) != int(t["outage"])
    ]
    return terms, index


def _build_parametric_model(data, ct_index, contingency_terms, attack_lines, enforce_generator_q_limits):
    model = pyo.ConcreteModel()
    buses = list(data.buses)
    non_slack = [bus for bus in buses if bus != data.slack_bus]
    pq_buses = list(data.pq_buses)
    pq_set = set(pq_buses)
    generators = list(data.generators)
    redispatchable = list(data.redispatchable_generators)
    redispatchable_set = set(redispatchable)
    fixed_generators = [g for g in generators if g not in redispatchable_set]
    renewables = list(data.renewable_generators)
    renewable_set = set(renewables)
    storages = list(data.storages)
    loads = list(data.loads)
    branches = sorted(data.branches)
    slack = int(data.slack_bus)

    model.BUS = pyo.Set(initialize=buses, ordered=True)
    model.NON_SLACK = pyo.Set(initialize=non_slack, ordered=True)
    model.PQ_BUS = pyo.Set(initialize=pq_buses, ordered=True)
    model.GEN = pyo.Set(initialize=generators, ordered=True)
    model.REDISPATCHABLE_GEN = pyo.Set(initialize=redispatchable, ordered=True)
    model.FIXED_GEN = pyo.Set(initialize=fixed_generators, ordered=True)
    model.RENEWABLE_GEN = pyo.Set(initialize=renewables, ordered=True)
    model.STORAGE = pyo.Set(initialize=storages, ordered=True)
    model.LINE = pyo.Set(initialize=branches, ordered=True)
    model.LOAD = pyo.Set(initialize=loads, ordered=True)

    P = dict(mutable=True, within=pyo.Reals)
    model.v0 = pyo.Param(model.BUS, initialize=lambda _m, b: float(data.voltage_0[b]), **P)
    model.a0 = pyo.Param(model.BUS, initialize=lambda _m, b: float(data.angle_0[b]), **P)
    model.p0 = pyo.Param(model.GEN, initialize=lambda _m, g: float(data.generator_p_0[g]), **P)
    model.pref = pyo.Param(model.REDISPATCHABLE_GEN, initialize=lambda _m, g: float(data.generator_p_reference[g]), **P)
    model.load_p = pyo.Param(model.LOAD, initialize=lambda _m, d: float(data.load_p[d]), **P)
    model.load_q = pyo.Param(model.LOAD, initialize=lambda _m, d: float(data.load_q[d]), **P)
    model.storage_energy0 = pyo.Param(model.STORAGE, initialize=lambda _m, k: float(data.storage_energy[k]), **P)
    model.rho_target = pyo.Param(initialize=0.95, **P)
    model.required_up = pyo.Param(initialize=0.0, **P)
    model.required_down = pyo.Param(initialize=0.0, **P)
    model.forecast_up = pyo.Param(initialize=0.0, **P)
    model.forecast_down = pyo.Param(initialize=0.0, **P)
    model.reserve_margin = pyo.Param(initialize=0.0, **P)
    model.reserve_slack_weight = pyo.Param(initialize=RESERVE_SLACK_WEIGHT, **P)
    model.preview_up = pyo.Param(initialize=0.0, **P)
    model.preview_down = pyo.Param(initialize=0.0, **P)
    model.balancing_cap_mw = pyo.Param(initialize=1e6, **P)
    model.curtailment_weight = pyo.Param(initialize=5.0, **P)
    model.storage_weight = pyo.Param(initialize=0.25, **P)
    model.recovery_weight = pyo.Param(initialize=0.35, **P)
    model.reserve_reward = pyo.Param(initialize=1e-3, **P)
    model.attack_weight = pyo.Param(initialize=200.0, **P)
    model.contingency_weight = pyo.Param(initialize=200.0, **P)

    model.theta = pyo.Var(model.NON_SLACK, initialize=lambda _m, b: float(data.angle_0[b]))
    model.voltage = pyo.Var(model.PQ_BUS, within=pyo.NonNegativeReals, initialize=lambda _m, b: float(data.voltage_0[b]))

    def voltage(m, bus):
        return m.voltage[bus] if bus in pq_set else m.v0[bus]

    def angle(m, bus):
        return m.theta[bus] if bus != slack else m.a0[slack]

    model.generator_p = pyo.Var(
        model.GEN, within=pyo.NonNegativeReals,
        bounds=lambda _m, g: (data.generator_p_min[g], data.generator_p_max[g]),
        initialize=lambda _m, g: float(data.generator_p_0[g]),
    )
    model.redispatch_up = pyo.Var(model.REDISPATCHABLE_GEN, within=pyo.NonNegativeReals,
                                  bounds=lambda _m, g: (0.0, data.generator_ramp_up[g]), initialize=0.0)
    model.redispatch_down = pyo.Var(model.REDISPATCHABLE_GEN, within=pyo.NonNegativeReals,
                                    bounds=lambda _m, g: (0.0, data.generator_ramp_down[g]), initialize=0.0)
    model.redispatch_definition = pyo.Constraint(
        model.REDISPATCHABLE_GEN,
        rule=lambda m, g: m.generator_p[g] == m.p0[g] + m.redispatch_up[g] - m.redispatch_down[g],
    )
    model.reserve_up = pyo.Var(model.REDISPATCHABLE_GEN, within=pyo.NonNegativeReals)
    model.reserve_down = pyo.Var(model.REDISPATCHABLE_GEN, within=pyo.NonNegativeReals)
    _add_reserve_ramp_limits(model, data)

    model.dispatch_target_abs = pyo.Var(model.REDISPATCHABLE_GEN, within=pyo.NonNegativeReals)
    model.dispatch_target_abs_positive = pyo.Constraint(model.REDISPATCHABLE_GEN, rule=lambda m, g: m.dispatch_target_abs[g] >= m.generator_p[g] - m.pref[g])
    model.dispatch_target_abs_negative = pyo.Constraint(model.REDISPATCHABLE_GEN, rule=lambda m, g: m.dispatch_target_abs[g] >= -(m.generator_p[g] - m.pref[g]))
    model.fixed_generator_output = pyo.Constraint(model.FIXED_GEN, rule=lambda m, g: m.generator_p[g] == m.p0[g])

    def generator_q_bounds(_m, g):
        if enforce_generator_q_limits:
            return data.generator_q_min[g], data.generator_q_max[g]
        return -1e4, 1e4

    model.generator_q = pyo.Var(model.GEN, bounds=generator_q_bounds, initialize=lambda _m, g: data.generator_q_0[g])
    model.curtailment = pyo.Var(model.RENEWABLE_GEN, within=pyo.NonNegativeReals, initialize=0.0)
    model.storage_charge = pyo.Var(model.STORAGE, within=pyo.NonNegativeReals,
                                   bounds=lambda _m, k: (0.0, data.storage_p_charge_max[k]),
                                   initialize=lambda _m, k: max(0.0, data.storage_power_0[k]))
    model.storage_discharge = pyo.Var(model.STORAGE, within=pyo.NonNegativeReals,
                                      bounds=lambda _m, k: (0.0, data.storage_p_discharge_max[k]),
                                      initialize=lambda _m, k: max(0.0, -data.storage_power_0[k]))

    def energy_after(m, k):
        dt = data.delta_time_hours
        return (m.storage_energy0[k] - data.storage_loss[k] * dt
                + data.storage_efficiency_charge[k] * m.storage_charge[k] * dt
                - m.storage_discharge[k] * dt / data.storage_efficiency_discharge[k])

    if renewables or storages:
        model.balancing_cap = pyo.Constraint(
            expr=sum(model.curtailment[g] for g in model.RENEWABLE_GEN)
            + sum(model.storage_charge[k] for k in model.STORAGE) <= model.balancing_cap_mw)
    model.storage_energy_minimum = pyo.Constraint(model.STORAGE, rule=lambda m, k: energy_after(m, k) >= data.storage_energy_min[k])
    model.storage_energy_maximum = pyo.Constraint(model.STORAGE, rule=lambda m, k: energy_after(m, k) <= data.storage_energy_max[k])
    _add_reserve_requirements(model, data, model.forecast_up, model.forecast_down, model.reserve_margin, model.reserve_slack_weight)
    _add_preview_requirements(model, model.preview_up, model.preview_down)

    generators_at_bus = {bus: [] for bus in buses}
    renewables_at_bus = {bus: [] for bus in buses}
    loads_at_bus = {bus: [] for bus in buses}
    storages_at_bus = {bus: [] for bus in buses}
    for g, bus in data.generator_bus.items():
        if bus in generators_at_bus:
            generators_at_bus[bus].append(g)
            if g in renewable_set:
                renewables_at_bus[bus].append(g)
    for d, bus in data.load_bus.items():
        if bus in loads_at_bus:
            loads_at_bus[bus].append(d)
    for k, bus in data.storage_bus.items():
        if bus in storages_at_bus:
            storages_at_bus[bus].append(k)

    def p_network(m, bus):
        return data.base_mva * sum(
            voltage(m, bus) * voltage(m, n) * (
                data.conductance[bus][n] * pyo.cos(angle(m, bus) - angle(m, n))
                + data.susceptance[bus][n] * pyo.sin(angle(m, bus) - angle(m, n)))
            for n in data.neighbors[bus])

    def q_network(m, bus):
        return data.base_mva * sum(
            voltage(m, bus) * voltage(m, n) * (
                data.conductance[bus][n] * pyo.sin(angle(m, bus) - angle(m, n))
                - data.susceptance[bus][n] * pyo.cos(angle(m, bus) - angle(m, n)))
            for n in data.neighbors[bus])

    def p_balance_rule(m, bus):
        injection = (sum(m.generator_p[g] for g in generators_at_bus[bus])
                     - sum(m.curtailment[g] for g in renewables_at_bus[bus])
                     + sum(m.storage_discharge[k] - m.storage_charge[k] for k in storages_at_bus[bus])
                     - sum(m.load_p[d] for d in loads_at_bus[bus]))
        return injection == p_network(m, bus)

    def q_balance_rule(m, bus):
        injection = (sum(m.generator_q[g] for g in generators_at_bus[bus]) - sum(m.load_q[d] for d in loads_at_bus[bus]))
        return injection == q_network(m, bus)

    model.p_balance = pyo.Constraint(model.BUS, rule=p_balance_rule)
    model.q_balance = pyo.Constraint(model.BUS, rule=q_balance_rule)

    def voltage_real(m, bus):
        return voltage(m, bus) * pyo.cos(angle(m, bus))

    def voltage_imaginary(m, bus):
        return voltage(m, bus) * pyo.sin(angle(m, bus))

    def current_squared(m, line, from_side):
        branch = data.branches[line]
        e_self = voltage_real(m, branch.from_bus); f_self = voltage_imaginary(m, branch.from_bus)
        e_other = voltage_real(m, branch.to_bus); f_other = voltage_imaginary(m, branch.to_bus)
        if from_side:
            y_self, y_other = branch.yff, branch.yft
        else:
            y_self, y_other = branch.ytt, branch.ytf
            e_self, e_other = e_other, e_self
            f_self, f_other = f_other, f_self
        current_real = (y_self.real * e_self - y_self.imag * f_self + y_other.real * e_other - y_other.imag * f_other)
        current_imaginary = (y_self.real * f_self + y_self.imag * e_self + y_other.real * f_other + y_other.imag * e_other)
        return current_real ** 2 + current_imaginary ** 2

    model.thermal_slack_from = pyo.Var(model.LINE, within=pyo.NonNegativeReals, initialize=0.0)
    model.thermal_slack_to = pyo.Var(model.LINE, within=pyo.NonNegativeReals, initialize=0.0)
    model.worst_thermal_slack = pyo.Var(within=pyo.NonNegativeReals, initialize=0.05)
    model.worst_from = pyo.Constraint(model.LINE, rule=lambda m, l: m.thermal_slack_from[l] <= m.worst_thermal_slack)
    model.worst_to = pyo.Constraint(model.LINE, rule=lambda m, l: m.thermal_slack_to[l] <= m.worst_thermal_slack)
    model.thermal_from = pyo.Constraint(
        model.LINE, rule=lambda m, l: current_squared(m, l, True)
        <= ((m.rho_target + m.thermal_slack_from[l]) * data.branch_i_max_from_pu[l]) ** 2)
    model.thermal_to = pyo.Constraint(
        model.LINE, rule=lambda m, l: current_squared(m, l, False)
        <= ((m.rho_target + m.thermal_slack_to[l]) * data.branch_i_max_to_pu[l]) ** 2)

    def active_flow_from(m, line):
        branch = data.branches[line]
        e_f = voltage_real(m, branch.from_bus); f_f = voltage_imaginary(m, branch.from_bus)
        e_t = voltage_real(m, branch.to_bus); f_t = voltage_imaginary(m, branch.to_bus)
        i_re = (branch.yff.real * e_f - branch.yff.imag * f_f + branch.yft.real * e_t - branch.yft.imag * f_t)
        i_im = (branch.yff.real * f_f + branch.yff.imag * e_f + branch.yft.real * f_t + branch.yft.imag * e_t)
        return data.base_mva * (e_f * i_re + f_f * i_im)

    model.active_flow_from = active_flow_from

    model.CT = pyo.Set(initialize=list(ct_index), dimen=2, ordered=True)
    model.ct_slack = pyo.Var(model.CT, within=pyo.NonNegativeReals, initialize=0.0)
    model.ct_lodf = pyo.Param(model.CT, initialize=lambda _m, i, k: float(contingency_terms[i]["monitored"][k]["lodf"]), **P)
    model.ct_pmax = pyo.Param(model.CT, initialize=lambda _m, i, k: float(contingency_terms[i]["monitored"][k]["pmax_mw"]), **P)
    model.ct_weight = pyo.Param(model.CT, initialize=lambda _m, i, k: float(contingency_terms[i]["weight"]), **P)
    ct_outage = {i: int(t["outage"]) for i, t in enumerate(contingency_terms)}

    def _ct_flow(m, i, k):
        return active_flow_from(m, k) + m.ct_lodf[i, k] * active_flow_from(m, ct_outage[i])

    model.ct_upper = pyo.Constraint(model.CT, rule=lambda m, i, k: _ct_flow(m, i, k) <= (1.0 + m.ct_slack[i, k]) * m.ct_pmax[i, k])
    model.ct_lower = pyo.Constraint(model.CT, rule=lambda m, i, k: _ct_flow(m, i, k) >= -(1.0 + m.ct_slack[i, k]) * m.ct_pmax[i, k])
    contingency_term = model.contingency_weight * sum(model.ct_weight[i, k] * model.ct_slack[i, k] for (i, k) in model.CT)

    attack = {int(l): float(t) for l, t in (attack_lines or {}).items() if int(l) in data.branches}
    model.ATTACK = pyo.Set(initialize=sorted(attack), ordered=True)
    model.attack_threshold = pyo.Param(model.ATTACK, initialize=lambda _m, l: attack[l], **P)
    model.attack_slack = pyo.Var(model.ATTACK, within=pyo.NonNegativeReals, initialize=0.0)
    model.attack_from = pyo.Constraint(
        model.ATTACK, rule=lambda m, l: current_squared(m, l, True)
        <= ((m.attack_threshold[l] + m.attack_slack[l]) * data.branch_i_max_from_pu[l]) ** 2)
    attack_term = model.attack_weight * sum(model.attack_slack[l] for l in model.ATTACK)

    model.stage1_objective = pyo.Objective(
        expr=model.worst_thermal_slack + 1e-4 * sum(model.thermal_slack_from[l] + model.thermal_slack_to[l] for l in model.LINE),
        sense=pyo.minimize)
    model.stage2_objective = pyo.Objective(
        expr=sum(model.redispatch_up[g] + model.redispatch_down[g] for g in model.REDISPATCHABLE_GEN)
        + model.storage_weight * sum(model.storage_charge[k] + model.storage_discharge[k] for k in model.STORAGE)
        + model.curtailment_weight * sum(model.curtailment[g] for g in model.RENEWABLE_GEN)
        + model.recovery_weight * sum(model.dispatch_target_abs[g] for g in model.REDISPATCHABLE_GEN)
        - model.reserve_reward * sum(model.reserve_up[g] + model.reserve_down[g] for g in model.REDISPATCHABLE_GEN)
        + 1e-3 * sum((model.voltage[b] - model.v0[b]) ** 2 for b in model.PQ_BUS)
        + attack_term + contingency_term + model.reserve_slack_penalty + model.preview_slack_penalty,
        sense=pyo.minimize)
    model.stage2_objective.deactivate()
    if SINGLE_PASS:
        model.stage1_objective.deactivate()
        model.single_pass_objective = pyo.Objective(
            expr=float(SINGLE_PASS_WEIGHT) * model.stage1_objective.expr + model.stage2_objective.expr, sense=pyo.minimize)

    model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zL_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    model.ipopt_zU_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
    model._parametric = True
    return model


def _update_parameters(
    model, data, *, rho_target, voltage_min, voltage_max, voltage_trust_region, angle_trust_region,
    forecast_up_requirement, forecast_down_requirement, reserve_margin_mw, recovery_weight, reserve_reward,
    balancing_cap_mw, attack_lines, attack_weight, curtailment_weight, contingency_terms, contingency_weight,
    storage_weight=0.25, preview_up=0.0, preview_down=0.0,
):
    for bus in model.BUS:
        model.v0[bus] = float(data.voltage_0[bus])
        model.a0[bus] = float(data.angle_0[bus])
    for bus in model.NON_SLACK:
        a = float(data.angle_0[bus])
        model.theta[bus].setlb(max(-np.pi, a - angle_trust_region))
        model.theta[bus].setub(min(np.pi, a + angle_trust_region))
    for bus in model.PQ_BUS:
        v = float(data.voltage_0[bus])
        model.voltage[bus].setlb(max(min(voltage_min, v), v - voltage_trust_region))
        model.voltage[bus].setub(min(max(voltage_max, v), v + voltage_trust_region))
    for g in model.GEN:
        model.p0[g] = float(data.generator_p_0[g])
    for g in model.REDISPATCHABLE_GEN:
        model.pref[g] = float(data.generator_p_reference[g])
    for g in model.RENEWABLE_GEN:
        model.curtailment[g].setub(max(0.0, float(min(data.generator_p_0[g], data.renewable_available_p[g]))))
    for d in model.LOAD:
        model.load_p[d] = float(data.load_p[d])
        model.load_q[d] = float(data.load_q[d])
    for k in model.STORAGE:
        model.storage_energy0[k] = float(data.storage_energy[k])
        c_ub, d_ub = _storage_power_bounds(data, k)
        model.storage_charge[k].setub(c_ub); model.storage_discharge[k].setub(d_ub)
        if model.storage_charge[k].value is not None:
            model.storage_charge[k].value = min(model.storage_charge[k].value, c_ub)
        if model.storage_discharge[k].value is not None:
            model.storage_discharge[k].value = min(model.storage_discharge[k].value, d_ub)
    required_up = max(0.0, float(forecast_up_requirement)) + float(reserve_margin_mw)
    required_down = max(0.0, float(forecast_down_requirement)) + float(reserve_margin_mw)
    model.required_up = required_up
    model.required_down = required_down
    model.forecast_up = max(0.0, float(forecast_up_requirement))
    model.forecast_down = max(0.0, float(forecast_down_requirement))
    model.reserve_margin = float(reserve_margin_mw)
    model.reserve_slack_weight = float(RESERVE_SLACK_WEIGHT)
    model.preview_up = max(0.0, float(preview_up))
    model.preview_down = max(0.0, float(preview_down))
    model.rho_target = float(rho_target)
    model.balancing_cap_mw = float(balancing_cap_mw) if balancing_cap_mw is not None else 1e6
    model.curtailment_weight = float(curtailment_weight)
    model.storage_weight = float(storage_weight)
    model.recovery_weight = float(recovery_weight)
    model.reserve_reward = float(reserve_reward)
    model.attack_weight = float(attack_weight)
    model.contingency_weight = float(contingency_weight)
    for l in model.ATTACK:
        model.attack_threshold[l] = float(attack_lines[l])
    terms, _index = _filtered_contingency_terms(data, contingency_terms)
    for (i, k) in model.CT:
        model.ct_lodf[i, k] = float(terms[i]["monitored"][k]["lodf"])
        model.ct_pmax[i, k] = float(terms[i]["monitored"][k]["pmax_mw"])
        model.ct_weight[i, k] = float(terms[i]["weight"])
    model._acopf_data = data
    model._rho_target = float(rho_target)
    model._reserve_requirements = (required_up, required_down)
    if hasattr(model, "stage2_feasibility"):
        model.del_component(model.stage2_feasibility)
    for name in ("_stage1_results", "_stage1_status", "_stage2_results", "_stage2_status", "_stage1_worst_slack"):
        if hasattr(model, name):
            delattr(model, name)


def _clip(value, low, high):
    return float(min(max(value, low), high))


def _warm_start_from_previous(model, data):
    for bus in model.NON_SLACK:
        v = model.theta[bus]
        v.value = _clip(v.value if v.value is not None else data.angle_0[bus], v.lb, v.ub)
    for bus in model.PQ_BUS:
        v = model.voltage[bus]
        v.value = _clip(v.value if v.value is not None else data.voltage_0[bus], v.lb, v.ub)
    for g in model.FIXED_GEN:
        model.generator_p[g].value = float(data.generator_p_0[g])
    for g in model.REDISPATCHABLE_GEN:
        p0 = float(data.generator_p_0[g])
        low = max(data.generator_p_min[g], p0 - data.generator_ramp_down[g])
        high = min(data.generator_p_max[g], p0 + data.generator_ramp_up[g])
        prev = model.generator_p[g].value
        p = _clip(prev if prev is not None else p0, low, high)
        model.generator_p[g].value = p
        model.redispatch_up[g].value = max(0.0, p - p0)
        model.redispatch_down[g].value = max(0.0, p0 - p)
        model.dispatch_target_abs[g].value = abs(p - float(data.generator_p_reference[g]))
        model.reserve_up[g].value = max(0.0, min(data.generator_ramp_up[g], data.generator_p_max[g] - p))
        model.reserve_down[g].value = max(0.0, min(data.generator_ramp_down[g], p - data.generator_p_min[g]))
    for g in model.RENEWABLE_GEN:
        v = model.curtailment[g]
        v.value = _clip(v.value if v.value is not None else 0.0, 0.0, v.ub)
    for k in model.STORAGE:
        for var in (model.storage_charge[k], model.storage_discharge[k]):
            var.value = _clip(var.value if var.value is not None else 0.0, var.lb, var.ub)
    for g in model.GEN:
        if model.generator_q[g].value is None:
            model.generator_q[g].value = float(data.generator_q_0[g])
    for l in model.LINE:
        for var in (model.thermal_slack_from[l], model.thermal_slack_to[l]):
            var.value = max(0.0, var.value or 0.0)
    model.worst_thermal_slack.value = max(0.0, model.worst_thermal_slack.value or 0.0)
    if WARM_START_DUALS:
        for var in model.component_data_objects(pyo.Var, descend_into=True):
            zl = model.ipopt_zL_out.get(var)
            zu = model.ipopt_zU_out.get(var)
            if zl is not None:
                model.ipopt_zL_in[var] = zl
            if zu is not None:
                model.ipopt_zU_in[var] = zu


def _cached_solver(solver_name, options, warm):
    key = (solver_name, tuple(sorted((options or {}).items())), bool(warm))
    solver = _SOLVERS.get(key)
    if solver is None:
        merged = dict(options or {})
        if warm:
            merged.update(WARM_START_OPTIONS)
        solver = _solver(solver_name, merged)
        _SOLVERS[key] = solver
    return solver


def acopf_solve_cached(
    data,
    rho_target=0.95,
    voltage_min=0.90,
    voltage_max=1.10,
    enforce_generator_q_limits=False,
    voltage_trust_region=0.10,
    angle_trust_region=np.deg2rad(30.0),
    forecast_up_requirement=0.0,
    forecast_down_requirement=0.0,
    reserve_margin_mw=5.0,
    recovery_weight=0.35,
    reserve_reward=1e-3,
    balancing_cap_mw=None,
    attack_lines=None,
    attack_weight=200.0,
    curtailment_weight=5.0,
    contingency_terms=None,
    contingency_weight=200.0,
    tee=False,
    solver_name="ipopt",
    options=None,
    info=None,
    storage_weight=0.25,
    preview_up=0.0,
    preview_down=0.0,
):
    info = info if info is not None else {}
    t0 = time.perf_counter()
    terms, ct_index = _filtered_contingency_terms(data, contingency_terms)
    attack = {int(l): float(t) for l, t in (attack_lines or {}).items() if int(l) in data.branches}
    key = _structure_key(data, ct_index, attack, terms, enforce_generator_q_limits)
    reused = MODEL_REUSE and _CACHE["model"] is not None and _CACHE["key"] == key
    if reused:
        model = _CACHE["model"]
        _CACHE["hits"] += 1
    else:
        model = _build_parametric_model(data, ct_index, terms, attack, enforce_generator_q_limits)
        _CACHE.update(key=key, model=model, warm=False)
        _CACHE["builds"] += 1
    info["key_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    _update_parameters(
        model, data, rho_target=rho_target, voltage_min=voltage_min, voltage_max=voltage_max,
        voltage_trust_region=voltage_trust_region, angle_trust_region=angle_trust_region,
        forecast_up_requirement=forecast_up_requirement, forecast_down_requirement=forecast_down_requirement,
        reserve_margin_mw=reserve_margin_mw, recovery_weight=recovery_weight, reserve_reward=reserve_reward,
        balancing_cap_mw=balancing_cap_mw, attack_lines=attack, attack_weight=attack_weight,
        curtailment_weight=curtailment_weight, contingency_terms=contingency_terms, contingency_weight=contingency_weight,
        storage_weight=storage_weight, preview_up=preview_up, preview_down=preview_down,
    )
    warm = WARM_START and reused and _CACHE["warm"]
    if warm:
        _warm_start_from_previous(model, data)
    else:
        model.ipopt_zL_in.clear()
        model.ipopt_zU_in.clear()
        initialization_twostage_mod(model, data)
    info.update(build_s=time.perf_counter() - t1, reused=reused, warm=warm)
    t2 = time.perf_counter()
    solver = _cached_solver(solver_name, options, warm)
    results = _solve_with(model, solver, tee)
    info["solve_s"] = time.perf_counter() - t2
    status = results.solver.termination_condition
    _CACHE["warm"] = status in ACCEPTABLE_TERMINATIONS
    info["status"] = str(status)
    return model, results


def _solve_with(model, solver, tee):
    if SINGLE_PASS:
        results = _solve_without_raising_opts(solver, model, tee)
        status = results.solver.termination_condition
        model._stage1_results = results
        model._stage1_status = status
        model._stage2_results = results
        model._stage2_status = status if status in ACCEPTABLE_TERMINATIONS else None
        if status in ACCEPTABLE_TERMINATIONS:
            model._stage1_worst_slack = max(0.0, float(pyo.value(model.worst_thermal_slack)))
        return results
    model.stage1_objective.activate()
    model.stage2_objective.deactivate()
    stage1 = _solve_without_raising_opts(solver, model, tee)
    model._stage1_results = stage1
    model._stage1_status = stage1.solver.termination_condition
    if stage1.solver.termination_condition not in ACCEPTABLE_TERMINATIONS:
        model._stage2_status = None
        return stage1
    best_slack = max(0.0, float(pyo.value(model.worst_thermal_slack)))
    model._stage1_worst_slack = best_slack
    model.stage1_objective.deactivate()
    model.stage2_feasibility = pyo.Constraint(expr=model.worst_thermal_slack <= best_slack + 1e-5)
    model.stage2_objective.activate()
    stage2 = _solve_without_raising_opts(solver, model, tee)
    model._stage2_results = stage2
    model._stage2_status = stage2.solver.termination_condition
    return stage2 if stage2.solver.termination_condition in ACCEPTABLE_TERMINATIONS else stage1


def _solve_without_raising_opts(solver, model, tee):
    snap = _scratch_snapshot()
    try:
        return _solve_without_raising_opts_inner(solver, model, tee)
    finally:
        _scratch_release(snap)


def _solve_without_raising_opts_inner(solver, model, tee):
    try:
        results = solver.solve(model, tee=tee, load_solutions=False)
    except Exception:
        results = SolverResults()
        results.solver.status = pyo.SolverStatus.error
        results.solver.termination_condition = pyo.TerminationCondition.error
        return results
    if results.solver.termination_condition in ACCEPTABLE_TERMINATIONS:
        try:
            model.solutions.load_from(results)
        except Exception:
            results.solver.status = pyo.SolverStatus.error
            results.solver.termination_condition = pyo.TerminationCondition.error
    return results


def cache_stats():
    return dict(hits=_CACHE["hits"], builds=_CACHE["builds"])
