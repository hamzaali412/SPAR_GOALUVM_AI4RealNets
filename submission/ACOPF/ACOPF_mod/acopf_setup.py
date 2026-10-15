import os

import numpy as np
import pyomo.environ as pyo

DEFAULT_IPOPT_OPTIONS = {
    "max_iter": 300,
    "max_cpu_time": float(os.environ.get("ACOPF_IPOPT_MAX_CPU_S", "8.0")),
    "tol": 1e-6,
    "constr_viol_tol": 1e-8,
    "acceptable_constr_viol_tol": 1e-8,
    "bound_relax_factor": 0.0,
    "print_frequency_iter": 20,
    "nlp_scaling_method": "gradient-based",
    "mu_strategy": "adaptive",
}


def acopf_setup(
    data,
    voltage_min=0.90,
    voltage_max=1.10,
    enforce_generator_q_limits=False,
):
    model = pyo.ConcreteModel()

    buses = list(data.buses)
    non_slack = [bus for bus in buses if bus != data.slack_bus]
    pq_buses = list(data.pq_buses)
    pq_set = set(pq_buses)
    generators = list(data.generators)
    redispatchable = list(data.redispatchable_generators)
    slack_generators = list(data.slack_generators)
    redispatchable_set = set(redispatchable)
    fixed_generators = [
        generator
        for generator in generators
        if generator not in redispatchable_set
    ]
    branches = sorted(data.branches)

    model.BUS = pyo.Set(initialize=buses, ordered=True)
    model.NON_SLACK = pyo.Set(initialize=non_slack, ordered=True)
    model.PQ_BUS = pyo.Set(initialize=pq_buses, ordered=True)
    model.GEN = pyo.Set(initialize=generators, ordered=True)
    model.REDISPATCHABLE_GEN = pyo.Set(initialize=redispatchable, ordered=True)
    model.SLACK_GEN = pyo.Set(initialize=slack_generators, ordered=True)
    model.FIXED_GEN = pyo.Set(initialize=fixed_generators, ordered=True)
    model.LINE = pyo.Set(initialize=branches, ordered=True)

    model.theta = pyo.Var(
        model.NON_SLACK,
        within=pyo.Reals,
        bounds=(-np.pi, np.pi),
        initialize={bus: data.angle_0[bus] for bus in non_slack},
    )

    def voltage_bounds(_model, bus):
        observed = data.voltage_0[bus]
        return min(voltage_min, observed), max(voltage_max, observed)

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
        bounds=lambda _model, generator: (
            data.generator_p_min[generator],
            data.generator_p_max[generator],
        ),
        initialize=lambda _model, generator: float(
            np.clip(
                data.generator_p_0[generator],
                data.generator_p_min[generator],
                data.generator_p_max[generator],
            )
        ),
    )

    model.generator_ramp_up_limit = pyo.Constraint(
        model.REDISPATCHABLE_GEN,
        rule=lambda model_, generator: model_.generator_p[generator]
        - data.generator_p_0[generator]
        <= data.generator_ramp_up[generator],
    )
    model.generator_ramp_down_limit = pyo.Constraint(
        model.REDISPATCHABLE_GEN,
        rule=lambda model_, generator: data.generator_p_0[generator]
        - model_.generator_p[generator]
        <= data.generator_ramp_down[generator],
    )

    model.fixed_generator_output = pyo.Constraint(
        model.FIXED_GEN,
        rule=lambda model_, generator: model_.generator_p[generator]
        == data.generator_p_0[generator],
    )

    def generator_q_bounds(_model, generator):
        if enforce_generator_q_limits:
            return (
                data.generator_q_min[generator],
                data.generator_q_max[generator],
            )
        return -1e4, 1e4

    model.generator_q = pyo.Var(
        model.GEN,
        within=pyo.Reals,
        bounds=generator_q_bounds,
        initialize=lambda _model, generator: data.generator_q_0[generator],
    )

    generators_at_bus = {bus: [] for bus in buses}
    loads_at_bus = {bus: [] for bus in buses}
    for generator, bus in data.generator_bus.items():
        if bus in generators_at_bus:
            generators_at_bus[bus].append(generator)
    for load, bus in data.load_bus.items():
        if bus in loads_at_bus:
            loads_at_bus[bus].append(load)

    def p_network(model_, bus):
        voltage_bus = voltage(model_, bus)
        angle_bus = angle(model_, bus)
        return data.base_mva * sum(
            voltage_bus
            * voltage(model_, neighbor)
            * (
                data.conductance[bus][neighbor]
                * pyo.cos(angle_bus - angle(model_, neighbor))
                + data.susceptance[bus][neighbor]
                * pyo.sin(angle_bus - angle(model_, neighbor))
            )
            for neighbor in data.neighbors[bus]
        )

    def q_network(model_, bus):
        voltage_bus = voltage(model_, bus)
        angle_bus = angle(model_, bus)
        return data.base_mva * sum(
            voltage_bus
            * voltage(model_, neighbor)
            * (
                data.conductance[bus][neighbor]
                * pyo.sin(angle_bus - angle(model_, neighbor))
                - data.susceptance[bus][neighbor]
                * pyo.cos(angle_bus - angle(model_, neighbor))
            )
            for neighbor in data.neighbors[bus]
        )

    def p_balance_rule(model_, bus):
        injection = (
            sum(model_.generator_p[generator] for generator in generators_at_bus[bus])
            - sum(data.load_p[load] for load in loads_at_bus[bus])
        )
        return injection == p_network(model_, bus)

    def q_balance_rule(model_, bus):
        injection = (
            sum(model_.generator_q[generator] for generator in generators_at_bus[bus])
            - sum(data.load_q[load] for load in loads_at_bus[bus])
        )
        return injection == q_network(model_, bus)

    model.p_balance = pyo.Constraint(model.BUS, rule=p_balance_rule)
    model.q_balance = pyo.Constraint(model.BUS, rule=q_balance_rule)

    def voltage_real(model_, bus):
        return voltage(model_, bus) * pyo.cos(angle(model_, bus))

    def voltage_imaginary(model_, bus):
        return voltage(model_, bus) * pyo.sin(angle(model_, bus))

    def current_components(model_, line, from_side):
        branch = data.branches[line]
        e_from = voltage_real(model_, branch.from_bus)
        f_from = voltage_imaginary(model_, branch.from_bus)
        e_to = voltage_real(model_, branch.to_bus)
        f_to = voltage_imaginary(model_, branch.to_bus)
        if from_side:
            y_self, y_other = branch.yff, branch.yft
        else:
            y_self, y_other = branch.ytt, branch.ytf
            e_from, e_to = e_to, e_from
            f_from, f_to = f_to, f_from
        current_real = (
            y_self.real * e_from
            - y_self.imag * f_from
            + y_other.real * e_to
            - y_other.imag * f_to
        )
        current_imaginary = (
            y_self.real * f_from
            + y_self.imag * e_from
            + y_other.real * f_to
            + y_other.imag * e_to
        )
        return current_real, current_imaginary

    def thermal_from_rule(model_, line):
        current_real, current_imaginary = current_components(model_, line, True)
        limit = data.branch_i_max_from_pu[line]
        return current_real**2 + current_imaginary**2 <= limit**2

    def thermal_to_rule(model_, line):
        current_real, current_imaginary = current_components(model_, line, False)
        limit = data.branch_i_max_to_pu[line]
        return current_real**2 + current_imaginary**2 <= limit**2

    model.thermal_from = pyo.Constraint(model.LINE, rule=thermal_from_rule)
    model.thermal_to = pyo.Constraint(model.LINE, rule=thermal_to_rule)

    model.objective = pyo.Objective(
        expr=sum(
            data.generator_cost[generator] * model.generator_p[generator]
            for generator in model.GEN
        ),
        sense=pyo.minimize,
    )

    model._acopf_data = data
    return model


def initialization(model, data=None):
    if data is None:
        data = model._acopf_data

    for bus in model.NON_SLACK:
        model.theta[bus].value = data.angle_0[int(bus)]
    for bus in model.PQ_BUS:
        model.voltage[bus].value = data.voltage_0[int(bus)]
    for generator in model.GEN:
        model.generator_q[generator].value = data.generator_q_0[int(generator)]
    redispatchable = [int(generator) for generator in model.REDISPATCHABLE_GEN]
    redispatchable_set = set(redispatchable)
    fixed_generators = [
        int(generator)
        for generator in model.GEN
        if int(generator) not in redispatchable_set
    ]

    fixed_generation = 0.0
    for generator in fixed_generators:
        value = float(data.generator_p_0[generator])
        model.generator_p[generator].value = value
        fixed_generation += value

    target_generation = float(
        sum(data.generator_p_0[int(generator)] for generator in model.GEN)
    )
    controllable_target = target_generation - fixed_generation

    lower = {}
    upper = {}
    for generator in redispatchable:
        lower[generator] = max(
            data.generator_p_min[generator],
            data.generator_p_0[generator]
            - data.generator_ramp_down[generator],
        )
        upper[generator] = min(
            data.generator_p_max[generator],
            data.generator_p_0[generator]
            + data.generator_ramp_up[generator],
        )
        model.generator_p[generator].value = lower[generator]

    remaining = controllable_target - sum(lower.values())
    for generator in sorted(
        redispatchable,
        key=lambda generator: (data.generator_cost[generator], generator),
    ):
        if remaining <= 0.0:
            break
        addition = min(remaining, upper[generator] - lower[generator])
        model.generator_p[generator].value = lower[generator] + addition
        remaining -= addition

    initialized_generation = float(
        sum(pyo.value(model.generator_p[generator]) for generator in model.GEN)
    )
    diagnostics = {
        "target_generation_mw": target_generation,
        "initialized_generation_mw": initialized_generation,
        "greedy_mismatch_mw": target_generation - initialized_generation,
    }
    model._initialization_diagnostics = diagnostics
    return model


def solve_acopf(model, tee=True, solver_name="ipopt", options=None):
    solver = pyo.SolverFactory(solver_name)
    if not solver.available(exception_flag=False):
        raise RuntimeError(f"The requested solver is not available: {solver_name}")

    solver_options = dict(DEFAULT_IPOPT_OPTIONS)
    if options:
        solver_options.update(options)
    for key, value in solver_options.items():
        solver.options[key] = value
    try:
        results = solver.solve(model, tee=tee, load_solutions=False)
    except Exception:
        results = pyo.SolverResults()
        results.solver.status = pyo.SolverStatus.error
        results.solver.termination_condition = pyo.TerminationCondition.error
        return results
    if results.solver.termination_condition in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
        pyo.TerminationCondition.feasible,
    ):
        try:
            model.solutions.load_from(results)
        except Exception:
            results.solver.termination_condition = pyo.TerminationCondition.error
    return results


def decode_action(env, obs, model, tolerance_mw=0.05):
    data = model._acopf_data
    action_dictionary = {}

    redispatch = []
    for generator in model.REDISPATCHABLE_GEN:
        generator = int(generator)
        desired_generation = float(pyo.value(model.generator_p[generator]))
        desired_target = (
            desired_generation - data.generator_p_reference[generator]
        )
        command = desired_target - data.generator_target_dispatch[generator]
        command = float(
            np.clip(command,-0.999 * data.generator_ramp_down[generator],
                0.999 * data.generator_ramp_up[generator],))
        if abs(command) > tolerance_mw:
            redispatch.append((generator, command))
    if redispatch:
        action_dictionary["redispatch"] = redispatch

    return env.action_space(action_dictionary), action_dictionary
