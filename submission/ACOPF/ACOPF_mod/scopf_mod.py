
import dataclasses

import numpy as np
import pyomo.environ as pyo

from ACOPF.ACOPF_mod.acopf_setup_twostage_mod import (
    ACCEPTABLE_TERMINATIONS,
    acopf_setup_twostage_mod,
    decode_action_twostage_mod,
    initialization_twostage_mod,
    solve_acopf_twostage_mod,
    twostage_diagnostics_mod,
)


def post_contingency_data(data, outage_line, ramp_horizon=1.0):
    line = int(outage_line)
    branch = data.branches.get(line)
    if branch is None:
        return None

    conductance = {bus: dict(row) for bus, row in data.conductance.items()}
    susceptance = {bus: dict(row) for bus, row in data.susceptance.items()}
    from_bus, to_bus = int(branch.from_bus), int(branch.to_bus)
    blocks = (
        (from_bus, from_bus, branch.yff),
        (from_bus, to_bus, branch.yft),
        (to_bus, from_bus, branch.ytf),
        (to_bus, to_bus, branch.ytt),
    )
    for row_bus, column_bus, admittance in blocks:
        if row_bus not in conductance or column_bus not in conductance[row_bus]:
            continue
        conductance[row_bus][column_bus] -= float(admittance.real)
        susceptance[row_bus][column_bus] -= float(admittance.imag)

    neighbors = {}
    for bus in data.buses:
        kept = []
        for other in data.neighbors[bus]:
            coupling = abs(conductance[bus].get(other, 0.0)) + abs(
                susceptance[bus].get(other, 0.0)
            )
            if other == bus or coupling > 1e-10:
                kept.append(other)
            else:
                conductance[bus].pop(other, None)
                susceptance[bus].pop(other, None)
        if not [other for other in kept if other != bus]:
            return None
        neighbors[bus] = kept

    branches = {
        identifier: value
        for identifier, value in data.branches.items()
        if identifier != line
    }

    horizon = max(1.0, float(ramp_horizon))
    ramp_up = {
        generator: float(value) * horizon
        for generator, value in data.generator_ramp_up.items()
    }
    ramp_down = {
        generator: float(value) * horizon
        for generator, value in data.generator_ramp_down.items()
    }
    return dataclasses.replace(
        data,
        branches=branches,
        neighbors=neighbors,
        conductance=conductance,
        susceptance=susceptance,
        generator_ramp_up=ramp_up,
        generator_ramp_down=ramp_down,
    )


def solve_post_contingency_dispatch(
    env,
    obs,
    data,
    outage_line,
    *,
    rho_target=0.95,
    priority_lines=(),
    reserve_margin_mw=0.0,
    ramp_horizon=1.0,
    solver_tee=False,
):
    contingency_data = post_contingency_data(
        data, outage_line, ramp_horizon=ramp_horizon
    )
    if contingency_data is None:
        return None

    model = acopf_setup_twostage_mod(
        contingency_data,
        rho_target=rho_target,
        priority_lines=priority_lines,
        forecast_up_requirement=0.0,
        forecast_down_requirement=0.0,
        reserve_margin_mw=reserve_margin_mw,
    )
    model = initialization_twostage_mod(model, contingency_data)
    try:
        results = solve_acopf_twostage_mod(model, tee=solver_tee)
    except Exception:
        return None
    if results.solver.termination_condition not in ACCEPTABLE_TERMINATIONS:
        return None

    diagnostics = twostage_diagnostics_mod(model)
    action, dictionary = decode_action_twostage_mod(env, obs, model)
    return {
        "action": action,
        "dictionary": dictionary,
        "outage_line": int(outage_line),
        "predicted_rho": diagnostics.rho_target + diagnostics.worst_slack,
        "predicted_priority_rho": (
            None
            if diagnostics.priority_slack is None
            else diagnostics.rho_target + diagnostics.priority_slack
        ),
        "diagnostics": diagnostics,
    }
