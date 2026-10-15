import dataclasses
import logging
import math
import os
import sys
import time

import grid2op
import numpy as np
import pyomo.environ as pyo

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from ACOPF.ACOPF_mod.acopf_setup_twostage_mod import (
    acopf_setup_twostage_mod,
    decode_action_twostage_mod,
    initialization_twostage_mod,
    solve_acopf_twostage_mod,
    twostage_diagnostics_mod,
    acopf_solve_cached,
)
from ACOPF.ACOPF_mod.data_mod import extract_grid2op_data_mod
from ACOPF.ACOPF_mod.recovery_mod import (
    select_bridge_storage_relief,
    select_continuous_recovery,
    select_line_restoration,
    select_emergency_topology_restoration,
    select_topology_restoration,
)
from ACOPF.ACOPF_mod.security_mod import (
    contingency_not_worse,
    dynamic_security_set,
    maximum_rho,
    outage_induced_bridges,
    screen_dynamic_contingencies,
    topology_bridges,
    validation_horizons,
    validate_action,
)
from ACOPF.ACOPF_mod.scopf_mod import post_contingency_data
from ACOPF.ACOPF_mod.topology_mod import (
    protected_bridge_relief_action,
    protected_contingency_bridge_action,
    protected_lodf_action,
)
from ACOPF.ACOPF_mod.interventions import (
    build_partial_restore_action,
    build_restore_action,
    cooldown_cleared,
    record_intervention,
    trim_record,
)
from ACOPF.ACOPF_mod.emergency_mod import (
    emergency_topology_action,
    evaluate as evaluate_transition,
    preventive_topology_action,
)
from ACOPF.ACOPF_mod.security_mod import imminent_lines, relevant_contingencies, valid_transition
from ACOPF.ACOPF_mod.recovery_mod import maintenance_blocks_reconnection as _maintenance_blocks_reconnection
from ACOPF.ACOPF_mod import pace_mod
from ACOPF.ACOPF_mod.ipopt_bootstrap import bootstrap as _bootstrap_ipopt

IPOPT_BINARY, IPOPT_OK = _bootstrap_ipopt()

from ACOPF.ACOPF_mod import fast_simulate_mod as _fast_simulate_mod

FAST_SIMULATE = _fast_simulate_mod.install()


logging.getLogger("pyomo.core").setLevel(logging.ERROR)


SOLVER_TEE = False

ACTIVATION_RHO = 0.95
RHO_TARGET = 0.95
OUTAGE_POSTURE = os.environ.get("ACOPF_OUTAGE_POSTURE", "1") == "1"
MAINT_STEP_TERMS = os.environ.get("ACOPF_MAINT_STEP_TERMS", "1") == "1"
MAINT_LODF_SCOPF = os.environ.get("ACOPF_MAINT_LODF_SCOPF", "1") == "1"
MAINT_SCOPF_SEVERE = float(os.environ.get("ACOPF_MAINT_SCOPF_SEVERE", "1.15"))
MAINT_SCOPF_CURTAIL_WEIGHT = float(os.environ.get("ACOPF_MAINT_SCOPF_CURTAIL_WEIGHT", "200.0"))
OUTAGE_ACTIVATION_RHO = float(os.environ.get("ACOPF_OUTAGE_ACTIVATION_RHO", "0.92"))
OUTAGE_RHO_TARGET = float(os.environ.get("ACOPF_OUTAGE_RHO_TARGET", "0.90"))
_OUTAGE_POSTURE = {"active": False}


def _planned_outage_in_progress(obs):
    try:
        tnm = np.asarray(obs.time_next_maintenance, dtype=int)
        dnm = np.asarray(obs.duration_next_maintenance, dtype=int)
        status = np.asarray(obs.line_status, dtype=bool)
        return bool(np.any((tnm == 0) & (dnm > 0) & ~status))
    except Exception:
        return False
RESTORE_RHO_LIMIT = 0.90
RECOVERY_RHO_LIMIT = 0.95
N1_RHO_LIMIT = 1.20
PREVENTIVE_SCREEN_INTERVAL = int(os.environ.get("ACOPF_SCREEN_INTERVAL", "6"))
RESTORATION_CHECK_INTERVAL = 6
EMERGENCY_MAX_MANAGED = int(os.environ.get("ACOPF_EMERGENCY_MAX_MANAGED", "4"))
PREVENTIVE_MAX_MANAGED = int(os.environ.get("ACOPF_PREVENTIVE_MAX_MANAGED", "2"))
PREVENTIVE_OVERLOAD_RHO = float(os.environ.get("ACOPF_PREVENTIVE_OVERLOAD_RHO", "1.05"))
ENABLE_EMERGENCY_SEARCH = os.environ.get("ACOPF_EMERGENCY_SEARCH", "1") == "1"
SLOW_PHASE_REPORT_S = float(os.environ.get("ACOPF_SLOW_PHASE_REPORT_S", "10"))
TEMPLATE_MAX_THREATS = int(os.environ.get("ACOPF_TEMPLATE_MAX_THREATS", "3"))
TEMPLATE_MAX_RECORDS = int(os.environ.get("ACOPF_TEMPLATE_MAX_RECORDS", "2"))
TEMPLATE_STAGE_BUDGET_S = float(os.environ.get("ACOPF_TEMPLATE_BUDGET_S", "30"))
EMERGENCY_MIN_RHO = float(os.environ.get("ACOPF_EMERGENCY_MIN_RHO", "1.05"))
EMERGENCY_SEARCH_BUDGET_S = float(os.environ.get("ACOPF_EMERGENCY_BUDGET_S", "30"))
ENABLE_PREVENTIVE_SEARCH = os.environ.get("ACOPF_PREVENTIVE_SEARCH", "0") == "1"
PREVENTIVE_MAINTENANCE_ONLY = os.environ.get("ACOPF_PREVENTIVE_MAINTENANCE", "1") == "1"
PREVENTIVE_SEARCH_BUDGET_S = float(os.environ.get("ACOPF_PREVENTIVE_BUDGET_S", "45"))
RESTORE_MIN_OUTAGES = int(os.environ.get("ACOPF_RESTORE_MIN_OUTAGES", "3"))
EMERGENCY_RESTORE_RHO_LIMIT = float(os.environ.get("ACOPF_EMERGENCY_RESTORE_RHO", "0.95"))
RESERVE_MARGIN_MW = float(os.environ.get("ACOPF_RESERVE_MARGIN_MW", "5.0"))
BEST_EFFORT_RHO_TOLERANCE = 0.01
CRITICAL_BRIDGE_RHO_TARGET = 0.85
CRITICAL_BRIDGE_ACTIVATION_RHO = 0.80
FORECAST_BRIDGE_ACTIVATION_RHO = 0.90
ENABLE_ENDPOINT_BRIDGE_TOPOLOGY = False


ACCEPTABLE = {
    pyo.TerminationCondition.optimal,
    pyo.TerminationCondition.locallyOptimal,
    pyo.TerminationCondition.feasible,
}


def _critical_line(obs):
    connected = np.asarray(obs.line_status, dtype=bool)
    rho = np.asarray(obs.rho, dtype=float)
    candidates = np.flatnonzero(connected & np.isfinite(rho))
    return int(candidates[np.argmax(rho[candidates])]) if candidates.size else None


def _managed_record(env, obs, action, label, step, replaced_record=None):
    record = record_intervention(env, obs, action, label, step)
    record["outages_at_creation"] = [
        int(line) for line in np.flatnonzero(~np.asarray(obs.line_status, dtype=bool))
    ]
    if replaced_record is None:
        return record
    old_topology = dict(
        zip(replaced_record["topo_positions"], replaced_record["prior_topo"])
    )
    record["prior_topo"] = [
        int(old_topology.get(position, prior))
        for position, prior in zip(record["topo_positions"], record["prior_topo"])
    ]
    old_lines = dict(
        zip(replaced_record["lines"], replaced_record["prior_connected"])
    )
    record["prior_connected"] = [
        bool(old_lines.get(line, prior))
        for line, prior in zip(record["lines"], record["prior_connected"])
    ]
    record["subs"] = sorted(
        set(record["subs"]) | set(replaced_record.get("subs", ()))
    )
    return record


DEGRADED_RECOVERY_MIN_RHO = float(os.environ.get("ACOPF_DEGRADED_RECOVERY_MIN_RHO", "0.95"))
RELEASE_UNDER_OUTAGE = os.environ.get("ACOPF_RELEASE_UNDER_OUTAGE", "1") == "1"
MINIMAL_PARITY = os.environ.get("ACOPF_MINIMAL_PARITY", "1") == "1"
MINIMAL_BRIDGE_ACTIVATION = os.environ.get("ACOPF_MINIMAL_BRIDGE", "1") == "1"
DEGRADED_DOOMED_ONLY = os.environ.get("ACOPF_DEGRADED_DOOMED_ONLY", "1") == "1"


def _degraded_recovery(env, obs, managed_records, opponent_blocked, horizons, rho_cap=0.98):
    status = np.asarray(obs.line_status, dtype=bool)
    if np.all(status):
        return None
    from ACOPF.ACOPF_mod import emergency_mod as _emergency

    pending = _emergency._PENDING_PLAN.get("plan")
    if pending is not None and pending["step"] == int(obs.current_step) - 1:
        return None
    own_lines = {int(l) for record in managed_records for l in record.get("lines", ())}
    blocked = set(map(int, opponent_blocked)) | own_lines
    cooldown = np.asarray(obs.time_before_cooldown_line, dtype=int)
    blocked |= {int(l) for l in np.flatnonzero(cooldown > 0)}
    try:
        ids = np.asarray(obs.alertable_line_ids, dtype=int); since = np.asarray(obs.time_since_last_attack, dtype=int)
        blocked |= {int(ids[p]) for p in range(min(ids.size, since.size)) if since[p] >= 0 and not status[int(ids[p])]}
    except Exception:
        pass
    exogenous = [
        int(line) for line in np.flatnonzero(~status)
        if int(line) not in blocked
        and int(obs.time_before_cooldown_line[line]) == 0
        and not _maintenance_blocks_reconnection(obs, line)
    ]
    if not exogenous:
        return None
    hold = evaluate_transition(env, obs, env.action_space({}), horizons)
    hold_key = (hold["n_over"], hold["max_rho"]) if hold is not None else (10 ** 6, float("inf"))
    if DEGRADED_DOOMED_ONLY and hold is not None and not imminent_lines(env, obs, remaining_steps=1):
        return None
    best = None
    for line in exogenous:
        reconnect = env.action_space({
            "set_line_status": [(line, 1)],
            "set_bus": {"lines_or_id": [(line, 1)], "lines_ex_id": [(line, 1)]},
        })
        options = [(reconnect, None)]
        for index, record in enumerate(managed_records):
            if not record.get("topo_positions") or not cooldown_cleared(obs, record):
                continue
            others = [
                int(l) for l in record.get("outages_at_creation", ())
                if int(l) != line and int(l) not in own_lines and not status[int(l)]
            ]
            if others:
                continue
            merge = build_partial_restore_action(env, record, include_lines=False, obs=obs)
            try:
                options.append((reconnect + merge, index))
            except Exception:
                continue
        for action, index in options:
            evaluation = evaluate_transition(env, obs, action, horizons, opened=())
            if evaluation is None or evaluation["max_rho"] > max(rho_cap, hold_key[1]):
                continue
            key = (evaluation["n_over"], round(evaluation["max_rho"], 4))
            if not (
                key[0] < hold_key[0] and key[1] <= hold_key[1] + 1e-6
                or key[0] == hold_key[0] and key[1] < hold_key[1] - 0.005
            ):
                continue
            if best is None or key < best[0]:
                best = (key, action, line, index)
    if best is None:
        return None
    key, action, line, index = best
    baseline = screen_dynamic_contingencies(env, obs, env.action_space({}), rho_limit=N1_RHO_LIMIT, time_step=max(horizons))
    candidate = screen_dynamic_contingencies(env, obs, action, rho_limit=N1_RHO_LIMIT, time_step=max(horizons))
    if not contingency_not_worse(candidate, baseline, rho_tolerance=0.05):
        print(f"[recovery] step {int(obs.current_step)} reconnect {line} vetoed by N-1: {_not_worse_reasons(candidate, baseline, 0.05)}", flush=True)
        return None
    label = f"reconnect line {line}" + ("" if index is None else f" + merge record {index}")
    print(f"[recovery] step {int(obs.current_step)} {label} -> {key} (hold {hold_key})", flush=True)
    return {
        "action": action,
        "dictionary": {},
        "decision": f"recover_degraded:{label}",
        "acopf_status": "degraded_recovery",
        "record": None,
        "replace_index": None,
        "trim_record_index": index,
    }


BALANCING_HEADROOM_FRACTION = float(os.environ.get("ACOPF_BALANCING_HEADROOM", "0.95"))
EMERGENCY_RESERVE_FLOOR = float(os.environ.get("ACOPF_EMERGENCY_RESERVE_FLOOR", "1.0"))
BALANCING_FORECAST_MARGIN_FRACTION = float(os.environ.get("ACOPF_BALANCING_MARGIN_FRACTION", "0.15"))
BALANCING_FORECAST_MARGIN_MW = float(os.environ.get("ACOPF_BALANCING_MARGIN_MW", "5.0"))
BALANCING_ERROR_QUANTILE = float(os.environ.get("ACOPF_BALANCING_ERROR_QUANTILE", "0.7"))
_RENEWABLE_ERRORS = []
_RENEWABLE_SIGNED_ERRORS = []
_LAST_RENEWABLE_FORECAST = {"step": None, "value": None, "corrected": None}
FORECAST_BIAS_STEPS = int(os.environ.get("ACOPF_FORECAST_BIAS_STEPS", "3"))


def _track_renewable_forecast(env, obs):
    cls = type(env)
    renewable = np.asarray(cls.gen_renewable, dtype=bool)
    actual = float(np.asarray(obs.gen_p, dtype=float)[renewable].sum())
    step = int(obs.current_step)
    if step <= 1:
        _RENEWABLE_ERRORS.clear()
        _RENEWABLE_SIGNED_ERRORS.clear()
    last = _LAST_RENEWABLE_FORECAST
    if last["step"] is not None and last["step"] == step - 1 and last["value"] is not None:
        reference = last["corrected"] if last["corrected"] is not None else last["value"]
        _RENEWABLE_ERRORS.append(abs(actual - reference))
        del _RENEWABLE_ERRORS[:-96]
        _RENEWABLE_SIGNED_ERRORS.append(actual - last["value"])
        del _RENEWABLE_SIGNED_ERRORS[:-96]
    forecast, _load = _expected_renewables_next(env, obs, raw=True)
    last["step"], last["value"] = step, forecast
    corrected, _load2 = _expected_renewables_next(env, obs)
    last["corrected"] = corrected


def _renewable_forecast_bias():
    n = FORECAST_BIAS_STEPS
    if n <= 0 or len(_RENEWABLE_SIGNED_ERRORS) < n:
        return 0.0
    recent = _RENEWABLE_SIGNED_ERRORS[-n:]
    if not (all(e > 0.0 for e in recent) or all(e < 0.0 for e in recent)):
        return 0.0
    return float(np.mean(recent))


LOAD_PROFILE = (
    (0.949, 0.885, 0.862, 0.818, 0.801, 0.829, 0.900, 0.986, 1.041, 1.076, 1.094, 1.107, 1.118, 1.110, 1.086, 1.056, 1.035, 1.030, 1.060, 1.096, 1.062, 1.019, 1.008, 1.044),
    (0.998, 0.928, 0.905, 0.861, 0.843, 0.869, 0.944, 1.039, 1.095, 1.124, 1.135, 1.145, 1.153, 1.141, 1.120, 1.091, 1.069, 1.062, 1.089, 1.120, 1.082, 1.037, 1.025, 1.060),
    (1.012, 0.940, 0.917, 0.874, 0.857, 0.883, 0.955, 1.048, 1.106, 1.137, 1.149, 1.158, 1.166, 1.154, 1.134, 1.104, 1.082, 1.075, 1.103, 1.135, 1.096, 1.050, 1.037, 1.072),
    (1.026, 0.954, 0.929, 0.884, 0.864, 0.888, 0.961, 1.056, 1.108, 1.134, 1.142, 1.149, 1.152, 1.139, 1.116, 1.086, 1.063, 1.056, 1.084, 1.115, 1.076, 1.033, 1.023, 1.060),
    (1.014, 0.943, 0.920, 0.877, 0.859, 0.884, 0.957, 1.050, 1.105, 1.133, 1.141, 1.145, 1.147, 1.133, 1.106, 1.073, 1.044, 1.037, 1.061, 1.081, 1.049, 1.013, 1.009, 1.048),
    (0.998, 0.922, 0.891, 0.841, 0.813, 0.815, 0.836, 0.867, 0.912, 0.965, 0.993, 1.004, 1.027, 1.020, 0.979, 0.946, 0.922, 0.920, 0.952, 0.983, 0.964, 0.940, 0.948, 0.999),
    (0.963, 0.892, 0.862, 0.810, 0.779, 0.772, 0.779, 0.793, 0.825, 0.876, 0.917, 0.944, 0.973, 0.963, 0.914, 0.881, 0.860, 0.865, 0.908, 0.957, 0.956, 0.942, 0.946, 0.991),
)
PEAK_PROJECTION = os.environ.get("ACOPF_PEAK_PROJECTION", "1") == "1"
POSTURE_HORIZON_H = float(os.environ.get("ACOPF_POSTURE_HORIZON_H", "8"))
NEED_HORIZON_H = float(os.environ.get("ACOPF_NEED_HORIZON_H", "2"))
NEED_MARGIN_DEFAULT = float(os.environ.get("ACOPF_DEMAND_MARGIN", "0.05"))
FORECAST_MARGIN = float(os.environ.get("ACOPF_FORECAST_MARGIN", "0.02"))
_LOAD_HISTORY = {"step": -1, "loads": [], "slots": [], "errors": {}}


def _track_load(obs):
    step = int(obs.current_step)
    if step <= _LOAD_HISTORY["step"]:
        _LOAD_HISTORY["loads"].clear(); _LOAD_HISTORY["slots"].clear(); _LOAD_HISTORY["errors"].clear()
    _LOAD_HISTORY["step"] = step
    now = float(np.sum(obs.load_p))
    loads, slots = _LOAD_HISTORY["loads"], _LOAD_HISTORY["slots"]
    for k in (24, 96):
        if len(loads) >= k:
            past = loads[-k]; d0, h0 = slots[-k]
            predicted = past * _shape_ratio(d0, h0, k / 12.0, point=True)
            if predicted > 0 and now > 0:
                errs = _LOAD_HISTORY["errors"].setdefault(k, [])
                errs.append(abs(predicted / now - 1.0))
                del errs[:-300]
    loads.append(now); slots.append((int(obs.day_of_week) % 7, int(obs.hour_of_day) % 24))
    del loads[:-2016]; del slots[:-2016]


def _shape_ratio(day, hour, hours, point=False):
    now = LOAD_PROFILE[day][hour]
    steps = max(1, int(round(hours)))
    if point:
        h = hour + steps
        return LOAD_PROFILE[(day + h // 24) % 7][h % 24] / max(now, 1e-6)
    worst = now
    for k in range(1, steps + 1):
        h = hour + k
        worst = max(worst, LOAD_PROFILE[(day + h // 24) % 7][h % 24])
    return worst / max(now, 1e-6)


def _hour_bias(day, hour):
    loads, slots = _LOAD_HISTORY["loads"], _LOAD_HISTORY["slots"]
    if len(loads) < 24:
        return 1.0
    level = float(np.mean(loads)) / max(float(np.mean([LOAD_PROFILE[d][h] for d, h in slots])), 1e-6)
    same = [l / (LOAD_PROFILE[d][h] * level) for l, (d, h) in zip(loads, slots) if h == hour]
    if not same:
        return 1.0
    return float(np.clip(np.mean(same), 0.8, 1.25))


def _forecast_load_ratios(obs):
    try:
        load_p, *_rest = obs.get_forecast_arrays()
        now = float(np.sum(obs.load_p))
        if now <= 0:
            return []
        return [float(np.sum(row)) / now for row in np.asarray(load_p, dtype=float)[1:]]
    except Exception:
        return []


def _demand_margin(hours):
    k = 24 if hours <= 3 else 96
    errs = _LOAD_HISTORY["errors"].get(k, [])
    if len(errs) < 24:
        return NEED_MARGIN_DEFAULT
    return float(np.clip(np.quantile(errs, 0.9), 0.02, 0.25))


def _demand_ratio(obs, hours):
    if not PEAK_PROJECTION or hours <= 0:
        return 1.0
    day = int(obs.day_of_week) % 7
    hour = int(obs.hour_of_day) % 24
    worst = 1.0
    fc = _forecast_load_ratios(obs)
    for r in fc[: int(min(12, hours * 12))]:
        worst = max(worst, r * (1.0 + FORECAST_MARGIN))
    if hours > 1.0 or not fc:
        margin = _demand_margin(hours)
        base_now = LOAD_PROFILE[day][hour] * _hour_bias(day, hour)
        for k in range(1, int(math.ceil(hours)) + 1):
            h = hour + k
            d2, h2 = (day + h // 24) % 7, h % 24
            projected = LOAD_PROFILE[d2][h2] * _hour_bias(d2, h2) / max(base_now, 1e-6)
            worst = max(worst, projected * (1.0 + margin))
    return float(max(1.0, worst))


def _peak_factor(obs):
    return _demand_ratio(obs, POSTURE_HORIZON_H)


def _renewable_error_quantile(q=None):
    if len(_RENEWABLE_ERRORS) < 6:
        return 0.0
    return float(np.quantile(_RENEWABLE_ERRORS, BALANCING_ERROR_QUANTILE if q is None else q))


EMERGENCY_RESERVE_QUANTILE = float(os.environ.get("ACOPF_EMERGENCY_RESERVE_QUANTILE", "0.9"))


def _expected_renewables_next(env, obs, ratios=None, raw=False):
    cls = type(env)
    renewable = np.asarray(cls.gen_renewable, dtype=bool)
    gen_p = np.asarray(obs.gen_p, dtype=float)
    try:
        gen_f, _v, load_f, _q = obs.get_forecasted_inj(1)
        gen_f = np.asarray(gen_f, dtype=float)
        load_f = np.asarray(load_f, dtype=float)
        if not (np.all(np.isfinite(gen_f)) and np.all(np.isfinite(load_f))):
            raise ValueError
    except Exception:
        return gen_p[renewable].sum(), float(np.sum(obs.load_p))
    limits = np.asarray(obs.curtailment_limit, dtype=float).copy()
    if ratios is not None:
        ratios = np.asarray(ratios, dtype=float)
        limits = np.where(ratios >= 0.0, ratios, limits)
    p_max = np.asarray(cls.gen_pmax, dtype=float)
    expected = np.minimum(gen_f, limits * p_max)
    total = float(expected[renewable].sum())
    if not raw:
        bias = _renewable_forecast_bias()
        if bias != 0.0:
            unlimited = float(gen_f[renewable].sum())
            scale = total / unlimited if unlimited > 1.0 else 1.0
            total = max(0.0, total + bias * scale)
    return total, float(np.sum(load_f))


BALANCING_FORM = os.environ.get("ACOPF_BALANCING_FORM", "fleet")
FLEET_MOVE_BIAS_STEPS = int(os.environ.get("ACOPF_FLEET_MOVE_BIAS_STEPS", "3"))
FLEET_MOVE_SANITY_MW = float(os.environ.get("ACOPF_FLEET_MOVE_SANITY_MW", "250"))
FLEET_MOVE_SPIKE_MW = float(os.environ.get("ACOPF_FLEET_MOVE_SPIKE_MW", "30"))
_FLEET_MOVE = {"step": None, "chronic_sum": None, "incr_fc": None, "est": None, "errors": [], "residuals": [],
               "last_realised": None}
PREVIEW_STEPS = int(os.environ.get("ACOPF_PREVIEW_STEPS", "3"))
PREVIEW_LONG = int(os.environ.get("ACOPF_PREVIEW_LONG", "6"))
PREVIEW_SOC_LOW = float(os.environ.get("ACOPF_PREVIEW_SOC_LOW", "0.4"))
PREVIEW_SOC_HIGH = float(os.environ.get("ACOPF_PREVIEW_SOC_HIGH", "0.6"))
PREVIEW_SOC_POWER = float(os.environ.get("ACOPF_PREVIEW_SOC_POWER", "6.0"))
PREVIEW_RAMP_FRACTION = float(os.environ.get("ACOPF_PREVIEW_RAMP_FRACTION", "1.0"))
BALANCE_COMPLETE_ALL = os.environ.get("ACOPF_BALANCE_COMPLETE_ALL", "1") == "1"
SOC_BAND_LOW = float(os.environ.get("ACOPF_SOC_BAND_LOW", "0.15"))
SOC_BAND_HIGH = float(os.environ.get("ACOPF_SOC_BAND_HIGH", "0.85"))
SOC_BAND_POWER = float(os.environ.get("ACOPF_SOC_BAND_POWER", "3.0"))


def _soc_band_action(env, obs):
    cls = type(env)
    n = int(getattr(cls, "n_storage", 0) or 0)
    if n == 0 or SOC_BAND_POWER <= 0.0:
        return None
    e_max = np.asarray(cls.storage_Emax, dtype=float); charge = np.asarray(obs.storage_charge, dtype=float)
    soc = charge / np.maximum(e_max, 1e-6)
    cap_c, cap_d = _storage_sustain_caps(env, obs)
    order = np.where(soc > SOC_BAND_HIGH, -np.minimum(SOC_BAND_POWER, cap_d), 0.0)
    order = np.where(soc < SOC_BAND_LOW, np.minimum(SOC_BAND_POWER, cap_c), order)
    if float(np.abs(order).sum()) < 1.0:
        return None
    try:
        action = env.action_space({"set_storage": [(int(i), float(v)) for i, v in enumerate(order) if abs(v) > 0.05]})
    except Exception:
        return None
    if not _balancing_feasible(env, obs, action)[0] or not _balancing_comfortable(env, obs, action):
        return None
    return action, float(order.sum())


def _complete_balance(env, obs, action, label):
    if not BALANCE_COMPLETE_ALL:
        return action
    ok, why = _balancing_feasible(env, obs, action)
    if ok:
        return action
    up, avail_up, down, avail_down = _balancing_requirement(env, obs, action)
    cls = type(env)
    now = np.asarray(getattr(obs, "storage_power", np.zeros(0)), dtype=float)
    ordered = np.asarray(action.storage_p, dtype=float) if now.size else now
    current = ordered if np.any(np.abs(ordered) > 0.05) else now
    cap_c, cap_d = _storage_sustain_caps(env, obs)
    candidate = action
    if down > BALANCING_HEADROOM_FRACTION * avail_down:
        deficit = down - BALANCING_HEADROOM_FRACTION * avail_down + 2.0
        headroom = np.maximum(cap_c - np.maximum(current, 0.0), 0.0)
        if float(headroom.sum()) >= 1.0:
            extra = headroom / float(headroom.sum()) * min(deficit, float(headroom.sum()))
            order = np.clip(current + extra, -cap_d, cap_c)
            try:
                trial = action + env.action_space({"set_storage": [(int(i), float(v)) for i, v in enumerate(order) if abs(v) > 0.05]})
                if _balancing_feasible(env, obs, trial)[0]:
                    print(f"[balance] step {int(obs.current_step)} {label}: completed with storage {float(order.sum()):+.1f} MW", flush=True)
                    return trial
                candidate = trial
            except Exception:
                pass
        curt = _curtail_for_balance_action(env, obs, candidate)
        if curt is not None:
            print(f"[balance] step {int(obs.current_step)} {label}: completed with storage and ~{curt[1]:.1f} MW of curtailment", flush=True)
            return curt[0]
    elif up > BALANCING_HEADROOM_FRACTION * avail_up:
        deficit = up - BALANCING_HEADROOM_FRACTION * avail_up + 2.0
        headroom = np.maximum(cap_d - np.maximum(-current, 0.0), 0.0)
        if float(headroom.sum()) >= 1.0:
            extra = headroom / float(headroom.sum()) * min(deficit, float(headroom.sum()))
            order = np.clip(current - extra, -cap_d, cap_c)
            try:
                trial = action + env.action_space({"set_storage": [(int(i), float(v)) for i, v in enumerate(order) if abs(v) > 0.05]})
                if _balancing_feasible(env, obs, trial)[0]:
                    print(f"[balance] step {int(obs.current_step)} {label}: completed with storage {float(order.sum()):+.1f} MW", flush=True)
                    return trial
            except Exception:
                pass
    return action
_PREVIEW = {"step": None, "chronic": {}, "fc": {}, "errors": {}, "est": {}}


def _track_preview(env, obs):
    st = _PREVIEW
    step = int(obs.current_step)
    if step <= 1:
        st["chronic"].clear(); st["fc"].clear(); st["errors"].clear(); st["est"].clear()
    st["step"] = step
    cls = type(env)
    redispatchable = np.asarray(cls.gen_redispatchable, dtype=bool)
    chronic = float(np.sum((np.asarray(obs.gen_p, dtype=float) - np.asarray(obs.actual_dispatch, dtype=float))[redispatchable]))
    st["chronic"][step] = chronic
    horizons = range(1, max(PREVIEW_STEPS, PREVIEW_LONG) + 1)
    for h in horizons:
        past = st["fc"].get(step - h, {}).get(h)
        if past is not None and (step - h) in st["chronic"]:
            realised = chronic - st["chronic"][step - h]
            if abs(realised) <= FLEET_MOVE_SANITY_MW * h:
                st["errors"].setdefault(h, []).append(past - realised); del st["errors"][h][:-48]
    now = {}
    chron_vec = np.asarray(obs.gen_p, dtype=float) - np.asarray(obs.actual_dispatch, dtype=float)
    for h in horizons:
        try:
            gen_f = np.asarray(obs.get_forecasted_inj(h)[0], dtype=float)
            if not np.all(np.isfinite(gen_f[redispatchable])):
                continue
            v = float(np.sum((gen_f - chron_vec)[redispatchable]))
            if abs(v) <= FLEET_MOVE_SANITY_MW * h:
                now[h] = v
        except Exception:
            continue
    st["fc"][step] = now
    for k in [k for k in st["fc"] if k < step - max(horizons) - 2]:
        del st["fc"][k]
    for k in [k for k in st["chronic"] if k < step - max(horizons) - 2]:
        del st["chronic"][k]
    est = {}
    for h, v in now.items():
        errs = st["errors"].get(h, [])
        if len(errs) >= 3:
            est[h] = v - float(np.median(errs[-3:]))
    st["est"] = est


def _preview_need():
    est = _PREVIEW.get("est", {})
    up = down = 0.0; prev = 0.0
    for h in range(1, PREVIEW_STEPS + 1):
        if h not in est:
            break
        d = est[h] - prev; prev = est[h]
        up = max(up, d); down = max(down, -d)
    return up, down


def _preview_long():
    return _PREVIEW.get("est", {}).get(PREVIEW_LONG)


def _storage_preposition_action(env, obs):
    long = _preview_long()
    if long is None:
        return None
    cls = type(env)
    n = int(getattr(cls, "n_storage", 0) or 0)
    if n == 0:
        return None
    redispatchable = np.asarray(cls.gen_redispatchable, dtype=bool)
    gen_p = np.asarray(obs.gen_p, dtype=float)
    room = float(np.sum(np.minimum(np.asarray(cls.gen_max_ramp_down, dtype=float), np.maximum(gen_p - np.asarray(cls.gen_pmin, dtype=float), 0.0))[redispatchable]))
    if abs(long) < PREVIEW_RAMP_FRACTION * room:
        return None
    e_max = np.asarray(cls.storage_Emax, dtype=float); charge = np.asarray(obs.storage_charge, dtype=float)
    soc = charge / np.maximum(e_max, 1e-6)
    cap_c, cap_d = _storage_sustain_caps(env, obs)
    if long < 0:
        units = soc > PREVIEW_SOC_LOW
        order = -np.minimum(PREVIEW_SOC_POWER, cap_d) * units
    else:
        units = soc < PREVIEW_SOC_HIGH
        order = np.minimum(PREVIEW_SOC_POWER, cap_c) * units
    if float(np.abs(order).sum()) < 1.0:
        return None
    try:
        action = env.action_space({"set_storage": [(int(i), float(v)) for i, v in enumerate(order) if abs(v) > 0.05]})
    except Exception:
        return None
    if not _balancing_feasible(env, obs, action)[0] or not _balancing_comfortable(env, obs, action):
        return None
    return action, float(order.sum()), float(long)
_BALANCE_DIAG = {}


def _chronic_sum(env, obs):
    cls = type(env)
    redispatchable = np.asarray(cls.gen_redispatchable, dtype=bool)
    return float(np.sum((np.asarray(obs.gen_p, dtype=float) - np.asarray(obs.actual_dispatch, dtype=float))[redispatchable]))


def _fleet_increment_forecast(env, obs):
    cls = type(env)
    redispatchable = np.asarray(cls.gen_redispatchable, dtype=bool)
    try:
        gen_f, _v, _lf, _q = obs.get_forecasted_inj(1)
        gen_f = np.asarray(gen_f, dtype=float)
        if not np.all(np.isfinite(gen_f[redispatchable])):
            return None
        chronic = np.asarray(obs.gen_p, dtype=float) - np.asarray(obs.actual_dispatch, dtype=float)
        return float(np.sum((gen_f - chronic)[redispatchable]))
    except Exception:
        return None


def _track_fleet_move(env, obs):
    step = int(obs.current_step)
    st = _FLEET_MOVE
    if step <= 1:
        st["errors"].clear(); st["residuals"].clear()
        st["step"] = st["chronic_sum"] = st["incr_fc"] = st["est"] = None
    chronic_sum = _chronic_sum(env, obs)
    if st["step"] is not None and st["step"] == step - 1 and st["chronic_sum"] is not None:
        realised = chronic_sum - st["chronic_sum"]
        if abs(realised) <= FLEET_MOVE_SANITY_MW:
            st["last_realised"] = realised
            if st["incr_fc"] is not None:
                st["errors"].append(st["incr_fc"] - realised); del st["errors"][:-96]
            if st["est"] is not None:
                st["residuals"].append(abs(st["est"] - realised)); del st["residuals"][:-96]
    incr_fc = _fleet_increment_forecast(env, obs)
    if incr_fc is not None and abs(incr_fc) > FLEET_MOVE_SANITY_MW:
        incr_fc = None
    est = None
    prev_est = st.get("est")
    st["forecast_rejected"] = False
    if incr_fc is not None:
        k = FLEET_MOVE_BIAS_STEPS
        bias = float(np.median(st["errors"][-k:])) if (k > 0 and len(st["errors"]) >= 2) else 0.0
        corrected = incr_fc - bias
        cands = [corrected]
        if st["last_realised"] is not None:
            cands.append(st["last_realised"])
        if prev_est is not None:
            cands.append(prev_est)
        est = float(np.median(cands)) if len(cands) == 3 else corrected
        st["forecast_rejected"] = len(cands) == 3 and abs(est - corrected) > FLEET_MOVE_SPIKE_MW
    elif st["last_realised"] is not None:
        est = st["last_realised"]
    st.update(step=step, chronic_sum=chronic_sum, incr_fc=incr_fc, est=est)


def _fleet_move_next(env, obs):
    cls = type(env)
    redispatchable = np.asarray(cls.gen_redispatchable, dtype=bool)
    try:
        g1 = np.asarray(obs.get_forecasted_inj(1)[0], dtype=float)
        g2 = np.asarray(obs.get_forecasted_inj(2)[0], dtype=float)
        if np.all(np.isfinite(g1[redispatchable])) and np.all(np.isfinite(g2[redispatchable])):
            v = float(np.sum((g2 - g1)[redispatchable]))
            if abs(v) <= FLEET_MOVE_SANITY_MW:
                return v
    except Exception:
        pass
    est = _expected_fleet_increment(env, obs)
    return float(est) if est is not None else 0.0


def _expected_fleet_increment(env, obs):
    if _FLEET_MOVE["step"] != int(obs.current_step):
        _track_fleet_move(env, obs)
    return _FLEET_MOVE["est"]


def _fleet_move_error_quantile(q):
    res = _FLEET_MOVE["residuals"]
    if len(res) < 6:
        return None
    return float(np.quantile(res, q))


def _balancing_requirement(env, obs, action):
    cls = type(env)
    redispatchable = np.asarray(cls.gen_redispatchable, dtype=bool)
    renewable = np.asarray(cls.gen_renewable, dtype=bool)
    gen_p = np.asarray(obs.gen_p, dtype=float)
    renewables_next, load_next = _expected_renewables_next(env, obs, np.asarray(action.curtail))
    load_change = load_next - float(np.sum(obs.load_p))
    renewable_change = renewables_next - float(gen_p[renewable].sum())
    storage_now = float(np.sum(np.asarray(getattr(obs, "storage_power", np.zeros(0)), dtype=float)))
    storage_charge = float(np.sum(np.asarray(action.storage_p, dtype=float))) - storage_now
    chronic_change = 0.0
    try:
        gen_f, _v, _lf, _q = obs.get_forecasted_inj(1)
        gen_f = np.asarray(gen_f, dtype=float)
        actual = np.asarray(obs.actual_dispatch, dtype=float)
        if np.all(np.isfinite(gen_f[redispatchable])):
            chronic_change = float(np.sum((gen_f - (gen_p - actual))[redispatchable]))
        if abs(chronic_change) > CHRONIC_CHANGE_SANITY_MW:
            chronic_change = 0.0
    except Exception:
        chronic_change = 0.0
    fleet = _expected_fleet_increment(env, obs) if BALANCING_FORM == "fleet" else None
    if fleet is not None:
        try:
            gen_f_r = np.asarray(obs.get_forecasted_inj(1)[0], dtype=float)[renewable]
            avail_next = float(np.sum(gen_f_r)) if np.all(np.isfinite(gen_f_r)) else None
        except Exception:
            avail_next = None
        if avail_next is not None:
            bias = _renewable_forecast_bias()
            avail_change = avail_next + bias - float(np.asarray(obs.gen_p_before_curtail, dtype=float)[renewable].sum())
            curtail_growth = avail_change - renewable_change
        else:
            curtail_growth = 0.0
        quant = _fleet_move_error_quantile(BALANCING_ERROR_QUANTILE)
        margin = max(
            BALANCING_FORECAST_MARGIN_FRACTION * abs(renewable_change) + BALANCING_FORECAST_MARGIN_MW,
            quant if quant is not None else _renewable_error_quantile(),
        )
        required_up = fleet + curtail_growth + storage_charge
        _BALANCE_DIAG.update(fleet=fleet, cg=curtail_growth, storage=storage_charge, margin=margin,
                             quant=quant, dren=renewable_change)
    else:
        margin = max(
            BALANCING_FORECAST_MARGIN_FRACTION * abs(renewable_change) + BALANCING_FORECAST_MARGIN_MW,
            _renewable_error_quantile(),
        )
        required_up = load_change - renewable_change - chronic_change + storage_charge
    ramp_up = np.asarray(cls.gen_max_ramp_up, dtype=float)
    ramp_down = np.asarray(cls.gen_max_ramp_down, dtype=float)
    p_max = np.asarray(cls.gen_pmax, dtype=float)
    p_min = np.asarray(cls.gen_pmin, dtype=float)
    redispatch = np.asarray(action.redispatch, dtype=float)
    room_up = np.maximum(np.minimum(ramp_up, np.maximum(p_max - gen_p, 0.0)) - np.maximum(redispatch, 0.0), 0.0)
    room_down = np.maximum(np.minimum(ramp_down, np.maximum(gen_p - p_min, 0.0)) - np.maximum(-redispatch, 0.0), 0.0)
    available_up = float(np.sum(room_up[redispatchable]))
    available_down = float(np.sum(room_down[redispatchable]))
    return required_up + margin, available_up, -required_up + margin, available_down


def _balancing_cap_mw(env, obs):
    cls = type(env)
    redispatchable = np.asarray(cls.gen_redispatchable, dtype=bool)
    renewable = np.asarray(cls.gen_renewable, dtype=bool)
    gen_p = np.asarray(obs.gen_p, dtype=float)
    ramp_up = np.asarray(cls.gen_max_ramp_up, dtype=float)
    p_max = np.asarray(cls.gen_pmax, dtype=float)
    available_up = float(np.sum(np.minimum(ramp_up, np.maximum(p_max - gen_p, 0.0))[redispatchable]))
    renewables_next, load_next = _expected_renewables_next(env, obs)
    change = (load_next - float(np.sum(obs.load_p))) - (renewables_next - float(gen_p[renewable].sum()))
    if BALANCING_FORM == "fleet":
        fleet = _expected_fleet_increment(env, obs)
        if fleet is not None:
            change = fleet
    return max(0.0, BALANCING_HEADROOM_FRACTION * available_up - max(0.0, change))


STORAGE_RELEASE_FRACTION = float(os.environ.get("ACOPF_STORAGE_RELEASE_FRACTION", "0.2"))
STORAGE_SUSTAIN_H = float(os.environ.get("ACOPF_STORAGE_SUSTAIN_H", "0.5"))
STORAGE_ABSORB = os.environ.get("ACOPF_STORAGE_ABSORB", "1") == "1"


def _storage_sustain_caps(env, obs):
    cls = type(env)
    n = int(getattr(cls, "n_storage", 0) or 0)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    e_max = np.asarray(cls.storage_Emax, dtype=float)
    e_min = np.asarray(cls.storage_Emin, dtype=float)
    charge = np.asarray(obs.storage_charge, dtype=float)
    eta_c = np.asarray(getattr(cls, "storage_charging_efficiency", np.ones(n)), dtype=float)
    eta_d = np.asarray(getattr(cls, "storage_discharging_efficiency", np.ones(n)), dtype=float)
    p_absorb = np.asarray(cls.storage_max_p_absorb, dtype=float)
    p_prod = np.asarray(cls.storage_max_p_prod, dtype=float)
    hours = max(STORAGE_SUSTAIN_H, 1.0 / 12.0)
    head_up = np.maximum(0.98 * e_max - charge, 0.0)
    head_dn = np.maximum(charge - np.maximum(e_min, 0.02 * e_max), 0.0)
    cap_c = np.minimum(p_absorb, head_up / (np.maximum(eta_c, 1e-6) * hours))
    cap_d = np.minimum(p_prod, head_dn * np.maximum(eta_d, 1e-6) / hours)
    return cap_c, cap_d


def _clip_storage_to_sustain(env, obs, action):
    if STORAGE_SUSTAIN_H <= 0.0:
        return None
    storage = np.asarray(action.storage_p, dtype=float)
    if storage.size == 0 or not np.any(np.abs(storage) > 0.05):
        return None
    cap_c, cap_d = _storage_sustain_caps(env, obs)
    clipped = np.clip(storage, -cap_d, cap_c)
    if np.allclose(clipped, storage, atol=0.05):
        return None
    kept = {}
    redispatch = np.asarray(action.redispatch, dtype=float)
    if np.any(np.abs(redispatch) > 1e-9):
        kept["redispatch"] = [(int(g), float(redispatch[g])) for g in np.flatnonzero(np.abs(redispatch) > 1e-9)]
    if np.any(np.abs(clipped) > 0.05):
        kept["set_storage"] = [(int(k), float(clipped[k])) for k in np.flatnonzero(np.abs(clipped) > 0.05)]
    ratios = np.asarray(action.curtail, dtype=float)
    idx = np.flatnonzero(ratios >= 0.0)
    if idx.size:
        kept["curtail"] = [(int(g), float(ratios[g])) for g in idx]
    set_bus = np.asarray(action.set_bus, dtype=int)
    if np.any(set_bus != 0):
        kept["set_bus"] = set_bus
    set_status = np.asarray(action.set_line_status, dtype=int)
    if np.any(set_status != 0):
        kept["set_line_status"] = set_status
    try:
        return env.action_space(kept)
    except Exception:
        return None


def _balancing_feasible(env, obs, action):
    up, avail_up, down, avail_down = _balancing_requirement(env, obs, action)
    if up > BALANCING_HEADROOM_FRACTION * avail_up:
        return False, f"needs +{up:.1f} MW, fleet ramp-up {avail_up:.1f} MW"
    if down > BALANCING_HEADROOM_FRACTION * avail_down:
        return False, f"needs -{down:.1f} MW, fleet ramp-down {avail_down:.1f} MW"
    ordered = float(np.sum(np.asarray(action.storage_p, dtype=float)))
    if abs(ordered) > 1.0:
        room = avail_down if ordered > 0 else avail_up
        step_mw = STORAGE_RELEASE_FRACTION * abs(ordered)
        if step_mw > BALANCING_HEADROOM_FRACTION * room - BALANCING_FORECAST_MARGIN_MW:
            return False, (f"storage order {ordered:+.1f} MW could not be tapered next step "
                           f"({step_mw:.1f} MW step, room {room:.1f} MW)")
    return True, ""


BALANCING_SCALE_FACTORS = (0.75, 0.5, 0.35, 0.2)


def _scale_action(env, obs, action, factor):
    kept = {}
    redispatch = np.asarray(action.redispatch, dtype=float) * factor
    if np.any(np.abs(redispatch) > 1e-9):
        kept["redispatch"] = [(int(g), float(redispatch[g])) for g in np.flatnonzero(np.abs(redispatch) > 1e-9)]
    storage = np.asarray(action.storage_p, dtype=float) * factor
    if np.any(np.abs(storage) > 1e-9):
        kept["set_storage"] = [(int(k), float(storage[k])) for k in np.flatnonzero(np.abs(storage) > 1e-9)]
    ratios = np.asarray(action.curtail, dtype=float)
    limits = np.asarray(obs.curtailment_limit, dtype=float)
    idx = np.flatnonzero(ratios >= 0.0)
    if idx.size:
        kept["curtail"] = [(int(g), float(limits[g] + factor * (ratios[g] - limits[g]))) for g in idx]
    set_bus = np.asarray(action.set_bus, dtype=int)
    if np.any(set_bus != 0):
        kept["set_bus"] = set_bus
    set_status = np.asarray(action.set_line_status, dtype=int)
    if np.any(set_status != 0):
        kept["set_line_status"] = set_status
    if not kept:
        return None
    try:
        return env.action_space(kept)
    except Exception:
        return None


def _strip_releases(env, obs, action):
    ratios = np.asarray(action.curtail, dtype=float)
    limits = np.asarray(obs.curtailment_limit, dtype=float)
    raising = (ratios >= 0.0) & (ratios > limits + 1e-6)
    if not np.any(raising):
        return None
    kept = {}
    redispatch = np.asarray(action.redispatch, dtype=float)
    if np.any(np.abs(redispatch) > 1e-9):
        kept["redispatch"] = [(int(g), float(redispatch[g])) for g in np.flatnonzero(np.abs(redispatch) > 1e-9)]
    lowering = np.flatnonzero((ratios >= 0.0) & ~raising)
    if lowering.size:
        kept["curtail"] = [(int(g), float(ratios[g])) for g in lowering]
    storage = np.asarray(action.storage_p, dtype=float)
    if np.any(np.abs(storage) > 1e-9):
        kept["set_storage"] = [(int(k), float(storage[k])) for k in np.flatnonzero(np.abs(storage) > 1e-9)]
    set_bus = np.asarray(action.set_bus, dtype=int)
    if np.any(set_bus != 0):
        kept["set_bus"] = set_bus
    set_status = np.asarray(action.set_line_status, dtype=int)
    if np.any(set_status != 0):
        kept["set_line_status"] = set_status
    try:
        return env.action_space(kept)
    except Exception:
        return None


def _strip_compensation(env, action):
    kept = {}
    redispatch = np.asarray(action.redispatch, dtype=float)
    if np.any(np.abs(redispatch) > 1e-9):
        kept["redispatch"] = [(int(g), float(redispatch[g])) for g in np.flatnonzero(np.abs(redispatch) > 1e-9)]
    set_bus = np.asarray(action.set_bus, dtype=int)
    if np.any(set_bus != 0):
        kept["set_bus"] = set_bus
    set_status = np.asarray(action.set_line_status, dtype=int)
    if np.any(set_status != 0):
        kept["set_line_status"] = set_status
    try:
        return env.action_space(kept)
    except Exception:
        return None


def _balancing_guard(env, obs, action, label):
    clipped = _clip_storage_to_sustain(env, obs, action)
    if clipped is not None:
        print(f"[balance] step {int(obs.current_step)} {label}: storage orders clipped to what the units can hold "
              f"for {STORAGE_SUSTAIN_H:g} h ({float(np.asarray(action.storage_p).sum()):+.1f} -> "
              f"{float(np.asarray(clipped.storage_p).sum()):+.1f} MW)", flush=True)
        action = clipped
    ok, why = _balancing_feasible(env, obs, action)
    if ok:
        return action
    without_release = _strip_releases(env, obs, action)
    if without_release is not None:
        ok2, _why2 = _balancing_feasible(env, obs, without_release)
        if ok2:
            print(f"[balance] step {int(obs.current_step)} {label}: {why}; kept without curtailment release", flush=True)
            return without_release
    for factor in BALANCING_SCALE_FACTORS:
        scaled = _scale_action(env, obs, action, factor)
        if scaled is None:
            continue
        ok3, _why3 = _balancing_feasible(env, obs, scaled)
        if ok3:
            print(f"[balance] step {int(obs.current_step)} {label}: {why}; scaled to {factor:.2f}", flush=True)
            return scaled
    stripped = _strip_compensation(env, action)
    if stripped is not None and stripped.can_affect_something():
        up, avail_up, down, avail_down = _balancing_requirement(env, obs, stripped)
        hard_infeasible = up > avail_up or down > avail_down
        print(
            f"[balance] step {int(obs.current_step)} {label}: {why}; "
            f"{'rejected even stripped (beyond full fleet ramp)' if hard_infeasible else 'kept without curtailment/storage'}",
            flush=True,
        )
        return None if hard_infeasible else stripped
    print(f"[balance] step {int(obs.current_step)} {label}: {why}; rejected", flush=True)
    return None


def _record_touched_by(env, action, managed_records):
    cls = type(env)
    cumulative = np.cumsum(cls.sub_info)
    touched = {
        int(np.searchsorted(cumulative, position, side="right"))
        for position in np.flatnonzero(np.asarray(action.set_bus, dtype=int) != 0)
    }
    for index, record in enumerate(managed_records):
        if touched & set(record.get("subs", ())):
            return index, record
    return None, None


def _security_accepts(env, obs, action):
    if pace_mod._STATE["mode"] == "minimal":
        return True, {"passed": True, "worst_rho": float("nan"), "results": []}
    lines = None
    if pace_mod._STATE["mode"] != "normal":
        baseline_full = screen_dynamic_contingencies(
            env, obs, env.action_space({}), rho_limit=N1_RHO_LIMIT
        )
        lines = relevant_contingencies(env, obs, baseline_full)
    candidate = screen_dynamic_contingencies(
        env, obs, action, rho_limit=N1_RHO_LIMIT, lines=lines
    )
    if candidate["passed"]:
        return True, candidate
    baseline = screen_dynamic_contingencies(
        env, obs, env.action_space({}), rho_limit=N1_RHO_LIMIT
    )
    if lines is not None:
        baseline = {
            "passed": baseline["passed"],
            "worst_rho": baseline["worst_rho"],
            "results": [item for item in baseline["results"] if int(item["line"]) in lines],
        }
    return contingency_not_worse(
        candidate, baseline, rho_tolerance=float("inf")
    ), candidate


CONTINGENCY_RAMP_HORIZON = float(
    os.environ.get("ACOPF_CONTINGENCY_RAMP_HORIZON", "8")
)
CONTINGENCY_POST_NETWORK = (
    os.environ.get("ACOPF_CONTINGENCY_POST_NETWORK", "1") == "1"
)


def _contingency_acopf_action(env, obs, contingency_report):
    outage = contingency_report.get("line")
    if outage is None or not bool(obs.line_status[int(outage)]):
        return None, None, None
    outage = int(outage)
    forced_outage = env.action_space(
        {"set_line_status": [(outage, -1)]}
    )
    contingency_obs, _reward, done, info = obs.simulate(
        forced_outage, time_step=1
    )
    contingency_rho = (
        maximum_rho(contingency_obs) if not done else float("inf")
    )
    if (
        done
        or info.get("is_illegal", False)
        or info.get("is_ambiguous", False)
        or info.get("exception", [])
        or not np.isfinite(contingency_rho)
    ):
        return None, None, None

    data = extract_grid2op_data_mod(env, contingency_obs, grid_obs=obs)
    if CONTINGENCY_POST_NETWORK:
        outage_data = post_contingency_data(
            data, outage, ramp_horizon=CONTINGENCY_RAMP_HORIZON
        )
        if outage_data is None:
            return None, None, None
        data = outage_data
    model = acopf_setup_twostage_mod(
        data,
        rho_target=RHO_TARGET,
        forecast_up_requirement=0.0,
        forecast_down_requirement=0.0,
        reserve_margin_mw=0.0,
        balancing_cap_mw=_balancing_cap_mw(env, obs),
    )
    model = initialization_twostage_mod(model, data)
    results = solve_acopf_twostage_mod(model, tee=SOLVER_TEE)
    status = results.solver.termination_condition
    if status not in ACCEPTABLE:
        return None, None, None
    action, dictionary = decode_action_twostage_mod(
        env, contingency_obs, model
    )
    action = _balancing_guard(env, obs, action, "contingency_acopf")
    if action is None:
        return None, None, None
    dictionary = action.as_dict()

    base_verdict = validate_action(
        env,
        obs,
        action,
        rho_cap=0.98,
        require_imminent_reset=True,
        horizons=validation_horizons(env, obs),
    )
    if not base_verdict["ok"]:
        return None, None, None
    combined = action + forced_outage
    protected_obs, _reward, done, info = obs.simulate(
        combined, time_step=1
    )
    protected_rho = maximum_rho(protected_obs) if not done else float("inf")
    invalid_transition = (
        done
        or info.get("is_illegal", False)
        or info.get("is_ambiguous", False)
        or info.get("exception", [])
        or not np.isfinite(protected_rho)
    )
    strict = protected_rho < 1.0 - 1e-6
    useful_best_effort = (
        protected_rho <= N1_RHO_LIMIT + 1e-6
        and protected_rho <= contingency_rho - 0.02
    )
    if invalid_transition or not (strict or useful_best_effort):
        return None, None, None
    return action, dictionary, {
        "outage": outage,
        "rho_before": contingency_rho,
        "rho_after": protected_rho,
        "best_effort": not strict,
        "status": str(status),
    }


LOW_RISK_BRIDGE_RHO = float(os.environ.get("ACOPF_LOW_RISK_BRIDGE_RHO", "0.8"))


def _tolerate_low_risk_bridges(env, obs, action, screen):
    alertable = set(map(int, obs.alertable_line_ids))
    try:
        sim_obs, _reward, done, info = obs.simulate(action, time_step=0)
        if done or info.get("exception", []):
            return screen
        candidate_rho = np.asarray(sim_obs.rho, dtype=float)
    except Exception:
        return screen
    maintenance_soon = np.asarray(obs.time_next_maintenance, dtype=int)
    results = []
    for item in screen["results"]:
        line = int(item["line"])
        if (
            not item["ok"]
            and not np.isfinite(float(item.get("rho", float("inf"))))
            and line not in alertable
            and not (0 <= int(maintenance_soon[line]) <= 24)
            and np.isfinite(candidate_rho[line])
            and float(candidate_rho[line]) < LOW_RISK_BRIDGE_RHO
        ):
            item = dict(item)
            item["ok"] = True
            item["rho"] = float(candidate_rho[line])
            item["bridge_rho"] = 0.0
        results.append(item)
    return {"passed": all(r["ok"] for r in results), "worst_rho": screen["worst_rho"], "results": results}


def _not_worse_reasons(candidate, baseline, rho_tolerance):
    base_by_line = {item["line"]: item for item in baseline["results"]}
    reasons = []
    for item in candidate["results"]:
        previous = base_by_line.get(item["line"])
        if previous is None:
            if not item["ok"]:
                reasons.append((int(item["line"]), "new failing"))
            continue
        if previous["ok"] and not item["ok"] and not np.isfinite(item["rho"]):
            reasons.append((int(item["line"]), "new non-convergent"))
        elif (
            np.isfinite(previous["rho"]) and np.isfinite(item["rho"])
            and item["rho"] > previous["rho"] + rho_tolerance
        ):
            reasons.append((int(item["line"]), f"rho {previous['rho']:.3f}->{item['rho']:.3f}"))
        else:
            pb = float(previous.get("bridge_rho", 0.0))
            cb = float(item.get("bridge_rho", 0.0))
            if np.isfinite(pb) and np.isfinite(cb) and cb > pb + rho_tolerance:
                reasons.append((int(item["line"]), f"bridge {pb:.3f}->{cb:.3f} ({item.get('bridge_line')})"))
            elif pb <= 0.95 and cb > 0.95:
                reasons.append((int(item["line"]), f"bridge>0.95 ({item.get('bridge_line')})"))
    return reasons


def _memo_veto(obs, outage):
    from ACOPF.ACOPF_mod import emergency_mod as _emergency

    _emergency._memo_record(obs, int(outage))


RECORD_AUDIT_INTERVAL = int(os.environ.get("ACOPF_RECORD_AUDIT_INTERVAL", "12"))


def _audit_records(env, obs, managed_records, horizons):
    from ACOPF.ACOPF_mod.security_mod import structural_single_outage_failures

    attackable = set(map(int, obs.alertable_line_ids))
    present = structural_single_outage_failures(env, obs) & attackable
    if not present:
        return None
    for index, record in enumerate(managed_records):
        if not cooldown_cleared(obs, record) or not record.get("topo_positions"):
            continue
        restore = build_partial_restore_action(env, record, include_lines=False, obs=obs)
        restored = structural_single_outage_failures(env, obs, restore) & attackable
        removed = present - restored
        if not removed:
            continue
        evaluation = evaluate_transition(env, obs, restore, horizons)
        if evaluation is None or evaluation["max_rho"] > RECOVERY_RHO_LIMIT:
            continue
        print(
            f"[audit] step {int(obs.current_step)} restoring '{record.get('label')}': "
            f"its topology islands the grid on the loss of {sorted(removed)}",
            flush=True,
        )
        return {
            "action": restore,
            "dictionary": {},
            "decision": f"audit_restore:{record.get('label')}",
            "acopf_status": "record_audit",
            "record": None,
            "replace_index": None,
            "trim_record_index": index,
        }
    return None


def _pending_contingency(env, obs, managed_records, horizons):
    if (not ENABLE_PREVENTIVE_SEARCH and not PREVENTIVE_MAINTENANCE_ONLY) or len(
        managed_records
    ) >= PREVENTIVE_MAX_MANAGED:
        return None
    if maximum_rho(obs) > 0.95:
        return None
    for record in managed_records:
        line = record.get("contingency_line")
        if line is None or not bool(obs.line_status[int(line)]):
            continue
        if not str(record.get("label", "")).endswith(")"):
            continue
        post = evaluate_transition(
            env, obs, env.action_space({}), horizons, outage=int(line)
        )
        if post is None or post["n_over"] > 0:
            return int(line)
    return None


ENABLE_COOPT = os.environ.get("ACOPF_COOPT", "1") == "1"
COOPT_CANDIDATES = int(os.environ.get("ACOPF_COOPT_CANDIDATES", "6"))


def cooptimised_preventive_action(env, obs, outage, horizons, base_cap=0.95, budget_s=12.0):
    from ACOPF.ACOPF_mod import emergency_mod as _emergency
    from ACOPF.ACOPF_mod import split_search_mod as _split

    started = time.perf_counter()
    outage = int(outage)
    if not bool(obs.line_status[outage]):
        return None, None, {"reason": "outage_line_already_out"}
    baseline = evaluate_transition(env, obs, env.action_space({}), horizons, outage=outage)
    monitored = [outage]
    if baseline is not None:
        post_rho = np.where(
            np.asarray(baseline["obs"].line_status, dtype=bool),
            np.asarray(baseline["obs"].rho, dtype=float), 0.0,
        )
        monitored = [int(l) for l in np.flatnonzero(post_rho > 0.95)] or [outage]
    region = _split.candidate_substations(env, obs, monitored + [outage], hops=1)
    subs = _emergency._ordered_substations(env, obs, monitored + [outage], region, 8)
    tried = 0
    best = None
    for sub in subs:
        if time.perf_counter() - started > budget_s or tried >= COOPT_CANDIDATES:
            break
        for split_action, _ranked in _emergency._split_candidates(
            env, obs, sub, monitored, outage_lines=(outage,), top=2
        ):
            if tried >= COOPT_CANDIDATES or time.perf_counter() - started > budget_s:
                break
            tried += 1
            if _emergency._reconnects_offline_line(env, obs, split_action):
                continue
            try:
                sim0, _r, done, info = obs.simulate(split_action, time_step=0)
            except Exception:
                continue
            if done or info.get("is_illegal") or info.get("is_ambiguous") or info.get("exception"):
                continue
            try:
                data = extract_grid2op_data_mod(env, sim0)
                post = post_contingency_data(data, outage, ramp_horizon=CONTINGENCY_RAMP_HORIZON)
                if post is None:
                    continue
                model = acopf_setup_twostage_mod(
                    post, rho_target=RHO_TARGET, forecast_up_requirement=0.0,
                    forecast_down_requirement=0.0, reserve_margin_mw=0.0,
                    balancing_cap_mw=_balancing_cap_mw(env, obs),
                )
                model = initialization_twostage_mod(model, post)
                results = solve_acopf_twostage_mod(model, tee=SOLVER_TEE)
                if results.solver.termination_condition not in ACCEPTABLE:
                    continue
                model._acopf_data = data
                dispatch, _dictionary = decode_action_twostage_mod(env, sim0, model)
            except Exception:
                continue
            dispatch = _balancing_guard(env, obs, dispatch, "coopt")
            if dispatch is None:
                continue
            try:
                combined = split_action + dispatch
            except Exception:
                continue
            base = evaluate_transition(env, obs, combined, horizons)
            if base is None or base["max_rho"] > base_cap:
                continue
            post_eval = evaluate_transition(env, obs, combined, horizons, outage=outage)
            if post_eval is None or post_eval["n_over"] > 0:
                continue
            key = (round(post_eval["max_rho"], 4), round(base["max_rho"], 4))
            if best is None or key < best[0]:
                best = (key, combined, sub, base, post_eval)
    meta = {"tried": tried, "elapsed_s": round(time.perf_counter() - started, 2)}
    if best is None:
        meta["reason"] = "no_secured_combination"
        return None, None, meta
    key, combined, sub, base, post_eval = best
    meta["best_key"] = key
    return combined, f"contingency {outage} co-optimised split @ sub {sub} + dispatch", meta


POSTURE_THREATS = os.environ.get("ACOPF_POSTURE_THREATS", "1") == "1"


def _alertable_set(env):
    from ACOPF.ACOPF_mod import posture_mod
    return posture_mod.alertable_lines(env)


CURTAILMENT_WEIGHT_EMERGENCY = float(os.environ.get("ACOPF_CURTAIL_WEIGHT_EMERGENCY", "5.0"))
CURTAILMENT_WEIGHT_PREVENTIVE = float(os.environ.get("ACOPF_CURTAIL_WEIGHT_PREVENTIVE", "50.0"))
STORAGE_WEIGHT_PREVENTIVE = float(os.environ.get("ACOPF_STORAGE_WEIGHT_PREVENTIVE", "0.25"))
STORAGE_WEIGHT_EMERGENCY = float(os.environ.get("ACOPF_STORAGE_WEIGHT_EMERGENCY", "0.25"))
ENABLE_SCOPF_STEP = os.environ.get("ACOPF_SCOPF_STEP", "0") == "1"


PREP_STEP_MIN_GAIN = float(os.environ.get("ACOPF_PREP_STEP_MIN_GAIN", "0.01"))
BLIND_MARCH_STEPS = int(os.environ.get("ACOPF_BLIND_MARCH_STEPS", "1000000"))
MARCH_LAST_STEPS = int(os.environ.get("ACOPF_MARCH_LAST_STEPS", "2"))
_BLIND_MARCH = {"key": None, "count": 0}


MAINT_PREP_TARGET_TIME = os.environ.get("ACOPF_MAINT_PREP_TARGET_TIME", "1") == "1"
PREP_PROJECTED_TOLERANCE = float(os.environ.get("ACOPF_PREP_PROJECTED_TOLERANCE", "0.03"))
MARCH_STEP_SCALES = tuple(float(x) for x in os.environ.get("ACOPF_MARCH_STEP_SCALES", "1,0.5,0.25").split(","))


def _outage_time_observation(env, obs, at_step):
    if at_step is None or int(at_step) < 2:
        return None
    try:
        fc_obs, _r, done, info = obs.simulate(env.action_space({}), time_step=int(at_step) - 1)
    except Exception:
        return None
    if done or not valid_transition(done, info, maximum_rho(fc_obs)):
        return None
    return fc_obs


def _balanced_march_step(env, obs, model, data, tolerance_mw=0.5, scale=1.0):
    gens = [int(g) for g in model.REDISPATCHABLE_GEN]
    gen_p = np.asarray(obs.gen_p, dtype=float)
    cur = {g: float(gen_p[g]) for g in gens}
    des = {g: max(0.0, float(pyo.value(model.generator_p[g]))) for g in gens}
    total_cur, total_des = sum(cur.values()), sum(des.values())
    if total_des <= 1e-6 or total_cur <= 1e-6:
        return None
    k = total_cur / total_des
    delta = {g: des[g] * k - cur[g] for g in gens}
    room = max(0.05, min(1.0, float(scale)))
    ups = {g: min(d, room * 0.999 * float(data.generator_ramp_up[g]), float(data.generator_p_max[g]) - cur[g])
           for g, d in delta.items() if d > tolerance_mw}
    downs = {g: max(d, -room * 0.999 * float(data.generator_ramp_down[g]), float(data.generator_p_min[g]) - cur[g])
             for g, d in delta.items() if d < -tolerance_mw}
    ups = {g: v for g, v in ups.items() if v > 0.05}
    downs = {g: v for g, v in downs.items() if v < -0.05}
    sum_up, sum_down = sum(ups.values()), -sum(downs.values())
    if sum_up < tolerance_mw or sum_down < tolerance_mw:
        return None
    moved = min(sum_up, sum_down)
    orders = [(g, v * moved / sum_up) for g, v in ups.items()] + [(g, v * moved / sum_down) for g, v in downs.items()]
    orders = [(g, float(v)) for g, v in orders if abs(v) > 0.05]
    if not orders:
        return None
    try:
        return env.action_space({"redispatch": orders})
    except Exception:
        return None


def contingency_dispatch_step(env, obs, outage, horizons, base_cap=0.95, min_gain=0.10, require_gain=False,
                              demand_scale=1.0, target_time=False):
    outages = [int(l) for l in (outage if isinstance(outage, (list, tuple, set, frozenset)) else [outage])]
    outages = [l for l in outages if bool(obs.line_status[l])]
    if not outages:
        return None, None, {"reason": "outage_line_already_out"}
    outage = outages[0]
    tnm = np.asarray(obs.time_next_maintenance, dtype=int)
    at_step = int(tnm[outage]) if all(0 < tnm[l] <= 12 for l in outages) and len({int(tnm[l]) for l in outages}) == 1 else None
    data_obs = obs
    note = None
    if not (MAINT_PREP_TARGET_TIME and target_time):
        demand_scale = 1.0
    elif MAINT_PREP_TARGET_TIME and at_step is not None:
        fc_obs = _outage_time_observation(env, obs, at_step)
        if fc_obs is not None:
            data_obs, demand_scale, note = fc_obs, 1.0, "outage-time forecast"

    def _post_state(action):
        if at_step is None:
            return evaluate_transition(env, obs, action, horizons, outage=outage)
        try:
            sim_obs, _r, done, info = obs.simulate(action, time_step=at_step)
        except Exception:
            return None
        rho = maximum_rho(sim_obs) if not done else float("nan")
        if not valid_transition(done, info, rho):
            return None
        live = np.asarray(sim_obs.rho, dtype=float)[np.asarray(sim_obs.line_status, dtype=bool)]
        return {"n_over": int(np.sum(live > 1.0)), "max_rho": float(rho)}

    baseline = _post_state(env.action_space({}))
    try:
        data = extract_grid2op_data_mod(env, data_obs)
        post = post_contingency_data(data, outages[0], ramp_horizon=CONTINGENCY_RAMP_HORIZON)
        for extra in outages[1:]:
            post = post_contingency_data(post, extra, ramp_horizon=1.0) if post is not None else None
        if post is None:
            return None, None, {"reason": "degenerate_post_contingency_network"}
        if MAINT_PREP_TARGET_TIME and target_time and demand_scale > 1.0 + 1e-6:
            post = dataclasses.replace(
                post,
                load_p={d: float(v) * float(demand_scale) for d, v in post.load_p.items()},
                load_q={d: float(v) * float(demand_scale) for d, v in post.load_q.items()},
            )
            note = f"post demand x{float(demand_scale):.2f}"
        reserve = EMERGENCY_RESERVE_FLOOR * _renewable_error_quantile(EMERGENCY_RESERVE_QUANTILE)
        req_up, _au, req_down, _ad = _balancing_requirement(env, obs, env.action_space({}))
        model = acopf_setup_twostage_mod(
            post, rho_target=RHO_TARGET, forecast_up_requirement=max(reserve, req_up),
            forecast_down_requirement=max(reserve, req_down), reserve_margin_mw=RESERVE_MARGIN_MW,
            balancing_cap_mw=_balancing_cap_mw(env, obs),
        )
        model = initialization_twostage_mod(model, post)
        results = solve_acopf_twostage_mod(model, tee=SOLVER_TEE)
        if results.solver.termination_condition not in ACCEPTABLE:
            return None, None, {"reason": f"solver_{results.solver.termination_condition}"}
        model._acopf_data = data
        dispatch, _dictionary = decode_action_twostage_mod(env, obs, model)
        march_model = model if note else None
    except Exception as error:
        return None, None, {"reason": f"error:{error}"}
    hold_base = evaluate_transition(env, obs, env.action_space({}), horizons)
    effective_cap = min(0.99, max(base_cap, (hold_base["max_rho"] + 0.01) if hold_base else base_cap))
    if march_model is not None:
        dispatch, base = None, None
        for _scale in MARCH_STEP_SCALES:
            candidate = _balanced_march_step(env, obs, march_model, data, scale=_scale)
            if candidate is None:
                continue
            candidate = _balancing_guard(env, obs, candidate, "scopf")
            if candidate is None:
                continue
            evaluation = evaluate_transition(env, obs, candidate, horizons)
            if evaluation is not None and evaluation["max_rho"] <= effective_cap:
                dispatch, base = candidate, evaluation
                if _scale < 1.0:
                    note = f"{note}, step x{_scale:g}"
                break
        if dispatch is None:
            return None, None, {"reason": "no_feasible_march_step", "target": note}
    else:
        dispatch = _balancing_guard(env, obs, dispatch, "scopf")
        if dispatch is None:
            return None, None, {"reason": "balancing_infeasible"}
        base = evaluate_transition(env, obs, dispatch, horizons)
    if base is None or base["max_rho"] > effective_cap:
        detail = {"reason": "base_case_over_cap", "base": None if base is None else round(base["max_rho"], 3), "cap": round(effective_cap, 3)}
        if base is None:
            for _h in horizons:
                try:
                    _s, _r, _d, _info = obs.simulate(dispatch, time_step=int(_h))
                    if _d or _info.get("exception") or _info.get("is_illegal") or _info.get("is_ambiguous"):
                        detail["why"] = f"h{int(_h)}: " + (str(_info.get("exception"))[:80] if _info.get("exception") else
                                                          "illegal" if _info.get("is_illegal") else "ambiguous" if _info.get("is_ambiguous") else f"done={_d}")
                        break
                    _lost = [int(l) for l in np.flatnonzero(np.asarray(obs.line_status, dtype=bool) & ~np.asarray(_s.line_status, dtype=bool))]
                    if _lost:
                        detail["why"] = f"h{int(_h)}: lost {_lost[:4]}"
                        break
                except Exception as error:
                    detail["why"] = f"h{int(_h)}: {str(error)[:80]}"
                    break
        return None, None, detail
    post_eval = _post_state(dispatch)
    if post_eval is None:
        if baseline is None:
            key = (tuple(outages), int(obs.current_step) + int(tnm[outage]))
            if _BLIND_MARCH["key"] != key:
                _BLIND_MARCH.update(key=key, count=0)
            if _BLIND_MARCH["count"] >= BLIND_MARCH_STEPS:
                return None, None, {"reason": "blind_march_budget", "steps": _BLIND_MARCH["count"], "target": note}
            _BLIND_MARCH["count"] += 1
            return dispatch, f"contingency {'+'.join(str(l) for l in outages)} dispatch step", {
                "post": None, "baseline": None, "base": round(base["max_rho"], 3), "note": "toward feasible target",
                "blind": _BLIND_MARCH["count"]}
        detail = {"reason": "post_contingency_unsolvable", "target": note}
        if at_step is not None:
            try:
                _s, _r, _d, _info = obs.simulate(dispatch, time_step=at_step)
                detail["why"] = (str(_info.get("exception"))[:90] if _info.get("exception") else
                                 f"done={_d} illegal={_info.get('is_illegal')} rho={maximum_rho(_s) if not _d else float('nan'):.3f}")
            except Exception as error:
                detail["why"] = str(error)[:90]
        return None, None, detail
    baseline_rho = baseline["max_rho"] if baseline is not None else float("inf")
    improved = post_eval["max_rho"] <= baseline_rho - min_gain
    if note:
        exact = at_step is not None
        gain = post_eval["max_rho"] <= baseline_rho - PREP_STEP_MIN_GAIN
        if gain or (not exact and post_eval["max_rho"] <= baseline_rho + PREP_PROJECTED_TOLERANCE):
            return dispatch, f"contingency {'+'.join(str(l) for l in outages)} dispatch step", {
                "post": round(post_eval["max_rho"], 3), "baseline": round(baseline_rho, 3),
                "base": round(base["max_rho"], 3), "target": note}
        return None, None, {"reason": "no_projected_gain" if exact else "worse_today", "post": round(post_eval["max_rho"], 3), "baseline": round(baseline_rho, 3), "target": note}
    if require_gain:
        any_gain = post_eval["max_rho"] <= baseline_rho - PREP_STEP_MIN_GAIN
        if not (improved or (post_eval["n_over"] > 0 and any_gain)):
            return None, None, {"reason": "no_material_gain", "post": round(post_eval["max_rho"], 3), "baseline": round(baseline_rho, 3)}
    elif not (post_eval["n_over"] == 0 or improved):
        return None, None, {"reason": "no_material_gain", "post": round(post_eval["max_rho"], 3), "baseline": round(baseline_rho, 3)}
    label_lines = "+".join(str(l) for l in outages)
    meta = {"post": round(post_eval["max_rho"], 3), "baseline": round(baseline_rho, 3), "base": round(base["max_rho"], 3)}
    if note:
        meta["target"] = note
    return dispatch, f"contingency {label_lines} dispatch step", meta


def _contingency_dispatch_for_screen(env, obs, screen, horizons):
    security = dynamic_security_set(env, obs)
    hazards = {int(l) for l in security["opponent"]} | {int(l) for l in security["maintenance"]}
    failing = [
        item for item in screen["results"]
        if not item["ok"] and int(item["line"]) in hazards and np.isfinite(float(item.get("rho", 0.0)))
        and float(item.get("rho", 0.0)) > 1.0
    ]
    if not failing:
        return None
    failing.sort(key=lambda item: -float(item["rho"]))
    for item in failing[:2]:
        action, label, meta = contingency_dispatch_step(env, obs, int(item["line"]), horizons)
        if action is None:
            continue
        print(f"[scopf] step {int(obs.current_step)} {label}: {meta}", flush=True)
        return {
            "action": action,
            "dictionary": action.as_dict(),
            "decision": f"preventive_{label.replace(' ', '_')}",
            "acopf_status": "scopf_step",
            "record": None,
            "replace_index": None,
        }
    return None


MAINT_PREP = os.environ.get("ACOPF_MAINT_PREP", "1") == "1"
MAINT_PREP_STEPS = int(os.environ.get("ACOPF_MAINT_PREP_STEPS", "48"))
MAINT_PREP_TRIGGER_RHO = float(os.environ.get("ACOPF_MAINT_PREP_TRIGGER_RHO", "0.8"))
MAINT_PREP_REQUIRE_GAIN = os.environ.get("ACOPF_MAINT_PREP_REQUIRE_GAIN", "1") == "1"
RELEASE_PREP_FLOOR = os.environ.get("ACOPF_RELEASE_PREP_FLOOR", "1") == "1"
_PREP_STATE = {"step": -1, "outage_step": -1, "lines": (), "floor_mw": 0.0}


def _curtailed_mw_of(env, obs, action):
    cls = type(env)
    renewable = np.asarray(cls.gen_renewable, dtype=bool)
    p_max = np.asarray(cls.gen_pmax, dtype=float)
    ratios = np.asarray(action.curtail, dtype=float)
    limits = np.asarray(obs.curtailment_limit, dtype=float)
    new_limits = np.where(ratios >= 0.0, ratios, limits)
    try:
        gen_f, _v, _l, _q = obs.get_forecasted_inj(1)
        f = np.asarray(gen_f, dtype=float)
    except Exception:
        f = np.asarray(obs.gen_p, dtype=float)
    return float(np.sum(np.maximum(f - new_limits * p_max, 0.0)[renewable & (new_limits < 0.999)]))


def _release_floor_mw(obs):
    if not RELEASE_PREP_FLOOR or _PREP_STATE["floor_mw"] <= 0.0:
        return 0.0
    step = int(obs.current_step)
    if step < _PREP_STATE["step"]:
        _PREP_STATE.update(step=-1, outage_step=-1, lines=(), floor_mw=0.0)
        return 0.0
    if step > _PREP_STATE["outage_step"]:
        return 0.0
    status = np.asarray(obs.line_status, dtype=bool)
    tnm = np.asarray(obs.time_next_maintenance, dtype=int)
    if not all(bool(status[l]) and int(tnm[l]) > 0 for l in _PREP_STATE["lines"]):
        return 0.0
    return float(_PREP_STATE["floor_mw"])


MAINT_TOPO = os.environ.get("ACOPF_MAINT_TOPO", "1") == "1"
MAINT_TOPO_TOTAL_BUDGET_S = float(os.environ.get("ACOPF_MAINT_TOPO_BUDGET_S", "90"))
MAINT_TOPO_INTERVAL = int(os.environ.get("ACOPF_MAINT_TOPO_INTERVAL", "8"))
_MAINT_TOPO_STATE = {"_step": -1}


def _maintenance_topology(env, obs, screen, managed_records, horizons, group, first, growth):
    from ACOPF.ACOPF_mod import emergency_mod as _emergency
    step = int(obs.current_step)
    if step < _MAINT_TOPO_STATE.get("_step", -1):
        _MAINT_TOPO_STATE.clear()
    _MAINT_TOPO_STATE["_step"] = step
    key = tuple(sorted(int(l) for l in group))
    st = _MAINT_TOPO_STATE.setdefault(key, {"attempts": 0, "last_step": -10 ** 6, "spent": 0.0, "done": False})
    if st["done"] or st["spent"] >= MAINT_TOPO_TOTAL_BUDGET_S:
        return None
    guards = [r for r in managed_records
              if r.get("contingency_line") is not None and int(r["contingency_line"]) in key]
    if guards and not all(str(r.get("label", "")).endswith(")") for r in guards):
        st["done"] = True
        return None
    if step - st["last_step"] < MAINT_TOPO_INTERVAL:
        return None
    if maximum_rho(obs) > 0.95:
        return None
    st["last_step"] = step; st["attempts"] += 1
    for l in list(key) + [key]:
        _emergency._PREVENTIVE_MEMO.pop(l, None)
    _emergency.set_search_offset(st["attempts"] - 1)
    if int(first) <= 12:
        horizons = tuple(sorted(set(int(h) for h in horizons) | {int(first)}))
        _emergency.set_rho_scale(1.0)
    else:
        _emergency.set_rho_scale(growth)
    started = time.perf_counter()
    try:
        result = _preventive_split(env, obs, screen, managed_records, horizons, only_lines=set(key), force=True,
                                   overload_rho=min(PREVENTIVE_OVERLOAD_RHO, 1.0 / max(float(growth), 1e-6)),
                                   outage_group=key)
    finally:
        _emergency.set_search_offset(0)
        _emergency.set_rho_scale(1.0)
        st["spent"] += time.perf_counter() - started
    if result is not None:
        st["done"] = True
        print(f"[maint-topo] step {step} maintenance of {list(key)} in {first} steps: {result.get('decision')} "
              f"(attempt {st['attempts']}, {st['spent']:.0f}s)", flush=True)
        result["decision"] = f"maintenance_topology:{result.get('decision')}"
        if isinstance(result.get("record"), dict):
            result["record"]["contingency_group"] = list(key)
        return result
    print(f"[maint-topo] step {step} maintenance of {list(key)} in {first} steps: no securing topology yet "
          f"(attempt {st['attempts']}, {st['spent']:.0f}s of {MAINT_TOPO_TOTAL_BUDGET_S:g}s)", flush=True)
    return None


def _maintenance_preparation(env, obs, horizons, managed_records=None):
    if not MAINT_PREP:
        return None
    tnm = np.asarray(obs.time_next_maintenance, dtype=int)
    status = np.asarray(obs.line_status, dtype=bool)
    soon = [int(l) for l in np.flatnonzero(status & (tnm > 0) & (tnm <= MAINT_PREP_STEPS))]
    if not soon:
        return None
    first = min(int(tnm[l]) for l in soon)
    group = [l for l in soon if int(tnm[l]) == first]
    screen = screen_dynamic_contingencies(
        env, obs, env.action_space({}), rho_limit=1.0, time_step=max(horizons), lines=(), extra_lines=group)
    growth = _demand_ratio(obs, first / 12.0)
    failing = any(int(it["line"]) in group and (not np.isfinite(it["rho"]) or it["rho"] * growth >= MAINT_PREP_TRIGGER_RHO) for it in screen["results"])
    outage_time_rho = None
    if first <= 12:
        try:
            sim_obs, _r, done, info = obs.simulate(env.action_space({}), time_step=first)
            rho_at = maximum_rho(sim_obs) if not done else float("nan")
            outage_time_rho = rho_at if valid_transition(done, info, rho_at) else float("inf")
        except Exception:
            outage_time_rho = float("inf")
        failing = failing or not np.isfinite(outage_time_rho) or outage_time_rho >= MAINT_PREP_TRIGGER_RHO
    if not failing:
        return None
    severe = any(int(it["line"]) in group and (not np.isfinite(it["rho"]) or it["rho"] * growth >= 1.0)
                 for it in screen["results"])
    if outage_time_rho is not None:
        severe = severe or not np.isfinite(outage_time_rho) or outage_time_rho >= 1.0
    if MAINT_TOPO and severe and managed_records is not None:
        topo = _maintenance_topology(env, obs, screen, managed_records, horizons, group, first, growth)
        if topo is not None:
            return topo
    if MAINT_LODF_SCOPF and managed_records is not None:
        from ACOPF.ACOPF_mod import maint_context as _mctx
        fc_obs = _outage_time_observation(env, obs, first) if first <= 12 else None
        step_now = int(obs.current_step)
        _sev = float("nan")
        try:
            _hold_fc0, _r0, _d0, _i0 = obs.simulate(env.action_space({}), time_step=1)
            _sev = _mctx.surrogate_worst(env, obs, group, growth, _hold_fc0, fc_obs)
        except Exception:
            pass
        if _mctx.dispatch_ineffective(obs, first, group):
            terms, note = None, "dispatch marked ineffective for this outage (retried after 12 steps): topology / march only"
        elif not (np.isfinite(_sev) and _sev >= MAINT_SCOPF_SEVERE):
            terms, note = None, f"predicted post-outage loading {_sev:.3f} below the severity gate {MAINT_SCOPF_SEVERE:.2f}: v99 preparations only"
        else:
            terms, note = _mctx.terms_for_group(env, obs, group, growth, state_obs=fc_obs, first=first)
        if terms is None:
            if not note.startswith("predicted post-outage loading"):
                print(f"[maint-scopf] step {step_now} maintenance of {group} in {first} steps: no dispatch context ({note})", flush=True)
            if note.startswith("dispatch marked ineffective"):
                pass
        else:
            standing = float(np.sum(np.abs(np.asarray(obs.actual_dispatch, dtype=float))))
            if standing > _mctx.DISPATCH_CAP_MW:
                print(f"[maint-scopf] step {step_now} maintenance of {group} in {first} steps: standing dispatch {standing:.0f} MW (telemetry; the flexibility gate decides)", flush=True)
            try:
                result = solve_current_step(env, obs, managed_records, contingency_terms=terms, curtailment_weight_override=MAINT_SCOPF_CURTAIL_WEIGHT)
            except Exception as error:
                result = None
                print(f"[maint-scopf] step {step_now} failed: {error!r}", flush=True)
            hold_fc = None
            try:
                hold_fc, _r, _d, _i = obs.simulate(env.action_space({}), time_step=1)
            except Exception:
                pass
            worst_hold = _mctx.surrogate_worst(env, obs, group, growth, hold_fc, fc_obs) if hold_fc is not None else float("nan")
            if result is not None and result.get("action") is not None and result["action"].can_affect_something() \
                    and not str(result.get("decision", "")).startswith(("do_nothing", "fallback_do_nothing")):
                worst_cand = float("nan")
                cand_fc = None
                try:
                    cand_fc, _r, done, info = obs.simulate(result["action"], time_step=1)
                    if not done and not info.get("exception") and not info.get("is_illegal"):
                        worst_cand = _mctx.surrogate_worst(env, obs, group, growth, cand_fc, fc_obs)
                except Exception:
                    pass
                progress = (np.isfinite(worst_cand) and np.isfinite(worst_hold)
                            and (worst_cand <= worst_hold - _mctx.PROGRESS_MIN or worst_cand <= _mctx.SECURE_RHO))
                try:
                    fleet_move = _fleet_move_next(env, obs)
                except Exception:
                    fleet_move = 0.0
                try:
                    move_err = _fleet_move_error_quantile(BALANCING_ERROR_QUANTILE)
                    move_err = float(move_err) if move_err is not None else 0.0
                except Exception:
                    move_err = 0.0
                flex_ok, flex_up, flex_down = (False, float("nan"), float("nan"))
                if cand_fc is not None:
                    flex_ok, flex_up, flex_down = _mctx.flexibility_ok(
                        env, cand_fc, need_up=max(0.0, fleet_move) + move_err, need_down=max(0.0, -fleet_move) + move_err)
                if progress and not flex_ok:
                    print(f"[maint-scopf] step {step_now} maintenance of {group} in {first} steps: action rejected by the flexibility gate "
                          f"(room after the action up {flex_up:.0f} / down {flex_down:.0f} MW against the fleet move {fleet_move:+.0f} MW)", flush=True)
                    progress = False
                if _mctx.record_progress(obs, first, group, worst_hold, worst_cand):
                    print(f"[maint-scopf] step {step_now} maintenance of {group}: dispatch ineffective ({_mctx.INEFFECTIVE_STREAK} solves under {_mctx.INEFFECTIVE_MIN_GAIN} gain): handing over to topology / march", flush=True)
                    progress = False
                if progress:
                    print(f"[maint-scopf] step {step_now} maintenance of {group} in {first} steps: {note}; "
                          f"predicted post-outage loading hold {worst_hold:.3f} -> action {worst_cand:.3f}: {result.get('decision')}", flush=True)
                    result["decision"] = f"maintenance_scopf:{result.get('decision')}"
                    return result
                print(f"[maint-scopf] step {step_now} maintenance of {group} in {first} steps: {note}; action rejected "
                      f"(predicted post-outage loading hold {worst_hold:.3f} -> action {worst_cand:.3f})", flush=True)
            elif np.isfinite(worst_hold) and worst_hold <= _mctx.SECURE_RHO:
                print(f"[maint-scopf] step {step_now} maintenance of {group} in {first} steps: {note}; the constrained dispatch holds and the surrogate reads the hold as secure ({worst_hold:.3f}); the usual preparations continue", flush=True)
    if first <= MARCH_LAST_STEPS:
        return None
    action, label, meta = contingency_dispatch_step(env, obs, group, horizons, base_cap=0.95, min_gain=0.05,
                                                    require_gain=MAINT_PREP_REQUIRE_GAIN and not severe,
                                                    demand_scale=growth, target_time=severe)
    if action is None:
        detail = {k: v for k, v in meta.items() if k != "reason"}
        print(f"[maint-prep] step {int(obs.current_step)} maintenance of {group} in {first} steps: no dispatch step ({meta.get('reason')}){' ' + str(detail) if detail else ''}", flush=True)
        return None
    try:
        _PREP_STATE.update(step=int(obs.current_step), outage_step=int(obs.current_step) + int(first),
                           lines=tuple(int(l) for l in group), floor_mw=_curtailed_mw_of(env, obs, action))
    except Exception:
        pass
    print(f"[maint-prep] step {int(obs.current_step)} maintenance of {group} in {first} steps: {label} {meta} "
          f"floor={_PREP_STATE['floor_mw']:.0f} MW", flush=True)
    return {
        "action": action,
        "dictionary": action.as_dict(),
        "decision": f"maintenance_prep:{label}",
        "acopf_status": "scopf_step",
        "record": None,
        "replace_index": None,
    }


RESERVE_WATCH = os.environ.get("ACOPF_RESERVE_WATCH", "1") == "1"
CHRONIC_CHANGE_SANITY_MW = float(os.environ.get("ACOPF_CHRONIC_CHANGE_SANITY_MW", "200"))

ATTACK_POSTURE_TARGET = float(os.environ.get("ACOPF_ATTACK_POSTURE_TARGET", "0.5"))
ENABLE_ATTACK_POSTURE = os.environ.get("ACOPF_ATTACK_POSTURE", "1") == "1"


RISK_SCOPF = os.environ.get("ACOPF_RISK_SCOPF", "0") == "1"
RISK_SCOPF_MONITOR_RHO = float(os.environ.get("ACOPF_RISK_SCOPF_MONITOR_RHO", "0.9"))
RISK_SCOPF_MAX_OUTAGES = int(os.environ.get("ACOPF_RISK_SCOPF_MAX_OUTAGES", "4"))


def _risk_contingency_terms(env, obs, screen):
    if not RISK_SCOPF or screen is None:
        return None
    from ACOPF.ACOPF_mod import posture_mod

    q = posture_mod.attack_response(env, obs)
    p_now = np.asarray(obs.p_or, dtype=float)
    rho_now = np.asarray(obs.rho, dtype=float)
    status = np.asarray(obs.line_status, dtype=bool)
    risky = []
    for item in screen["results"]:
        line = int(item["line"])
        if line not in q or item.get("p_or") is None or item.get("rho_all") is None:
            continue
        sev = posture_mod.severity(item)
        if sev <= 0.0 or abs(p_now[line]) < 1.0:
            continue
        risky.append((q[line] * sev, line, item))
    if not risky:
        return None
    risky.sort(key=lambda r: -r[0])
    q_max = max(q[l] for _r, l, _i in risky)
    terms = []
    for _risk, line, item in risky[:RISK_SCOPF_MAX_OUTAGES]:
        post_p = np.asarray(item["p_or"], dtype=float)
        post_rho = np.asarray(item["rho_all"], dtype=float)
        monitored = {}
        for k in np.flatnonzero(status & np.isfinite(post_rho) & (post_rho >= RISK_SCOPF_MONITOR_RHO)):
            k = int(k)
            if k == line or rho_now[k] < 0.05 or not np.isfinite(rho_now[k]):
                continue
            lodf = float((post_p[k] - p_now[k]) / p_now[line])
            if not np.isfinite(lodf) or abs(lodf) > 3.0:
                continue
            monitored[k] = {"lodf": lodf, "pmax_mw": float(abs(p_now[k]) / rho_now[k])}
        if monitored:
            terms.append({"outage": line, "weight": float(q[line] / q_max), "monitored": monitored})
    return terms or None


def _attack_posture_lines(env, obs, screen):
    if not ENABLE_ATTACK_POSTURE or screen is None:
        return None
    from ACOPF.ACOPF_mod import posture_mod, capacity_mod
    status = np.asarray(obs.line_status, dtype=bool)
    rho = np.asarray(obs.rho, dtype=float)
    upper = set()
    armed_lines = []
    for group in posture_mod.attackable_groups(env, obs):
        live = [int(l) for l in group if status[int(l)]]
        if not live:
            continue
        ranked = sorted(live, key=lambda l: rho[l] if np.isfinite(rho[l]) else 0.0)
        upper.update(ranked[len(ranked) // 2:])
        armed_lines.extend(live)
    if not upper:
        return None
    failing = {
        int(item["line"]) for item in screen["results"]
        if not item["ok"] and int(item["line"]) in upper
    }
    failing |= capacity_mod.fatal_lines(env, obs, [l for l in armed_lines if l in upper], demand_scale=_peak_factor(obs))
    if not failing:
        return None
    return {l: min(ATTACK_POSTURE_TARGET, float(rho[l])) for l in failing}


ATTACK_AWARE_RESTORE = os.environ.get("ACOPF_ATTACK_AWARE_RESTORE", "1") == "1"
NEED_CHECK_INTERVAL = int(os.environ.get("ACOPF_NEED_CHECK_INTERVAL", "3"))
NEED_RHO = float(os.environ.get("ACOPF_NEED_RHO", "0.92"))
NEED_FORECAST_HORIZONS = tuple(int(x) for x in os.environ.get("ACOPF_NEED_FORECAST_HORIZONS", "6,12").split(",") if x)


def _forecast_peak_rho(env, obs, action, horizons):
    worst = float("nan")
    for h in horizons:
        try:
            sim, _r, done, info = obs.simulate(action, time_step=int(h))
        except Exception:
            continue
        if done or info.get("is_illegal") or info.get("is_ambiguous") or info.get("exception"):
            continue
        live = np.asarray(sim.line_status, dtype=bool)
        rho = np.asarray(sim.rho, dtype=float)[live]
        if rho.size and np.all(np.isfinite(rho)):
            worst = float(rho.max()) if not np.isfinite(worst) else max(worst, float(rho.max()))
    return worst
NEED_MIN_RELIEF = float(os.environ.get("ACOPF_NEED_MIN_RELIEF", "0.02"))
RESTORE_MIN_RISK_GAIN = float(os.environ.get("ACOPF_RESTORE_MIN_RISK_GAIN", "0.005"))
STEP_LOG = os.environ.get("ACOPF_STEP_LOG", "0") == "1"


MAINTENANCE_RISK_HORIZON_STEPS = int(os.environ.get("ACOPF_MAINT_RISK_STEPS", "96"))
MAINTENANCE_NEED_HORIZON_STEPS = int(os.environ.get("ACOPF_MAINT_NEED_STEPS", "36"))


def _maintenance_within(obs, steps):
    tnm = np.asarray(obs.time_next_maintenance, dtype=int)
    status = np.asarray(obs.line_status, dtype=bool)
    return {int(l) for l in np.flatnonzero(status & (tnm >= 0) & (tnm <= int(steps)))}


RISK_TIE_EPS = float(os.environ.get("ACOPF_RISK_TIE_EPS", "1e-6"))


def _key_better(candidate, best, eps=RISK_TIE_EPS):
    for left, right in zip(candidate, best):
        if isinstance(left, float) or isinstance(right, float):
            if float(left) < float(right) - eps:
                return True
            if float(left) > float(right) + eps:
                return False
        elif left != right:
            return left < right
    return False


def _attack_aware_restoration(env, obs, blocked_lines, horizons):
    from ACOPF.ACOPF_mod import posture_mod, capacity_mod
    from ACOPF.ACOPF_mod.recovery_mod import line_restoration_candidates
    from ACOPF.ACOPF_mod.security_mod import topology_protection_safe, structural_outage_failures_for

    candidates = line_restoration_candidates(env, obs, blocked_lines)
    if not candidates:
        return None, None, None
    hold = validate_action(env, obs, env.action_space({}), horizons=horizons)
    if not hold["ok"] or hold.get("forecast_obs") is None:
        return None, None, None
    hold_obs = hold["forecast_obs"]
    maint_risk = _maintenance_within(obs, MAINTENANCE_RISK_HORIZON_STEPS)
    maint_need = _maintenance_within(obs, MAINTENANCE_NEED_HORIZON_STEPS)
    struct_hold = structural_outage_failures_for(env, obs, maint_need) if maint_need else set()
    hold_maint_screen = None
    if maint_need:
        hold_maint_screen = screen_dynamic_contingencies(
            env, obs, env.action_space({}), rho_limit=1.0, time_step=max(horizons), lines=(), extra_lines=maint_need)
        struct_hold = struct_hold | {int(it["line"]) for it in hold_maint_screen["results"]
                                     if int(it["line"]) in maint_need and (not np.isfinite(it["rho"]) or it["rho"] >= 1.0)}
    need_ratio = _demand_ratio(obs, NEED_HORIZON_H)
    peak_ratio = _peak_factor(obs)
    live = np.asarray(hold_obs.line_status, dtype=bool)
    hold_max = float(np.max(np.asarray(hold_obs.rho, dtype=float)[live])) if live.any() else float("inf")
    hold_projected = hold_max * need_ratio
    if NEED_FORECAST_HORIZONS:
        fc = _forecast_peak_rho(env, obs, env.action_space({}), NEED_FORECAST_HORIZONS)
        if np.isfinite(fc):
            hold_projected = max(hold_projected, fc)
    step = int(obs.current_step)
    hold_screen = None
    hold_risk = None
    best = None
    for line, action in candidates:
        safe, _structural = topology_protection_safe(env, obs, action)
        if not safe:
            continue
        verdict = validate_action(env, obs, action, rho_cap=None, require_imminent_reset=True, horizons=horizons)
        if not verdict["ok"] or verdict.get("forecast_obs") is None:
            continue
        cand_obs = verdict["forecast_obs"]
        c_live = np.asarray(cand_obs.line_status, dtype=bool)
        cand_max = float(np.max(np.asarray(cand_obs.rho, dtype=float)[c_live])) if c_live.any() else float("inf")
        cand_projected = cand_max * need_ratio
        if NEED_FORECAST_HORIZONS and hold_projected >= NEED_RHO:
            fc = _forecast_peak_rho(env, obs, action, NEED_FORECAST_HORIZONS)
            cand_projected = max(cand_projected, fc) if np.isfinite(fc) else hold_projected
        if cand_max > max(hold_max, RECOVERY_RHO_LIMIT) + 0.005:
            continue
        arms = any(
            not np.asarray(obs.line_status, dtype=bool)[g].all() and c_live[g].all()
            for g in posture_mod.areas(env)
        )
        need = hold_projected >= NEED_RHO and cand_projected <= hold_projected - NEED_MIN_RELIEF
        struct_fixed = set()
        if struct_hold:
            cand_maint = screen_dynamic_contingencies(
                env, obs, action, rho_limit=1.0, time_step=max(horizons), lines=(), extra_lines=maint_need)
            still_bad = structural_outage_failures_for(env, obs, maint_need, action) | {
                int(it["line"]) for it in cand_maint["results"]
                if int(it["line"]) in maint_need and (not np.isfinite(it["rho"]) or it["rho"] >= 1.0)}
            struct_fixed = struct_hold - still_bad
        if struct_hold:
            key = (-2 if struct_fixed else -1, -len(struct_fixed), cand_max)
            reason = (f"need: maintenance of {sorted(struct_hold)} within {MAINTENANCE_NEED_HORIZON_STEPS} steps "
                      f"would island the grid as it stands" + (" (this reconnection removes the cut)" if struct_fixed else ""))
        elif need:
            key = (0, -(hold_projected - cand_projected))
            reason = f"need: projected {hold_projected:.3f} -> {cand_projected:.3f} within {NEED_HORIZON_H:g} h"
        else:
            if step % RESTORATION_CHECK_INTERVAL != 0:
                continue
            if step - _OPEN_STATE["opened"].get(int(line), -10 ** 6) < OPEN_RESTORE_BLOCK:
                continue
            if hold_screen is None:
                hold_screen = screen_dynamic_contingencies(
                    env, obs, env.action_space({}), rho_limit=N1_RHO_LIMIT, time_step=max(horizons), extra_lines=maint_risk)
                armed = [l for l, qv in posture_mod.attack_response(env, hold_obs).items() if qv > 0]
                fatal = capacity_mod.fatal_lines(env, hold_obs, armed + sorted(maint_risk), demand_scale=peak_ratio)
                r, _terms = posture_mod.expected_loss(env, hold_obs, hold_screen, demand_scale=peak_ratio, fatal=fatal, certain=maint_risk)
                hold_risk = r + posture_mod.base_severity(hold_obs, peak_ratio)
            cand_screen = screen_dynamic_contingencies(
                env, obs, action, rho_limit=N1_RHO_LIMIT, time_step=max(horizons), extra_lines=maint_risk)
            unsolved_hold = sum(1 for it in hold_screen["results"] if not np.isfinite(it["rho"]))
            unsolved_cand = sum(1 for it in cand_screen["results"] if not np.isfinite(it["rho"]))
            if unsolved_cand > unsolved_hold:
                continue
            armed = [l for l, qv in posture_mod.attack_response(env, cand_obs).items() if qv > 0]
            fatal = capacity_mod.fatal_lines(env, cand_obs, armed + sorted(maint_risk), demand_scale=peak_ratio)
            r, _terms = posture_mod.expected_loss(env, cand_obs, cand_screen, demand_scale=peak_ratio, fatal=fatal, certain=maint_risk)
            cand_risk = r + posture_mod.base_severity(cand_obs, peak_ratio)
            if cand_risk > hold_risk - RESTORE_MIN_RISK_GAIN:
                if arms:
                    print(f"[restore] step {step} line {line} stays out: risk {hold_risk:.3f} -> {cand_risk:.3f} (re-arms its area)", flush=True)
                continue
            key = (1, cand_risk - hold_risk)
            reason = f"risk {hold_risk:.3f} -> {cand_risk:.3f}"
        if best is None or _key_better(key, best[0]):
            best = (key, action, line, verdict, reason)
    if best is None:
        return None, None, None
    _key, action, line, verdict, reason = best
    _OPEN_STATE["restored"][int(line)] = step
    print(f"[restore] step {step} reconnect line {line}: {reason}", flush=True)
    return action, f"reconnect line {line}", verdict


CALM_MODE = os.environ.get("ACOPF_CALM_MODE", "0") == "1"
CALM_NO_PREVENTIVE = os.environ.get("ACOPF_CALM_NO_PREVENTIVE", "1") == "1"
CALM_MAINT_BLOCK_STEPS = int(os.environ.get("ACOPF_CALM_MAINT_BLOCK_STEPS", "48"))
CALM_NO_REARM_WITH_MAINT = os.environ.get("ACOPF_CALM_NO_REARM_WITH_MAINT", "1") == "1"
CALM_DISARM_INTERLOCK = os.environ.get("ACOPF_CALM_DISARM_INTERLOCK", "1") == "1"
CALM_RECONNECT_WORST = float(os.environ.get("ACOPF_CALM_RECONNECT_WORST", "0.95"))
CALM_RETRY_STEPS = int(os.environ.get("ACOPF_CALM_RETRY_STEPS", "12"))
CALM_RHO = float(os.environ.get("ACOPF_CALM_RHO", "1.0"))
CALM_BASE_RHO = float(os.environ.get("ACOPF_CALM_BASE_RHO", "0.95"))
CALM_RECONNECT_RHO = float(os.environ.get("ACOPF_CALM_RECONNECT_RHO", "0.95"))
CALM_HORIZON_H = float(os.environ.get("ACOPF_CALM_HORIZON_H", "4"))
CALM_CHECK_MAX = int(os.environ.get("ACOPF_CALM_CHECK_MAX", "6"))
CALM_THIN_MARGIN = float(os.environ.get("ACOPF_CALM_THIN_MARGIN", "0.05"))
CALM_RISING_RHO = float(os.environ.get("ACOPF_CALM_RISING_RHO", "0.02"))
CALM_HYSTERESIS = float(os.environ.get("ACOPF_CALM_HYSTERESIS", "0.05"))
CALM_DISARM_SKIP = int(os.environ.get("ACOPF_CALM_DISARM_SKIP", "96"))
CALM_DISARM_MIN_GAIN = float(os.environ.get("ACOPF_CALM_DISARM_MIN_GAIN", "0.20"))
CALM_LOG = os.environ.get("ACOPF_CALM_LOG", "0") == "1"
_CALM_STATE = {"step": -1, "calm": False, "last_step": -1, "interval": 1, "worst": 0.0, "rho": 0.0,
               "reconnections": 0, "checks": 0, "returned": {}, "calm_until": -1}


def _disarm_gain_required(step):
    from ACOPF.ACOPF_mod import posture_mod
    if not posture_mod.OPPONENT_ABORT_MODEL:
        return OPEN_MIN_GAIN
    return CALM_DISARM_MIN_GAIN if step <= _CALM_STATE["calm_until"] else DISARM_MIN_GAIN


def _calm_hazards(env, obs, status):
    from ACOPF.ACOPF_mod import posture_mod
    hazards = set(_maintenance_within(obs, MAINTENANCE_RISK_HORIZON_STEPS))
    for group in posture_mod.areas(env):
        if status[group].all():
            hazards.update(int(l) for l in group)
    return hazards


def _calm_worst(screen, hazards, scale):
    worst = 0.0
    for item in screen["results"]:
        if int(item["line"]) not in hazards:
            continue
        rho = float(item["rho"])
        if not np.isfinite(rho):
            return float("inf")
        worst = max(worst, rho * scale)
    return worst


def _calm_interval(worst, rho_now):
    if not np.isfinite(worst) or worst >= CALM_RHO - CALM_THIN_MARGIN:
        return 1
    if rho_now > _CALM_STATE["rho"] + CALM_RISING_RHO:
        return 1
    margin = CALM_RHO - worst
    return int(min(CALM_CHECK_MAX, max(1, round(margin / CALM_THIN_MARGIN))))


def _calm_week(env, obs, hold, horizons):
    step = int(obs.current_step)
    if step <= _CALM_STATE["last_step"]:
        _CALM_STATE.update(step=-1, calm=False, interval=1, worst=0.0, rho=0.0, reconnections=0, checks=0, returned={}, calm_until=-1)
    _CALM_STATE["last_step"] = step
    was = _CALM_STATE["calm"]
    rho_now = maximum_rho(obs)
    if not hold["ok"] or hold.get("forecast_obs") is None or dynamic_security_set(env, obs)["protection"] \
            or rho_now > CALM_BASE_RHO:
        calm = False
        reason = "protection or loading"
    elif CALM_MAINT_BLOCK_STEPS > 0 and _maintenance_within(obs, CALM_MAINT_BLOCK_STEPS):
        calm = False
        reason = f"maintenance within {CALM_MAINT_BLOCK_STEPS} steps"
    elif _CALM_STATE["step"] >= 0 and step - _CALM_STATE["step"] < _CALM_STATE["interval"]:
        return was
    else:
        _CALM_STATE["checks"] += 1
        scale = _demand_ratio(obs, CALM_HORIZON_H)
        forecast = hold["forecast_obs"]
        live = np.asarray(forecast.line_status, dtype=bool)
        rho = np.asarray(forecast.rho, dtype=float)
        base = float(np.max(rho[live])) if live.any() and np.all(np.isfinite(rho[live])) else float("inf")
        worst = base * scale
        if worst <= CALM_RHO:
            maint = _maintenance_within(obs, MAINTENANCE_RISK_HORIZON_STEPS)
            screen = screen_dynamic_contingencies(
                env, obs, env.action_space({}), rho_limit=N1_RHO_LIMIT, time_step=max(horizons), extra_lines=maint)
            worst = max(worst, _calm_worst(screen, _calm_hazards(env, obs, live), scale))
        threshold = CALM_RHO + (CALM_HYSTERESIS if was else -CALM_HYSTERESIS)
        calm = worst <= threshold
        reason = f"worst {worst:.3f} at x{scale:.2f}"
        _CALM_STATE.update(step=step, interval=_calm_interval(worst, rho_now), worst=worst, rho=rho_now)
        if CALM_LOG:
            print(f"[calm] step {step} check: {reason}, next in {_CALM_STATE['interval']}", flush=True)
    _CALM_STATE["calm"] = calm
    if calm:
        _CALM_STATE["calm_until"] = step + CALM_DISARM_SKIP
    if calm != was:
        print(f"[calm] step {step} {'enter' if calm else 'leave'} calm mode ({reason})", flush=True)
    return calm


def _calm_restoration(env, obs, blocked_lines, horizons):
    from ACOPF.ACOPF_mod.recovery_mod import line_restoration_candidates
    from ACOPF.ACOPF_mod.security_mod import topology_protection_safe

    cls = type(env)
    sub_cooldown = np.asarray(obs.time_before_cooldown_sub, dtype=int)
    scale = _demand_ratio(obs, CALM_HORIZON_H)
    maint = _maintenance_within(obs, MAINTENANCE_RISK_HORIZON_STEPS)
    best = None
    for line, action in line_restoration_candidates(env, obs, blocked_lines):
        if sub_cooldown[int(cls.line_or_to_subid[line])] > 0 or sub_cooldown[int(cls.line_ex_to_subid[line])] > 0:
            continue
        safe, _structural = topology_protection_safe(env, obs, action)
        if not safe:
            continue
        verdict = validate_action(env, obs, action, rho_cap=CALM_RECONNECT_RHO, require_imminent_reset=True, horizons=horizons)
        if not verdict["ok"] or verdict.get("forecast_obs") is None:
            continue
        status_after = np.asarray(verdict["forecast_obs"].line_status, dtype=bool)
        if CALM_NO_REARM_WITH_MAINT and maint:
            from ACOPF.ACOPF_mod import posture_mod as _pm
            status_now = np.asarray(obs.line_status, dtype=bool)
            rearms_with_maint = any(
                (not status_now[g].all()) and status_after[g].all() and any(int(l) in maint for l in g)
                for g in _pm.areas(env)
            )
            if rearms_with_maint:
                if CALM_LOG:
                    print(f"[calm] step {int(obs.current_step)} line {line} stays out: would re-arm an area with a maintenance due", flush=True)
                continue
        screen = screen_dynamic_contingencies(
            env, obs, action, rho_limit=N1_RHO_LIMIT, time_step=max(horizons), extra_lines=maint)
        worst = _calm_worst(screen, _calm_hazards(env, obs, status_after), scale)
        if int(obs.current_step) - _CALM_STATE["returned"].get(int(line), -10 ** 9) < CALM_RETRY_STEPS:
            continue
        if worst > CALM_RECONNECT_WORST:
            if CALM_LOG:
                print(f"[calm] step {int(obs.current_step)} line {line} stays out: worst case {worst:.3f} with it back", flush=True)
            continue
        key = (worst, float(verdict["reports"][-1]["rho"]))
        if best is None or key < best[0]:
            best = (key, action, int(line), verdict)
    if best is None:
        return None, None, None
    (worst, rho_after), action, line, verdict = best
    _CALM_STATE["reconnections"] += 1
    _CALM_STATE["interval"] = 1
    _CALM_STATE["returned"][line] = int(obs.current_step)
    print(f"[calm] step {int(obs.current_step)} reconnect line {line}: base {rho_after:.3f}, "
          f"worst attack/maintenance case {worst:.3f} projected", flush=True)
    return action, f"reconnect line {line}", verdict


def calm_stats():
    return f"checks {_CALM_STATE['checks']} reconnections {_CALM_STATE['reconnections']}"


AREA_DISARM = os.environ.get("ACOPF_AREA_DISARM", "1") == "1"
DISARM_CHECK_INTERVAL = int(os.environ.get("ACOPF_DISARM_CHECK_INTERVAL", "6"))
DISARM_MAX_RHO = float(os.environ.get("ACOPF_DISARM_MAX_RHO", "0.95"))
DISARM_FAR_HORIZONS = tuple(int(h) for h in os.environ.get("ACOPF_DISARM_FAR_HORIZONS", "3,6").split(",") if h.strip())
DISARM_MIN_GAIN = float(os.environ.get("ACOPF_DISARM_MIN_GAIN", "0.02"))
DISARM_CANDIDATES_PER_AREA = int(os.environ.get("ACOPF_DISARM_CANDIDATES", "3"))
_DISARM_STATE = {"step": -1}


def _disarm_reset(obs):
    from ACOPF.ACOPF_mod import emergency_mod as _emergency
    step = int(obs.current_step)
    if step <= _DISARM_STATE["step"]:
        _emergency.SELF_OPENED.clear()
        _OPEN_STATE["opened"].clear(); _OPEN_STATE["restored"].clear()
    _DISARM_STATE["step"] = step
    status = np.asarray(obs.line_status, dtype=bool)
    _emergency.SELF_OPENED.difference_update({l for l in list(_emergency.SELF_OPENED) if status[l]})


DISARM_CHEAPEST_FIRST = os.environ.get("ACOPF_DISARM_CHEAPEST_FIRST", "1") == "1"
DISARM_SKIP_IF_MAINT_DISARMS = os.environ.get("ACOPF_DISARM_SKIP_IF_MAINT", "1") == "1"
DISARM_MAINT_HORIZON = int(os.environ.get("ACOPF_DISARM_MAINT_HORIZON", "48"))


def _disarm_candidate_cost(env, obs, line):
    from ACOPF.ACOPF_mod.security_mod import outage_induced_bridges
    cost = abs(float(np.asarray(obs.p_or, dtype=float)[int(line)]))
    try:
        if int(line) in {int(b) for b in outage_induced_bridges(env, obs)}:
            cost += 1e4
    except Exception:
        pass
    return cost


ATTACK_SURVIVE_DISPATCH = os.environ.get("ACOPF_ATTACK_SURVIVE_DISPATCH", "0") == "1"
PREVENTIVE_ATTACK_DISPATCH = os.environ.get("ACOPF_PREVENTIVE_ATTACK_DISPATCH", "0") == "1"
ATTACK_SURVIVE_MIN_RISK = float(os.environ.get("ACOPF_ATTACK_SURVIVE_MIN_RISK", "0.10"))
OPEN_MIN_GAIN = float(os.environ.get("ACOPF_OPEN_MIN_GAIN", "0.10"))
ATTACK_TOPOLOGY_MIN_RISK = float(os.environ.get("ACOPF_ATTACK_TOPOLOGY_MIN_RISK", "0.05"))
ATTACK_TOPOLOGY_MIN_SEVERITY = float(os.environ.get("ACOPF_ATTACK_TOPOLOGY_MIN_SEVERITY", "0.3"))
OPEN_MIN_SEVERITY = float(os.environ.get("ACOPF_OPEN_MIN_SEVERITY", "0.8"))
OPEN_REOPEN_BLOCK = int(os.environ.get("ACOPF_OPEN_REOPEN_BLOCK", "288"))
QUIET_SPEND_MWSTEPS = float(os.environ.get("ACOPF_QUIET_SPEND_MWSTEPS", "3000"))
QUIET_OPEN_FACTOR = float(os.environ.get("ACOPF_QUIET_OPEN_FACTOR", "5.0"))
_SPEND = {"step": -1, "mwsteps": 0.0}


def _track_spend(env, obs):
    step = int(obs.current_step)
    if step <= _SPEND["step"]:
        _SPEND["mwsteps"] = 0.0
    _SPEND["step"] = step
    try:
        cls = type(env)
        redisp = float(np.sum(np.abs(np.asarray(obs.actual_dispatch, dtype=float))))
        renewable = np.asarray(cls.gen_renewable, dtype=bool)
        curtail = float(np.sum(np.maximum(np.asarray(obs.gen_p_before_curtail, dtype=float) - np.asarray(obs.gen_p, dtype=float), 0.0)[renewable]))
        storage = float(np.sum(np.abs(np.asarray(getattr(obs, "storage_power", np.zeros(0)), dtype=float))))
        _SPEND["mwsteps"] += redisp + curtail + storage
    except Exception:
        pass


def _week_may_still_score():
    return _SPEND["mwsteps"] < QUIET_SPEND_MWSTEPS
OPEN_COST_WINDOW_STEPS = int(os.environ.get("ACOPF_OPEN_COST_WINDOW_STEPS", "288"))
OPEN_RESTORE_BLOCK = int(os.environ.get("ACOPF_OPEN_RESTORE_BLOCK", "24"))
_OPEN_STATE = {"opened": {}, "restored": {}}
ATTACK_SURVIVE_MIN_SEVERITY = float(os.environ.get("ACOPF_ATTACK_SURVIVE_MIN_SEVERITY", "0.5"))
HAZARD_FROM_ENV = os.environ.get("ACOPF_HAZARD_FROM_ENV", "1") == "1"
EPISODE_LOSS_POINTS = float(os.environ.get("ACOPF_EPISODE_LOSS_POINTS", "54.0"))
SCORE_PER_MW_STEP = float(os.environ.get("ACOPF_SCORE_PER_MW_STEP", "5.2e-3"))
_HAZARD = {"rate": None}
_THREAT_STATS = {"survive_dispatch": 0, "opened": 0, "open_not_worth": 0}


def _forecast_losses(forecast_obs):
    try:
        gen = float(np.sum(np.asarray(forecast_obs.gen_p, dtype=float)))
        load = float(np.sum(np.asarray(forecast_obs.load_p, dtype=float)))
        if not (np.isfinite(gen) and np.isfinite(load)) or load <= 0.0:
            return None
        return gen - load
    except Exception:
        return None


def _attack_hazard_rate(env):
    if not HAZARD_FROM_ENV:
        return 0.0
    if _HAZARD["rate"] is None:
        from ACOPF.ACOPF_mod import posture_mod as _pm
        rate = float(_pm.ATTACK_HAZARD_RATE)
        _HAZARD["rate"] = rate
        if rate > 0.0:
            print(f"[threat] opponent hazard rate {rate:.6f} per step per area (mean wait {1.0 / rate:.0f} steps)", flush=True)
    return _HAZARD["rate"]


def _opening_is_worth_it(env, hold_obs, cand_obs, risk_gain, window):
    h = _attack_hazard_rate(env)
    if h <= 0.0:
        return True, 0.0, 0.0
    arrival = 1.0 - (1.0 - h) ** max(1, int(window))
    benefit = arrival * max(0.0, risk_gain) * EPISODE_LOSS_POINTS
    hold_loss, cand_loss = _forecast_losses(hold_obs), _forecast_losses(cand_obs)
    extra = max(0.0, cand_loss - hold_loss) if (hold_loss is not None and cand_loss is not None) else 0.0
    cost = extra * max(1, int(window)) * SCORE_PER_MW_STEP
    factor = QUIET_OPEN_FACTOR if _week_may_still_score() else 1.0
    return benefit > factor * cost, benefit, cost


def _threat_terms(env, obs, horizons):
    from ACOPF.ACOPF_mod import posture_mod, capacity_mod
    hold = validate_action(env, obs, env.action_space({}), horizons=horizons)
    if not hold["ok"] or hold.get("forecast_obs") is None:
        return None, 0.0, None
    hold_obs = hold["forecast_obs"]
    peak = _peak_factor(obs)
    maint = _maintenance_within(obs, MAINTENANCE_RISK_HORIZON_STEPS)
    screen = screen_dynamic_contingencies(
        env, obs, env.action_space({}), rho_limit=N1_RHO_LIMIT, time_step=max(horizons), extra_lines=maint)
    q = posture_mod.attack_response(env, hold_obs)
    fatal = capacity_mod.fatal_lines(env, hold_obs, [l for l, qv in q.items() if qv > 0] + sorted(maint), demand_scale=peak)
    risk, terms = posture_mod.expected_loss(env, hold_obs, screen, demand_scale=peak, fatal=fatal, certain=maint)
    return terms, risk, hold_obs, screen, fatal


_SURVIVE_STATE = {"step": -1, "tried": {}, "hold_until": -1, "threat": None, "threat_step": -1}


def _attack_survivability_dispatch(env, obs, horizons, step):
    out = _threat_terms(env, obs, horizons)
    _SURVIVE_STATE["threat"], _SURVIVE_STATE["threat_step"] = out, step
    if not ATTACK_SURVIVE_DISPATCH:
        return None, None, None
    terms = out[0] if out else None
    if not terms:
        return None, None, None
    _terms, _risk, _hold_obs, screen, fatal = out
    by_line = {int(it["line"]): it for it in screen["results"]}
    status = np.asarray(obs.line_status, dtype=bool)
    if step <= 1:
        _SURVIVE_STATE["tried"].clear()
    tried = _SURVIVE_STATE["tried"]
    ranked = sorted(((c, l, sev) for l, (_p, sev, c) in terms.items() if status[int(l)]), reverse=True)
    for contribution, line, sev in ranked[:2]:
        if contribution < ATTACK_SURVIVE_MIN_RISK or sev < ATTACK_SURVIVE_MIN_SEVERITY:
            break
        item = by_line.get(int(line))
        loading_threat = item is not None and np.isfinite(float(item.get("rho", float("nan")))) and float(item["rho"]) > 1.0
        if int(line) in fatal or not loading_threat:
            continue
        if step - tried.get(int(line), -10 ** 6) < DISARM_CHECK_INTERVAL:
            continue
        action, label, meta = contingency_dispatch_step(env, obs, [int(line)], horizons, base_cap=0.98, min_gain=0.05, require_gain=True)
        if action is None:
            tried[int(line)] = step
            continue
        _THREAT_STATS["survive_dispatch"] += 1
        _SURVIVE_STATE["hold_until"] = step + 2 * DISARM_CHECK_INTERVAL
        print(f"[survive] step {step} dispatch against an attack on line {line} "
              f"(q*severity {contribution:.3f}, severity {sev:.2f}): {label} {meta}", flush=True)
        return action, f"attack survivability dispatch vs {line}", None
    return None, None, None


_HARDEN_STATE = {"next": 0, "status": None, "step": -1}


def _hardening_due(obs, step):
    st = _HARDEN_STATE
    if step <= st["step"]:
        st.update(next=0, status=None)
    st["step"] = step
    status = tuple(np.asarray(obs.line_status, dtype=bool).tolist())
    changed = st["status"] is not None and status != st["status"]
    st["status"] = status
    return step < 6 or changed or step >= st["next"]


def _hardening_done(step):
    out = _SURVIVE_STATE.get("threat") if _SURVIVE_STATE.get("threat_step") == step else None
    worst = 0.0
    if out and out[0]:
        worst = max((sev for (_p, sev, _c) in out[0].values()), default=0.0)
    _HARDEN_STATE["next"] = step + (DISARM_CHECK_INTERVAL if worst >= ATTACK_SURVIVE_MIN_SEVERITY else 2 * DISARM_CHECK_INTERVAL)


def _attack_survivability_topology(env, obs, horizons, step, managed_records):
    out = _SURVIVE_STATE.get("threat") if _SURVIVE_STATE.get("threat_step") == step else None
    if not out or not out[0]:
        return None
    terms, _risk, _hold_obs, screen, fatal = out
    status = np.asarray(obs.line_status, dtype=bool)
    ranked = sorted(((c, l, sev) for l, (_p, sev, c) in terms.items() if status[int(l)]), reverse=True)
    for contribution, line, sev in ranked[:3]:
        if contribution < ATTACK_TOPOLOGY_MIN_RISK:
            break
        if int(line) not in fatal and sev < ATTACK_TOPOLOGY_MIN_SEVERITY:
            continue
        if QR_POSTURE and len(managed_records) < TEMPLATE_MAX_RECORDS and maximum_rho(obs) <= 0.98:
            threats = [it for it in screen["results"] if int(it["line"]) == int(line)]
            if threats:
                started = time.perf_counter()
                action, label, verdict = _qr_posture_choice(env, obs, screen, threats, None, horizons, started)
                if action is not None:
                    record = _managed_record(env, obs, action, label, int(obs.current_step), None)
                    try:
                        record["contingency_line"] = int(label.split()[1])
                    except (IndexError, ValueError, AttributeError):
                        pass
                    _THREAT_STATS["survive_topology"] = _THREAT_STATS.get("survive_topology", 0) + 1
                    print(f"[survive] step {step} posture split against an attack on line {line} "
                          f"(q*severity {contribution:.3f}, severity {sev:.2f}): {label}", flush=True)
                    return {"action": action, "dictionary": {}, "decision": f"attack survivability topology vs {line}:posture:{label}",
                            "acopf_status": "contingency_bridge_protection", "record": record, "replace_index": None}
        result = _preventive_split(env, obs, screen, managed_records, horizons, only_lines=(int(line),), force=True)
        if result is not None:
            _THREAT_STATS["survive_topology"] = _THREAT_STATS.get("survive_topology", 0) + 1
            print(f"[survive] step {step} topology against an attack on line {line} "
                  f"(q*severity {contribution:.3f}, severity {sev:.2f}): {result.get('decision')}", flush=True)
            result["decision"] = f"attack survivability topology vs {line}:{result.get('decision')}"
            return result
    return None


def _area_disarm(env, obs, horizons):
    from ACOPF.ACOPF_mod import posture_mod, capacity_mod, emergency_mod as _emergency
    from ACOPF.ACOPF_mod.security_mod import (
        topology_protection_safe, structural_single_outage_failures, dynamic_security_set,
    )

    status = np.asarray(obs.line_status, dtype=bool)
    rho_now = np.asarray(obs.rho, dtype=float)
    cooldown = np.asarray(obs.time_before_cooldown_line, dtype=int)
    maint_in = np.asarray(obs.time_next_maintenance, dtype=int)
    armed = [g for g in posture_mod.attackable_groups(env, obs) if int(status[g].sum()) >= 2]
    if not armed:
        return None, None, None
    peak = _peak_factor(obs)
    maint = _maintenance_within(obs, MAINTENANCE_RISK_HORIZON_STEPS)
    cached = _SURVIVE_STATE.get("threat") if _SURVIVE_STATE.get("threat_step") == int(obs.current_step) else None
    if cached and cached[0] is not None:
        hold_terms, r, hold_obs, hold_screen, fatal = cached
    else:
        hold = validate_action(env, obs, env.action_space({}), horizons=horizons)
        if not hold["ok"] or hold.get("forecast_obs") is None:
            return None, None, None
        hold_obs = hold["forecast_obs"]
        hold_screen = screen_dynamic_contingencies(
            env, obs, env.action_space({}), rho_limit=N1_RHO_LIMIT, time_step=max(horizons), extra_lines=maint)
        q_hold = posture_mod.attack_response(env, hold_obs)
        fatal = capacity_mod.fatal_lines(env, hold_obs, [l for l, qv in q_hold.items() if qv > 0] + sorted(maint), demand_scale=peak)
        r, hold_terms = posture_mod.expected_loss(env, hold_obs, hold_screen, demand_scale=peak, fatal=fatal, certain=maint)
    hold_risk = r + posture_mod.base_severity(hold_obs, peak)
    unsolved_hold = sum(1 for it in hold_screen["results"] if not np.isfinite(it["rho"]))
    security = dynamic_security_set(env, obs)
    hazards = {int(l) for l in security["opponent"]} | {int(l) for l in security["maintenance"]} | maint
    base_failures = structural_single_outage_failures(env, obs) & hazards
    from ACOPF.ACOPF_mod.security_mod import structural_outage_failures_for
    if maint and structural_outage_failures_for(env, obs, maint):
        return None, None, None
    if any(int(it["line"]) in maint and (not np.isfinite(it["rho"]) or it["rho"] >= 1.0) for it in hold_screen["results"]):
        return None, None, None
    def area_risk(group):
        return sum(hold_terms.get(int(l), (0, 0, 0))[2] for l in group)
    best = None
    step = int(obs.current_step)
    for group in sorted(armed, key=area_risk, reverse=True):
        if posture_mod.OPPONENT_ABORT_MODEL and DISARM_SKIP_IF_MAINT_DISARMS and any(0 <= maint_in[l] <= DISARM_MAINT_HORIZON for l in group):
            due = min(int(maint_in[l]) for l in group if 0 <= maint_in[l] <= DISARM_MAINT_HORIZON)
            print(f"[disarm] step {step} area left armed: a maintenance outage disarms it in {due} steps", flush=True)
            continue
        cands = [l for l in group if status[l] and cooldown[l] == 0 and not (0 <= maint_in[l] <= 24) and np.isfinite(rho_now[l])]
        if not posture_mod.OPPONENT_ABORT_MODEL:
            cands = [l for l in cands
                     if hold_terms.get(int(l), (0, 0, 0))[1] >= OPEN_MIN_SEVERITY
                     and step - _OPEN_STATE["restored"].get(int(l), -10 ** 6) >= OPEN_REOPEN_BLOCK]
        cheapest_first = DISARM_CHEAPEST_FIRST and posture_mod.OPPONENT_ABORT_MODEL
        cands = sorted(cands, key=(lambda l: _disarm_candidate_cost(env, obs, l)) if cheapest_first
                       else (lambda l: -hold_terms.get(int(l), (0, 0, 0))[2]))[:DISARM_CANDIDATES_PER_AREA]
        for line in cands:
            action = env.action_space({"set_line_status": [(int(line), -1)]})
            safe, _structural = topology_protection_safe(env, obs, action)
            if not safe:
                continue
            if (structural_single_outage_failures(env, obs, action) & hazards) - base_failures:
                continue
            if maint and structural_outage_failures_for(env, obs, maint, action):
                continue
            verdict = validate_action(env, obs, action, rho_cap=DISARM_MAX_RHO, require_imminent_reset=True, horizons=horizons)
            if not verdict["ok"] or verdict.get("forecast_obs") is None:
                continue
            if DISARM_FAR_HORIZONS and len(horizons) > 1:
                far = [h for h in DISARM_FAR_HORIZONS if h > max(horizons)]
                if far:
                    far_verdict = validate_action(env, obs, action, rho_cap=DISARM_MAX_RHO,
                                                  require_imminent_reset=False, horizons=tuple(far))
                    hold_far = validate_action(env, obs, env.action_space({}), rho_cap=DISARM_MAX_RHO,
                                               require_imminent_reset=False, horizons=tuple(far))
                    if not far_verdict["ok"] and hold_far["ok"]:
                        _THREAT_STATS["open_far_horizon"] = _THREAT_STATS.get("open_far_horizon", 0) + 1
                        print(f"[disarm] step {step} line {int(line)} not opened: {far_verdict['reports'][-1].get('reason', 'far horizon')} "
                              f"at horizon {far[-1]} (rho {far_verdict['reports'][-1].get('rho', float('nan')):.3f})", flush=True)
                        continue
            cand_obs = verdict["forecast_obs"]
            cand_screen = screen_dynamic_contingencies(env, obs, action, rho_limit=N1_RHO_LIMIT, time_step=max(horizons), extra_lines=maint)
            if sum(1 for it in cand_screen["results"] if not np.isfinite(it["rho"])) > unsolved_hold:
                continue
            if any(int(it["line"]) in maint and (not np.isfinite(it["rho"]) or it["rho"] >= 1.0) for it in cand_screen["results"]):
                continue
            q_c = posture_mod.attack_response(env, cand_obs)
            fatal_c = capacity_mod.fatal_lines(env, cand_obs, [l for l, qv in q_c.items() if qv > 0] + sorted(maint), demand_scale=peak)
            rc, terms = posture_mod.expected_loss(env, cand_obs, cand_screen, demand_scale=peak, fatal=fatal_c, certain=maint)
            cand_risk = rc + posture_mod.base_severity(cand_obs, peak)
            if not posture_mod.OPPONENT_ABORT_MODEL:
                remaining = max(1, int(getattr(obs, "max_step", 2016) or 2016) - step)
                window = min(remaining, OPEN_COST_WINDOW_STEPS)
                worth, benefit, cost = _opening_is_worth_it(env, hold_obs, cand_obs, hold_risk - cand_risk, window)
                if not worth:
                    _THREAT_STATS["open_not_worth"] += 1
                    continue
            if best is None or _key_better((cand_risk,), (best[0],)):
                best = (cand_risk, action, int(line), verdict, terms)
            if cheapest_first and hold_risk - cand_risk >= _disarm_gain_required(step):
                break
        if best is not None and hold_risk - best[0] >= _disarm_gain_required(step):
            break
    if best is None or hold_risk - best[0] < _disarm_gain_required(step):
        return None, None, None
    risk, action, line, verdict, terms = best
    _emergency.SELF_OPENED.add(line)
    _OPEN_STATE["opened"][int(line)] = step
    _THREAT_STATS["opened"] += 1
    kind = "disarm" if posture_mod.OPPONENT_ABORT_MODEL else "open"
    print(f"[{kind}] step {step} open line {line}: risk {hold_risk:.3f} -> {risk:.3f} ({posture_mod.format_terms(hold_terms)} | after: {posture_mod.format_terms(terms)})", flush=True)
    return action, f"{'disarm area' if posture_mod.OPPONENT_ABORT_MODEL else 'risk opening'}: open line {line}", verdict


QR_POSTURE = os.environ.get("ACOPF_QR_POSTURE", "1") == "1"
QR_POSTURE_MIN_GAIN = float(os.environ.get("ACOPF_QR_POSTURE_MIN_GAIN", "0.02"))
QR_POSTURE_CANDIDATES = int(os.environ.get("ACOPF_QR_POSTURE_CANDIDATES", "3"))
QR_POSTURE_MIN_REL_GAIN = float(os.environ.get("ACOPF_QR_POSTURE_MIN_REL_GAIN", "0.25"))


def _qr_posture_choice(env, obs, screen, ranked_threats, prefix_action, horizons, started):
    from ACOPF.ACOPF_mod import posture_mod

    hold_forecast = validate_action(env, obs, env.action_space({}), horizons=horizons)
    if not hold_forecast["ok"]:
        return None, None, None
    from ACOPF.ACOPF_mod import capacity_mod
    scale = _peak_factor(obs)
    armed = [l for l, qv in posture_mod.attack_response(env, hold_forecast["forecast_obs"]).items() if qv > 0]
    hold_fatal = capacity_mod.fatal_lines(env, hold_forecast["forecast_obs"], armed, demand_scale=scale)
    hold_risk, hold_terms = posture_mod.expected_loss(
        env, hold_forecast["forecast_obs"], screen, demand_scale=scale, fatal=hold_fatal)
    if hold_risk <= QR_POSTURE_MIN_GAIN:
        return None, None, None
    candidates = []
    for threat in ranked_threats:
        if time.perf_counter() - started > TEMPLATE_STAGE_BUDGET_S:
            break
        found = protected_contingency_bridge_action(
            env, obs, threat, prefix_action=prefix_action, return_all=True
        )
        if not isinstance(found, list):
            continue
        for action, label, verdict in found[:QR_POSTURE_CANDIDATES]:
            candidates.append((action, label, verdict))
    best = None
    for action, label, verdict in candidates:
        if time.perf_counter() - started > TEMPLATE_STAGE_BUDGET_S:
            break
        forecast = verdict.get("forecast_obs") if isinstance(verdict, dict) else None
        if forecast is None:
            continue
        cand_screen = screen_dynamic_contingencies(
            env, obs, action, rho_limit=N1_RHO_LIMIT, time_step=max(horizons)
        )
        cand_armed = [l for l, qv in posture_mod.attack_response(env, forecast).items() if qv > 0]
        cand_fatal = capacity_mod.fatal_lines(env, forecast, cand_armed, demand_scale=scale)
        risk, terms = posture_mod.expected_loss(env, forecast, cand_screen, demand_scale=scale, fatal=cand_fatal)
        if best is None or _key_better((risk,), (best[0],)):
            best = (risk, action, label, verdict, terms)
    gain_needed = QR_POSTURE_MIN_GAIN
    if any(t[1] >= 1.0 for t in hold_terms.values()):
        gain_needed = min(QR_POSTURE_MIN_GAIN, QR_POSTURE_MIN_REL_GAIN * hold_risk)
    if best is None or hold_risk - best[0] < gain_needed:
        if best is not None:
            print(f"[posture] step {int(obs.current_step)} hold risk {hold_risk:.3f} ({posture_mod.format_terms(hold_terms)}); best split {best[2]} risk {best[0]:.3f} -- not worth it", flush=True)
        return None, None, None
    print(f"[posture] step {int(obs.current_step)} hold risk {hold_risk:.3f} ({posture_mod.format_terms(hold_terms)}) -> {best[2]} risk {best[0]:.3f} ({posture_mod.format_terms(best[4])})", flush=True)
    return best[1], best[2], best[3]


def _preventive_split(env, obs, screen, managed_records, horizons, only_lines=None, force=False, overload_rho=None,
                      outage_group=None):
    if screen is None:
        return None
    if not force and not ENABLE_PREVENTIVE_SEARCH and not PREVENTIVE_MAINTENANCE_ONLY:
        return None
    if len(managed_records) >= PREVENTIVE_MAX_MANAGED:
        return None
    maintenance_lines = set(map(int, dynamic_security_set(env, obs)["maintenance"]))
    if not ENABLE_PREVENTIVE_SEARCH and not force:
        if not maintenance_lines:
            return None
        only_lines = maintenance_lines if only_lines is None else (set(only_lines) & maintenance_lines)
        if not only_lines:
            return None
    if maximum_rho(obs) > 0.95 or bool(dynamic_security_set(env, obs)["protection"]):
        return None
    status = np.asarray(obs.line_status, dtype=bool)
    bridges = set(map(int, dynamic_security_set(env, obs)["bridges"]))

    def severity(item):
        rho = float(item.get("rho", 0.0))
        return (0, 0.0) if not np.isfinite(rho) else (1, -rho)

    threats = [
        item
        for item in screen["results"]
        if item.get("line") is not None
        and bool(status[int(item["line"])])
        and int(item["line"]) not in bridges
        and (
            not np.isfinite(float(item.get("rho", 0.0)))
            or float(item.get("rho", 0.0)) > (PREVENTIVE_OVERLOAD_RHO if overload_rho is None else float(overload_rho))
        )
        and (only_lines is None or int(item["line"]) in set(only_lines))
    ]
    threats.sort(key=severity)
    started = time.perf_counter()
    for threat in threats[:3]:
        remaining = PREVENTIVE_SEARCH_BUDGET_S - (time.perf_counter() - started)
        if remaining < 1.0:
            break
        outage = int(threat["line"])
        group = tuple(sorted({outage} | {int(l) for l in (outage_group or ()) if bool(status[int(l)])}))
        action, label, meta = preventive_topology_action(
            env, obs, group if len(group) > 1 else outage, horizons=horizons, budget_s=min(20.0, remaining)
        )
        if action is None and ENABLE_COOPT and len(group) == 1 and meta.get("reason") == "no_full_security":
            remaining2 = PREVENTIVE_SEARCH_BUDGET_S - (time.perf_counter() - started)
            if remaining2 > 3.0:
                action, label, meta2 = cooptimised_preventive_action(
                    env, obs, outage, horizons, budget_s=min(12.0, remaining2)
                )
                print(
                    f"[coopt] step {int(obs.current_step)} line {outage}: "
                    f"{'applying ' + repr(label) if action is not None else 'nothing'} {meta2}",
                    flush=True,
                )
                if action is not None:
                    meta = meta2
        if action is None:
            if meta.get("reason") != "recently_searched_without_result":
                print(
                    f"[preventive] step {int(obs.current_step)} line {outage} "
                    f"rho {float(threat.get('rho', float('nan'))):.3f}: {meta.get('reason')} "
                    f"best={meta.get('best_key')} {meta.get('elapsed_s')}s",
                    flush=True,
                )
            continue
        candidate_screen = screen_dynamic_contingencies(
            env, obs, action, rho_limit=N1_RHO_LIMIT, time_step=max(horizons)
        )
        def _without_target_rho(screen_like):
            results = []
            for item in screen_like["results"]:
                if int(item["line"]) in group and np.isfinite(float(item["rho"])):
                    item = dict(item)
                    item["ok"] = True
                    item["bridge_rho"] = 0.0
                results.append(item)
            return {"passed": screen_like["passed"], "worst_rho": screen_like["worst_rho"], "results": results}

        candidate_rest = _without_target_rho(candidate_screen)
        baseline_rest = _without_target_rho(screen)
        tolerant = _tolerate_low_risk_bridges(env, obs, action, candidate_rest)
        if not contingency_not_worse(tolerant, baseline_rest, rho_tolerance=0.05):
            print(
                f"[preventive] step {int(obs.current_step)} line {outage}: '{label}' "
                f"vetoed by N-1 not-worse: {_not_worse_reasons(tolerant, baseline_rest, 0.05)}",
                flush=True,
            )
            _memo_veto(obs, outage)
            continue
        print(f"[preventive] step {int(obs.current_step)} line {outage}: applying '{label}' {meta.get('best_key')}", flush=True)
        replace_index, replaced_record = _record_touched_by(
            env, action, managed_records
        )
        record = _managed_record(
            env, obs, action, label, int(obs.current_step), replaced_record
        )
        record["contingency_line"] = outage
        return {
            "action": action,
            "dictionary": {},
            "decision": f"preventive_split:{label}",
            "acopf_status": "preventive_topology_search",
            "record": record,
            "replace_index": replace_index,
        }
    return None


def solve_current_step(
    env,
    obs,
    managed_records,
    opponent_blocked=(),
    preventive_screen=None,
    attack_lines=None,
    contingency_terms=None,
    curtailment_weight_override=None,
):
    horizons = validation_horizons(env, obs)
    if preventive_screen is not None:
        _preventive_started = time.perf_counter()
        preventive_result = _preventive_split(
            env, obs, preventive_screen, managed_records, horizons
        )
        if time.perf_counter() - _preventive_started > SLOW_PHASE_REPORT_S:
            print(
                f"[timing] step {int(obs.current_step)} preventive_split="
                f"{time.perf_counter() - _preventive_started:.1f}s",
                flush=True,
            )
        if preventive_result is not None:
            return preventive_result
    data = extract_grid2op_data_mod(env, obs)
    _connected, current_bridges = topology_bridges(env, obs)
    induced_bridges = outage_induced_bridges(env, obs)
    current_security = dynamic_security_set(env, obs)
    contingency_threats = [] if preventive_screen is None else [
        item
        for item in preventive_screen["results"]
        if (
            np.isfinite(float(item.get("bridge_rho", float("inf"))))
            and float(item.get("bridge_rho", 0.0)) > 0.95
        )
        or (
            (bool(managed_records) or (POSTURE_THREATS and int(item["line"]) in _alertable_set(env)))
            and
            np.isfinite(float(item.get("rho", float("inf"))))
            and float(item.get("rho", 0.0)) > 1.15
            and item.get("max_rho_line") is not None
        )
    ]
    _template_started = time.perf_counter()
    if contingency_threats and pace_mod._STATE["mode"] == "minimal":
        contingency_threats = [
            item for item in contingency_threats
            if not np.isfinite(float(item.get("rho", 0.0)))
        ]
    if contingency_threats:
        ranked_bridge_contingencies = sorted(
            contingency_threats,
            key=lambda item: max(
                float(item.get("bridge_rho", 0.0)) / 0.95,
                float(item.get("rho", 0.0)),
            ),
            reverse=True,
        )[:TEMPLATE_MAX_THREATS]
        top_line = (
            int(ranked_bridge_contingencies[0]["line"])
            if ranked_bridge_contingencies else None
        )
        guarded = {
            int(record["contingency_line"]): index
            for index, record in enumerate(managed_records)
            if record.get("contingency_line") is not None
        }
        replace_index = None
        replaced_record = None
        prefix_action = None
        if top_line is not None and top_line in guarded and cooldown_cleared(
            obs, managed_records[guarded[top_line]]
        ):
            replace_index = guarded[top_line]
            replaced_record = managed_records[replace_index]
            prefix_action = build_restore_action(env, replaced_record, obs=obs)
        elif len(managed_records) < TEMPLATE_MAX_RECORDS:
            pass
        elif managed_records:
            for index, record in enumerate(managed_records):
                if cooldown_cleared(obs, record):
                    replace_index = index
                    replaced_record = record
                    prefix_action = build_restore_action(env, record, obs=obs)
                    break
        if len(managed_records) < TEMPLATE_MAX_RECORDS or prefix_action is not None:
            preventive_action = preventive_label = preventive_verdict = None
            if QR_POSTURE:
                preventive_action, preventive_label, preventive_verdict = _qr_posture_choice(
                    env, obs, preventive_screen, ranked_bridge_contingencies,
                    prefix_action, horizons, _template_started,
                )
            else:
                for bridge_contingency in ranked_bridge_contingencies:
                    if time.perf_counter() - _template_started > TEMPLATE_STAGE_BUDGET_S:
                        break
                    preventive_action, preventive_label, preventive_verdict = (
                        protected_contingency_bridge_action(
                            env,
                            obs,
                            bridge_contingency,
                            prefix_action=prefix_action,
                        )
                    )
                    if preventive_action is not None:
                        break
        else:
            preventive_action, preventive_label, preventive_verdict = (
                None,
                None,
                None,
            )
        if preventive_action is not None:
            template_record = _managed_record(
                env,
                obs,
                preventive_action,
                preventive_label,
                int(obs.current_step),
                replaced_record,
            )
            try:
                template_record["contingency_line"] = int(preventive_label.split()[1])
            except (IndexError, ValueError):
                pass
            return {
                "action": preventive_action,
                "dictionary": {},
                "decision": f"preventive_bridge:{preventive_label}",
                "acopf_status": "contingency_bridge_protection",
                "record": template_record,
                "replace_index": replace_index,
            }
        for contingency_threat in ranked_bridge_contingencies:
            if time.perf_counter() - _template_started > TEMPLATE_STAGE_BUDGET_S:
                break
            contingency_action, contingency_dictionary, contingency_verdict = (
                _contingency_acopf_action(env, obs, contingency_threat)
            )
            if contingency_action is not None:
                contingency_dictionary["_contingency_acopf"] = (
                    contingency_verdict
                )
                return {
                    "action": contingency_action,
                    "dictionary": contingency_dictionary,
                    "decision": (
                        "preventive_contingency_acopf"
                        f"{'_best_effort' if contingency_verdict['best_effort'] else ''}:"
                        f"outage {contingency_verdict['outage']}"
                    ),
                    "acopf_status": contingency_verdict["status"],
                    "record": None,
                    "replace_index": None,
                }
    if time.perf_counter() - _template_started > SLOW_PHASE_REPORT_S:
        print(
            f"[timing] step {int(obs.current_step)} template_bridge_stage="
            f"{time.perf_counter() - _template_started:.1f}s threats={len(contingency_threats)}",
            flush=True,
        )
    live_rho = np.asarray(obs.rho, dtype=float)
    stressed_bridges = {
        int(line)
        for line in induced_bridges
        if bool(obs.line_status[int(line)])
        and np.isfinite(live_rho[int(line)])
        and live_rho[int(line)] >= CRITICAL_BRIDGE_ACTIVATION_RHO
    }
    hold_forecast = validate_action(
        env,
        obs,
        env.action_space({}),
        require_imminent_reset=False,
        horizons=horizons,
    )
    forecast_stressed_bridges = set()
    if hold_forecast["ok"]:
        forecast_obs = hold_forecast["forecast_obs"]
        forecast_rho = np.asarray(forecast_obs.rho, dtype=float)
        forecast_stressed_bridges = {
            int(line)
            for line in current_bridges
            if bool(forecast_obs.line_status[int(line)])
            and np.isfinite(forecast_rho[int(line)])
            and forecast_rho[int(line)] >= FORECAST_BRIDGE_ACTIVATION_RHO
        }
    priority_bridge_lines = (
        stressed_bridges
        | (set(current_bridges) & current_security["protection"])
        | forecast_stressed_bridges
    )
    critical_bridge_emergency = bool(
        priority_bridge_lines
    )
    bridge_topology_urgent = bool(
        set(current_bridges) & current_security["protection"]
    ) or any(
        int(line) in forecast_stressed_bridges
        and hold_forecast["ok"]
        and float(hold_forecast["forecast_obs"].rho[int(line)]) >= 1.0
        for line in priority_bridge_lines
    )
    if (
        bridge_topology_urgent
        and bool(set(current_bridges) & current_security["protection"])
    ):
        storage_action, storage_verdict = select_bridge_storage_relief(
            env, obs, priority_bridge_lines
        )
        if storage_action is not None:
            return {
                "action": storage_action,
                "dictionary": storage_action.as_dict(),
                "decision": "bridge_storage_relief",
                "acopf_status": "bridge_protection",
                "record": None,
                "replace_index": None,
            }
    if (
        ENABLE_ENDPOINT_BRIDGE_TOPOLOGY
        and
        bool(set(current_bridges) & current_security["protection"])
        and not managed_records
    ):
        bridge_action, bridge_label, bridge_verdict = (
            protected_bridge_relief_action(
                env, obs, priority_bridge_lines
            )
        )
        if bridge_action is not None:
            return {
                "action": bridge_action,
                "dictionary": {},
                "decision": f"bridge_relief:{bridge_label}",
                "acopf_status": "bridge_protection",
                "record": _managed_record(
                    env,
                    obs,
                    bridge_action,
                    bridge_label,
                    int(obs.current_step),
                ),
                "replace_index": None,
            }
    solve_rho_target = (
        CRITICAL_BRIDGE_RHO_TARGET
        if critical_bridge_emergency
        else (min(RHO_TARGET, OUTAGE_RHO_TARGET) if _OUTAGE_POSTURE["active"] else RHO_TARGET)
    )
    recovery = _degraded_recovery(env, obs, managed_records, opponent_blocked, horizons)
    if recovery is not None:
        return recovery

    _acopf_started = time.perf_counter()
    req_up, _avail_up, req_down, _avail_down = _balancing_requirement(env, obs, env.action_space({}))
    pv_up, pv_down = _preview_need()
    if _FLEET_MOVE.get("forecast_rejected"):
        pv_up = pv_down = 0.0
    preview_up = max(0.0, pv_up - max(0.0, req_up))
    preview_down = max(0.0, pv_down - max(0.0, req_down))
    model, results = acopf_solve_cached(
        data,
        rho_target=solve_rho_target,
        forecast_up_requirement=(
            EMERGENCY_RESERVE_FLOOR * _renewable_error_quantile(EMERGENCY_RESERVE_QUANTILE)
            if critical_bridge_emergency
            else max(data.forecast_up_requirement_mw, req_up if RESERVE_WATCH else 0.0)
        ),
        forecast_down_requirement=(
            EMERGENCY_RESERVE_FLOOR * _renewable_error_quantile(EMERGENCY_RESERVE_QUANTILE)
            if critical_bridge_emergency
            else max(data.forecast_down_requirement_mw, req_down if RESERVE_WATCH else 0.0)
        ),
        reserve_margin_mw=(
            0.0 if critical_bridge_emergency else RESERVE_MARGIN_MW
        ),
        balancing_cap_mw=_balancing_cap_mw(env, obs),
        attack_lines=attack_lines,
        contingency_terms=contingency_terms,
        preview_up=preview_up, preview_down=preview_down,
        curtailment_weight=(
            float(curtailment_weight_override) if curtailment_weight_override is not None
            else CURTAILMENT_WEIGHT_EMERGENCY
            if (maximum_rho(obs) >= 1.0 or bool(current_security["protection"]))
            else CURTAILMENT_WEIGHT_PREVENTIVE
        ),
        storage_weight=(
            STORAGE_WEIGHT_EMERGENCY
            if (maximum_rho(obs) >= 1.0 or bool(current_security["protection"]))
            else STORAGE_WEIGHT_PREVENTIVE
        ),
        tee=SOLVER_TEE,
    )
    status = results.solver.termination_condition
    emergency_no_reserve = critical_bridge_emergency

    if status not in ACCEPTABLE:
        floor = EMERGENCY_RESERVE_FLOOR * _renewable_error_quantile(EMERGENCY_RESERVE_QUANTILE)
        model, results = acopf_solve_cached(
            data,
            rho_target=solve_rho_target,
            forecast_up_requirement=floor,
            forecast_down_requirement=floor,
            reserve_margin_mw=0.0,
            balancing_cap_mw=_balancing_cap_mw(env, obs),
            tee=SOLVER_TEE,
        )
        status = results.solver.termination_condition
        emergency_no_reserve = True

    candidate_action = None
    candidate_dictionary = {}
    diagnostics = None
    direct_reason = str(status)
    if status in ACCEPTABLE:
        diagnostics = twostage_diagnostics_mod(model)
        candidate_action, candidate_dictionary = decode_action_twostage_mod(
            env, obs, model
        )
        guarded = _balancing_guard(env, obs, candidate_action, "acopf")
        if guarded is None:
            status = pyo.TerminationCondition.infeasible
            direct_reason = "balancing_infeasible"
            candidate_action = None
        elif guarded is not candidate_action:
            candidate_action = guarded
            candidate_dictionary = guarded.as_dict()
    if status in ACCEPTABLE:
        predicted_rho = diagnostics.rho_target + diagnostics.worst_slack
        model_best_effort = predicted_rho > 1.0 + 1e-5
        verdict = validate_action(
            env,
            obs,
            candidate_action,
            rho_cap=None,
            require_imminent_reset=True,
            require_critical_improvement=model_best_effort,
            horizons=horizons,
        )
        best_effort = model_best_effort or (
            verdict["ok"]
            and max(item["rho"] for item in verdict["reports"]) > 1.0 + 1e-5
        )
        if verdict["ok"] and best_effort:
            base = validate_action(
                env,
                obs,
                env.action_space({}),
                require_imminent_reset=False,
                horizons=horizons,
            )
            candidate_next = verdict["reports"][-1]["rho"]
            base_next = (
                base["reports"][-1]["rho"] if base["ok"] else float("inf")
            )
            if candidate_next > base_next + BEST_EFFORT_RHO_TOLERANCE:
                verdict["ok"] = False
                direct_reason = "best_effort_materially_worse_than_hold"
        if verdict["ok"]:
            secure, security = _security_accepts(env, obs, candidate_action)
            if secure:
                candidate_dictionary["_twostage_mod"] = {
                    "stage1_status": str(diagnostics.stage1_status),
                    "stage2_status": str(diagnostics.stage2_status),
                    "predicted_rho": predicted_rho,
                    "validated_rho_t0": verdict["reports"][0]["rho"],
                    "validated_rho_t1": verdict["reports"][-1]["rho"],
                    "best_effort": best_effort,
                    "reserve_up": diagnostics.reserve_up,
                    "reserve_down": diagnostics.reserve_down,
                    "n1_worst_rho": security["worst_rho"],
                }
                return {
                    "action": candidate_action,
                    "dictionary": candidate_dictionary,
                    "decision": (
                        "acopf_emergency_no_reserve_best_effort"
                        if emergency_no_reserve and best_effort
                        else "acopf_emergency_no_reserve"
                        if emergency_no_reserve
                        else "acopf_best_effort"
                        if best_effort
                        else "acopf"
                    ),
                    "acopf_status": str(status),
                    "record": None,
                    "replace_index": None,
                }
            direct_reason = "dynamic_contingency_rejected"
        elif direct_reason == str(status):
            direct_reason = verdict["reports"][-1]["reason"]

    _acopf_seconds = time.perf_counter() - _acopf_started
    if _acopf_seconds > SLOW_PHASE_REPORT_S:
        print(
            f"[timing] step {int(obs.current_step)} acopf_stage={_acopf_seconds:.1f}s "
            f"status={status} reason={direct_reason}",
            flush=True,
        )
    line = _critical_line(obs)
    security_now = dynamic_security_set(env, obs)
    topology_emergency = (
        maximum_rho(obs) >= 1.0
        or bool(security_now["overflow"])
        or bool(security_now["protection"])
    )
    if line is not None and topology_emergency:
        _phase_started = time.perf_counter()
        topology_action, label, verdict, replace_index = protected_lodf_action(
            env,
            obs,
            line,
            managed_records,
            n1_rho_limit=N1_RHO_LIMIT,
            blocked_lines=opponent_blocked,
        )
        security_counters = np.asarray(obs.timestep_overflow, dtype=int)
        emergency_warranted = (
            maximum_rho(obs) >= EMERGENCY_MIN_RHO
            or any(
                int(security_counters[int(l)]) >= 1
                for l in security_now["critical"]
            )
            or not hold_forecast["ok"]
        )
        if pace_mod._STATE["mode"] == "minimal":
            emergency_warranted = (
                (not hold_forecast["ok"])
                or bool(imminent_lines(env, obs, remaining_steps=1))
                or (MINIMAL_PARITY and (
                    any(
                        int(security_counters[int(l)]) >= 1
                        for l in security_now["critical"]
                    )
                    or maximum_rho(obs) >= EMERGENCY_MIN_RHO
                ))
            )
        _lodf_seconds = time.perf_counter() - _phase_started
        search_action = search_label = search_meta = None
        _phase_started = time.perf_counter()
        hold_doomed_now = (not hold_forecast["ok"]) or bool(
            imminent_lines(env, obs, remaining_steps=1)
        )
        counting_now = any(
            int(security_counters[int(l)]) >= 1 for l in security_now["critical"]
        )
        if (
            ENABLE_EMERGENCY_SEARCH
            and emergency_warranted
            and (
                len(managed_records) < EMERGENCY_MAX_MANAGED
                or hold_doomed_now
                or (counting_now and len(managed_records) < EMERGENCY_MAX_MANAGED + 2)
            )
        ):
            search_action, search_label, search_meta = emergency_topology_action(
                env, obs, blocked_lines=opponent_blocked, horizons=horizons,
                allow_openings=pace_mod._STATE["mode"] != "minimal" or hold_doomed_now,
                budget_s=EMERGENCY_SEARCH_BUDGET_S,
            )
        _search_seconds = time.perf_counter() - _phase_started
        if _lodf_seconds + _search_seconds > SLOW_PHASE_REPORT_S:
            print(
                f"[timing] step {int(obs.current_step)} lodf={_lodf_seconds:.1f}s "
                f"search={_search_seconds:.1f}s "
                f"search_meta={None if search_meta is None else {k: v for k, v in search_meta.items() if k != 'plans'}}",
                flush=True,
            )
        if search_action is not None and topology_action is not None:
            lodf_eval = evaluate_transition(env, obs, topology_action, horizons)
            lodf_key = (
                (lodf_eval["n_over"], lodf_eval["max_rho"])
                if lodf_eval is not None
                else (10 ** 6, float("inf"))
            )
            best_key = search_meta["best_key"]
            best_plain = (best_key[0], best_key[-1])
            if not (
                best_plain[0] < lodf_key[0]
                or (best_plain[0] == lodf_key[0] and best_plain[1] < lodf_key[1] - 0.02)
            ):
                search_action = None
        hold_eval_for_gate = evaluate_transition(env, obs, env.action_space({}), horizons)
        hold_has_overload = hold_eval_for_gate is None or hold_eval_for_gate["n_over"] > 0
        if search_action is not None and not hold_doomed_now and not hold_has_overload:
            hold_screen = screen_dynamic_contingencies(
                env, obs, env.action_space({}), rho_limit=N1_RHO_LIMIT, time_step=max(horizons)
            )
            search_screen = screen_dynamic_contingencies(
                env, obs, search_action, rho_limit=N1_RHO_LIMIT, time_step=max(horizons)
            )
            if not contingency_not_worse(
                _tolerate_low_risk_bridges(env, obs, search_action, search_screen),
                hold_screen,
                rho_tolerance=float("inf"),
            ):
                print(
                    f"[guard] step {int(obs.current_step)} emergency '{search_label}' vetoed: "
                    f"{_not_worse_reasons(search_screen, hold_screen, float('inf'))}",
                    flush=True,
                )
                search_action = None
        if search_action is not None:
            replace_index, replaced_record = _record_touched_by(
                env, search_action, managed_records
            )
            return {
                "action": search_action,
                "dictionary": {},
                "decision": f"emergency_search:{search_label}",
                "acopf_status": direct_reason,
                "record": _managed_record(
                    env,
                    obs,
                    search_action,
                    search_label,
                    int(obs.current_step),
                    replaced_record,
                ),
                "replace_index": replace_index,
            }
        if topology_action is not None:
            replaced_record = (
                managed_records[replace_index]
                if replace_index is not None
                else None
            )
            return {
                "action": topology_action,
                "dictionary": {},
                "decision": f"lodf:{label}",
                "acopf_status": direct_reason,
                "record": (
                    None
                    if label.startswith("restore_managed:")
                    else _managed_record(
                        env,
                        obs,
                        topology_action,
                        label,
                        int(obs.current_step),
                        replaced_record,
                    )
                ),
                "replace_index": replace_index,
            }

    restore_action, restore_index, restore_verdict = (
        select_emergency_topology_restoration(env, obs, managed_records)
    )
    if restore_action is not None:
        hold_eval = evaluate_transition(env, obs, env.action_space({}), horizons)
        restore_eval = evaluate_transition(env, obs, restore_action, horizons)
        if restore_eval is None or (
            hold_eval is not None
            and (restore_eval["n_over"], restore_eval["max_rho"] - 0.02)
            > (hold_eval["n_over"], hold_eval["max_rho"])
        ):
            print(
                f"[guard] step {int(obs.current_step)} emergency restoration rejected: "
                f"restore={None if restore_eval is None else (restore_eval['n_over'], round(restore_eval['max_rho'], 3))} "
                f"hold={None if hold_eval is None else (hold_eval['n_over'], round(hold_eval['max_rho'], 3))}",
                flush=True,
            )
            restore_action = None
    if restore_action is not None:
        return {
            "action": restore_action,
            "dictionary": {},
            "decision": "emergency_restore_topology",
            "acopf_status": direct_reason,
            "record": None,
            "replace_index": restore_index,
        }

    if candidate_action is not None and status in ACCEPTABLE:
        hold_eval = evaluate_transition(env, obs, env.action_space({}), horizons)
        imminent = imminent_lines(env, obs, remaining_steps=1)
        hold_doomed = hold_eval is None or (
            bool(imminent) and hold_eval["n_over"] > 0
        )
        if hold_doomed:
            candidate_eval = evaluate_transition(env, obs, candidate_action, horizons)
            hold_key = (
                (hold_eval["n_over"], hold_eval["max_rho"])
                if hold_eval is not None
                else (10 ** 6, float("inf"))
            )
            if candidate_eval is not None and (
                candidate_eval["n_over"] < hold_key[0]
                or (
                    candidate_eval["n_over"] == hold_key[0]
                    and candidate_eval["max_rho"] < hold_key[1] - 0.005
                )
            ):
                return {
                    "action": candidate_action,
                    "dictionary": candidate_dictionary,
                    "decision": f"acopf_least_bad:{direct_reason}",
                    "acopf_status": str(status),
                    "record": None,
                    "replace_index": None,
                }

    return {
        "action": env.action_space({}),
        "dictionary": {},
        "decision": f"fallback_do_nothing:{direct_reason}",
        "acopf_status": direct_reason,
        "record": None,
        "replace_index": None,
    }


STORAGE_TAPER = os.environ.get("ACOPF_STORAGE_TAPER", "1") == "1"
STORAGE_TAPER_FACTORS = (1.0, 0.8, 0.6, 0.4, 0.2)


def _storage_taper_action(env, obs, prefer_comfortable=False):
    if not STORAGE_TAPER:
        return None
    now = np.asarray(getattr(obs, "storage_power", np.zeros(0)), dtype=float)
    if now.size == 0 or float(np.abs(now).sum()) < 1.0:
        return None
    cls = type(env)
    e_max = np.asarray(cls.storage_Emax, dtype=float)
    charge = np.asarray(obs.storage_charge, dtype=float)
    p_max = np.minimum(np.asarray(cls.storage_max_p_absorb, dtype=float), np.asarray(cls.storage_max_p_prod, dtype=float))
    cap_c, cap_d = _storage_sustain_caps(env, obs)
    first_feasible = None
    for factor in STORAGE_TAPER_FACTORS:
        order = now * factor
        full = (order > 0) & (charge + order / 12.0 >= e_max * 0.98)
        empty = (order < 0) & (charge + order / 12.0 <= e_max * 0.02)
        order[full | empty] = 0.0
        order = np.clip(order, -p_max, p_max)
        if STORAGE_SUSTAIN_H > 0.0:
            order = np.clip(order, -cap_d, cap_c)
        if float(np.abs(order).sum()) < 1.0:
            continue
        try:
            action = env.action_space({"set_storage": [(int(i), float(v)) for i, v in enumerate(order) if abs(v) > 0.05]})
        except Exception:
            continue
        ok, _reason = _balancing_feasible(env, obs, action)
        if not ok:
            continue
        if prefer_comfortable and not _balancing_comfortable(env, obs, action):
            first_feasible = first_feasible if first_feasible is not None else (action, factor)
            continue
        return action, factor
    return first_feasible


def _storage_absorb_action(env, obs):
    if not STORAGE_ABSORB:
        return None
    cls = type(env)
    if int(getattr(cls, "n_storage", 0) or 0) == 0:
        return None
    now = np.asarray(getattr(obs, "storage_power", np.zeros(0)), dtype=float)
    up, avail_up, down, avail_down = _balancing_requirement(env, obs, env.action_space({}))
    cap_c, cap_d = _storage_sustain_caps(env, obs)
    if down > BALANCING_HEADROOM_FRACTION * avail_down:
        deficit = down - BALANCING_HEADROOM_FRACTION * avail_down + 2.0
        direction, caps = 1.0, cap_c
    elif up > BALANCING_HEADROOM_FRACTION * avail_up:
        deficit = up - BALANCING_HEADROOM_FRACTION * avail_up + 2.0
        direction, caps = -1.0, cap_d
    else:
        return None
    base = np.clip(now, -cap_d, cap_c)
    headroom = np.maximum(caps - direction * base, 0.0)
    if float(headroom.sum()) < 1.0:
        return None
    for boost in (1.0, 1.25, 1.6, 2.0):
        want = min(deficit * boost, float(headroom.sum()))
        order = base + direction * headroom / float(headroom.sum()) * want
        try:
            action = env.action_space({"set_storage": [(int(i), float(v)) for i, v in enumerate(order) if abs(v) > 0.05]})
        except Exception:
            continue
        ok, _reason = _balancing_feasible(env, obs, action)
        if ok:
            return action, float(direction * (order.sum() - now.sum()))
    return None


CURTAIL_FOR_BALANCE = os.environ.get("ACOPF_CURTAIL_FOR_BALANCE", "1") == "1"


def _curtail_for_balance_action(env, obs, base=None):
    if not CURTAIL_FOR_BALANCE:
        return None
    cls = type(env)
    base = base if base is not None else env.action_space({})
    up, avail_up, down, avail_down = _balancing_requirement(env, obs, base)
    if down <= BALANCING_HEADROOM_FRACTION * avail_down:
        return None
    renewable = np.asarray(cls.gen_renewable, dtype=bool)
    p_max = np.asarray(cls.gen_pmax, dtype=float)
    try:
        gen_f, _v, _l, _q = obs.get_forecasted_inj(1)
        f = np.asarray(gen_f, dtype=float)
    except Exception:
        f = np.asarray(obs.gen_p, dtype=float)
    limits = np.asarray(obs.curtailment_limit, dtype=float)
    expected = np.minimum(f, limits * p_max)
    producing = renewable & (expected > 0.5)
    total = float(expected[producing].sum())
    if total < 1.0:
        return None
    for boost in (1.0, 1.25, 1.6, 2.0):
        deficit = (down - BALANCING_HEADROOM_FRACTION * avail_down + 2.0) * boost
        deficit = min(deficit, max(0.0, BALANCING_HEADROOM_FRACTION * avail_up - 2.0))
        if deficit <= 0.0:
            return None
        scale = max(0.0, (total - deficit) / total)
        ratios = []
        for g in np.flatnonzero(producing):
            ratios.append((int(g), float(min(limits[g], max(0.0, expected[g] * scale / p_max[g])))))
        try:
            action = base + env.action_space({"curtail": ratios})
        except Exception:
            continue
        ok, _reason = _balancing_feasible(env, obs, action)
        if ok:
            return action, deficit
    return None


RELEASE_FOR_BALANCE = os.environ.get("ACOPF_RELEASE_FOR_BALANCE", "1") == "1"
RELEASE_MIN_MW = float(os.environ.get("ACOPF_RELEASE_MIN_MW", "5.0"))
RELEASE_MAX_MW = float(os.environ.get("ACOPF_RELEASE_MAX_MW", "100.0"))
RELEASE_RHO_CAP = float(os.environ.get("ACOPF_RELEASE_RHO_CAP", "0.90"))
RELEASE_HORIZON = int(os.environ.get("ACOPF_RELEASE_HORIZON", "3"))
RELEASE_DOWN_RESERVE_FRACTION = float(os.environ.get("ACOPF_RELEASE_DOWN_RESERVE_FRACTION", "0.5"))


def _release_down_room_mw(env, obs, base):
    cls = type(env)
    redispatchable = np.asarray(cls.gen_redispatchable, dtype=bool)
    ramp_down = float(np.sum(np.asarray(cls.gen_max_ramp_down, dtype=float)[redispatchable]))
    _up, _avail_up, down, avail_down = _balancing_requirement(env, obs, base)
    return float(avail_down - max(0.0, down) - RELEASE_DOWN_RESERVE_FRACTION * ramp_down)


def _release_horizons(env, obs):
    base = tuple(validation_horizons(env, obs))
    if RELEASE_HORIZON > max(base) and len(base) > 1:
        return tuple(sorted(set(base) | {RELEASE_HORIZON}))
    return base
RELEASE_ON_TOP = os.environ.get("ACOPF_RELEASE_ON_TOP", "1") == "1"
RELEASE_ON_TOP_PREFIXES = ("recover_", "neutralize_storage", "release_curtailment", "storage_", "reserve_storage", "soc_band")


def _release_on_top_eligible(action, decision):
    label = str(decision or "")
    while label.startswith("balanced:"):
        label = label[len("balanced:"):]
    if not label.startswith(RELEASE_ON_TOP_PREFIXES):
        return False
    try:
        if np.any(np.asarray(action.curtail, dtype=float) >= 0.0):
            return False
        if np.any(np.asarray(action.set_bus) != 0) or np.any(np.asarray(action.change_bus)):
            return False
        if np.any(np.asarray(action.set_line_status) != 0) or np.any(np.asarray(action.change_line_status)):
            return False
    except Exception:
        return False
    return True


def _release_for_balance_action(env, obs, base=None):
    if not RELEASE_FOR_BALANCE:
        return None
    cls = type(env)
    renewable = np.asarray(cls.gen_renewable, dtype=bool)
    p_max = np.asarray(cls.gen_pmax, dtype=float)
    limits = np.asarray(obs.curtailment_limit, dtype=float)
    curtailed = renewable & (limits < 0.999)
    if not np.any(curtailed):
        return None
    try:
        gen_f, _v, _l, _q = obs.get_forecasted_inj(1)
        f = np.asarray(gen_f, dtype=float)
    except Exception:
        f = np.asarray(obs.gen_p, dtype=float)
    expected = np.minimum(f, limits * p_max)
    gap = np.where(curtailed, np.maximum(f - expected, 0.0), 0.0)
    total_gap = float(gap.sum())
    on_top = base is not None
    base = base if base is not None else env.action_space({})
    floor = _release_floor_mw(obs)
    if floor > 0.0 and total_gap - floor < RELEASE_MIN_MW:
        return None
    cap_mw = min(RELEASE_MAX_MW, total_gap - floor) if floor > 0.0 else RELEASE_MAX_MW
    if RELEASE_DOWN_RESERVE_FRACTION > 0.0:
        room = _release_down_room_mw(env, obs, base)
        if room < RELEASE_MIN_MW:
            return None
        cap_mw = min(cap_mw, room)
    if total_gap < RELEASE_MIN_MW:
        action = base + env.action_space({"curtail": [(int(g), 1.0) for g in np.flatnonzero(curtailed)]})
        if not _balancing_feasible(env, obs, action)[0]:
            return None
        if on_top and not validate_action(env, obs, action, horizons=_release_horizons(env, obs))["ok"]:
            return None
        return action, 0.0
    hold = validate_action(env, obs, base, horizons=_release_horizons(env, obs))
    hold_rho = float(hold["reports"][0]["rho"]) if hold["ok"] and hold.get("reports") else maximum_rho(obs)
    cap_frac = min(1.0, cap_mw / total_gap)
    for frac in (1.0, 0.6, 0.35, 0.2, 0.1):
        frac = min(frac, cap_frac)
        ratios = []
        for g in np.flatnonzero(curtailed):
            target = expected[g] + frac * gap[g]
            ratios.append((int(g), float(min(1.0, max(limits[g], target / p_max[g])))))
        try:
            action = base + env.action_space({"curtail": ratios})
        except Exception:
            continue
        if not _balancing_feasible(env, obs, action)[0] or not _balancing_comfortable(env, obs, action):
            continue
        verdict = validate_action(env, obs, action, rho_cap=max(RELEASE_RHO_CAP, hold_rho + 0.01),
                                  require_imminent_reset=True, horizons=_release_horizons(env, obs))
        if verdict["ok"]:
            return action, frac * total_gap
    return None


STORAGE_CARRY = os.environ.get("ACOPF_STORAGE_CARRY", "1") == "1"


def _sets_storage(action):
    try:
        return bool(getattr(action, "_modif_storage", False))
    except Exception:
        return False


def _carry_storage(env, obs, action, label):
    if not STORAGE_CARRY:
        return action
    now = np.asarray(getattr(obs, "storage_power", np.zeros(0)), dtype=float)
    if now.size == 0 or float(np.abs(now).sum()) < 1.0:
        return action
    if np.any(np.abs(np.asarray(action.storage_p, dtype=float)) > 0.05):
        return action
    if _sets_storage(action):
        return action
    cap_c, cap_d = _storage_sustain_caps(env, obs)
    for factor in STORAGE_TAPER_FACTORS:
        order = np.clip(now * factor, -cap_d, cap_c) if STORAGE_SUSTAIN_H > 0.0 else now * factor
        if float(np.abs(order).sum()) < 1.0:
            break
        try:
            carried = action + env.action_space({"set_storage": [(int(i), float(v)) for i, v in enumerate(order) if abs(v) > 0.05]})
        except Exception:
            break
        if _balancing_feasible(env, obs, carried)[0]:
            if factor < 1.0:
                print(f"[reserve] step {int(obs.current_step)} {label}: standing storage order carried at {factor:.1f}x", flush=True)
            return carried
    return action


REDISPATCH_GUARD = os.environ.get("ACOPF_REDISPATCH_GUARD", "1") == "1"


def _impossible_redispatch(obs, act):
    try:
        _o, _r, _d, info = obs.simulate(act, time_step=1)
    except Exception:
        return False
    for e in (info.get("exception") or []):
        if "ImpossibleRedispatching" in type(e).__name__ or "ImpossibleRedispatching" in repr(e)[:200]:
            return True
    return False


def _redispatch_feasibility_guard(env, obs, result, step):
    if not REDISPATCH_GUARD:
        return result
    act = result.get("action")
    if act is None:
        return result
    sp = np.asarray(getattr(act, "storage_p", np.zeros(0)), dtype=float)
    rd = np.asarray(getattr(act, "redispatch", np.zeros(0)), dtype=float)
    if not (np.any(np.abs(sp) > 0.05) or np.any(np.abs(rd) > 1e-6)):
        return result
    if not _impossible_redispatch(obs, act):
        return result
    import copy as _copy
    tries = []
    if sp.size and np.any(np.abs(sp) > 0.05):
        a2 = _copy.deepcopy(act); a2.storage_p = np.zeros_like(sp); tries.append(("storage dropped", a2))
        if rd.size and np.any(np.abs(rd) > 1e-6):
            a3 = _copy.deepcopy(a2); a3.redispatch = np.zeros_like(rd); tries.append(("storage and redispatch dropped", a3))
    elif rd.size and np.any(np.abs(rd) > 1e-6):
        a3 = _copy.deepcopy(act); a3.redispatch = np.zeros_like(rd); tries.append(("redispatch dropped", a3))
    tries.append(("do nothing", env.action_space({})))
    for label, alt in tries:
        if not _impossible_redispatch(obs, alt):
            print(f"[guard] step {step} {str(result.get('decision'))[:40]}: impossible redispatch in simulation, {label}", flush=True)
            return dict(result, action=alt, decision=f"guard_{label.replace(' ', '_')}:{result.get('decision')}")
    print(f"[guard] step {step} {str(result.get('decision'))[:40]}: impossible redispatch in simulation, no feasible variant; kept", flush=True)
    return result


def choose_action(env, obs, step, managed_records, opponent_blocked=()):
    from ACOPF.ACOPF_mod import posture_mod as _posture_obs
    _posture_obs.note_observation(obs)
    _fast_simulate_mod.reset_memo()
    result = _choose_action(env, obs, step, managed_records, opponent_blocked)
    try:
        if result.get("action") is not None and not str(result.get("decision", "")).startswith("do_nothing"):
            carried = _carry_storage(env, obs, result["action"], str(result.get("decision", ""))[:40])
            if carried is not result["action"]:
                result = dict(result, action=carried)
            completed = _complete_balance(env, obs, result["action"], str(result.get("decision", ""))[:40])
            if completed is not result["action"]:
                result = dict(result, action=completed, decision=f"balanced:{result.get('decision')}")
    except Exception as error:
        print(f"[reserve] step {step} storage carry failed: {error!r}", flush=True)
    try:
        if RESERVE_WATCH and STORAGE_TAPER and result.get("action") is not None:
            from ACOPF.ACOPF_mod.security_mod import is_do_nothing
            act = result["action"]
            holding = is_do_nothing(act) or (
                not np.any(np.asarray(act.set_bus) != 0) and not np.any(np.asarray(act.set_line_status) != 0)
                and not np.any(np.asarray(act.redispatch) != 0) and not np.any(np.asarray(act.storage_p) != 0)
                and not _sets_storage(act))
            hold_action = env.action_space({})
            hold_feasible = holding and _balancing_feasible(env, obs, hold_action)[0]
            hold_comfortable = hold_feasible and _balancing_comfortable(env, obs, hold_action) and not _FLEET_MOVE.get("forecast_rejected")
            standing_now = np.asarray(getattr(obs, "storage_power", np.zeros(0)), dtype=float)
            standing = standing_now.size > 0 and float(np.abs(standing_now).sum()) >= 1.0
            if holding and hold_feasible and standing and not hold_comfortable:
                taper = _storage_taper_action(env, obs, prefer_comfortable=True)
                if taper is not None:
                    action, factor = taper
                    print(f"[reserve] step {step} standing storage order kept at {factor:.1f}x "
                          f"({float(np.asarray(action.storage_p).sum()):+.1f} MW): dropping it is not comfortable", flush=True)
                    result = dict(result, action=action, decision=f"storage_taper_{factor:.1f}:{result.get('decision')}")
            elif holding and hold_comfortable:
                action = None; label = result.get("decision")
                if RELEASE_FOR_BALANCE:
                    rel = _release_for_balance_action(env, obs)
                    if rel is not None:
                        action, mw = rel
                        if mw > 0:
                            print(f"[reserve] step {step} releasing ~{mw:.1f} MW of curtailed renewables", flush=True)
                        label = f"release_for_balance:{label}"
                storage_action = None
                if float(np.abs(standing_now).sum()) < 5.0:
                    pre = _storage_preposition_action(env, obs)
                    if pre is not None:
                        storage_action, mw, long = pre
                        print(f"[preview] step {step} ramp of {long:+.0f} MW expected within {PREVIEW_LONG} steps: storage pre-positioned {mw:+.1f} MW", flush=True)
                        label = f"storage_preposition:{label}"
                    elif not standing:
                        band = _soc_band_action(env, obs)
                        if band is not None:
                            storage_action, mw = band
                            label = f"soc_band:{label}"
                if storage_action is not None:
                    combined = (action + storage_action) if action is not None else storage_action
                    if action is None or _balancing_feasible(env, obs, combined)[0]:
                        action = combined
                if action is not None:
                    result = dict(result, action=action, decision=label)
            elif holding and not hold_feasible:
                absorb = _storage_absorb_action(env, obs)
                taper = None if absorb is not None else _storage_taper_action(env, obs)
                if absorb is not None:
                    action, mw = absorb
                    print(f"[reserve] step {step} holding is not balanceable; storage takes {mw:+.1f} MW "
                          f"(order now {float(np.asarray(action.storage_p).sum()):+.1f} MW)", flush=True)
                    result = dict(result, action=action, decision=f"storage_absorb:{result.get('decision')}")
                elif taper is not None:
                    action, factor = taper
                    print(f"[reserve] step {step} holding is not balanceable; storage order kept at {factor:.1f}x "
                          f"({float(np.asarray(action.storage_p).sum()):+.1f} MW)", flush=True)
                    result = dict(result, action=action, decision=f"storage_taper_{factor:.1f}:{result.get('decision')}")
                else:
                    base = None
                    now = np.asarray(getattr(obs, "storage_power", np.zeros(0)), dtype=float)
                    if now.size and float(np.abs(now).sum()) >= 1.0:
                        try:
                            base = env.action_space({"set_storage": [(int(i), float(v)) for i, v in enumerate(now) if abs(v) > 0.05]})
                        except Exception:
                            base = None
                    curt = _curtail_for_balance_action(env, obs, base)
                    if curt is None and base is not None:
                        curt = _curtail_for_balance_action(env, obs, None)
                    if curt is not None:
                        action, deficit = curt
                        print(f"[reserve] step {step} holding is not balanceable; renewables curtailed by ~{deficit:.1f} MW "
                              f"(storage kept {float(np.asarray(action.storage_p).sum()):+.1f} MW)", flush=True)
                        result = dict(result, action=action, decision=f"curtail_for_balance:{result.get('decision')}")
            elif RELEASE_ON_TOP and not holding and _release_on_top_eligible(act, result.get("decision")):
                rel = _release_for_balance_action(env, obs, base=act)
                if rel is not None:
                    action, mw = rel
                    if mw > 0:
                        print(f"[reserve] step {step} releasing ~{mw:.1f} MW of curtailed renewables on top of "
                              f"{str(result.get('decision'))[:40]}", flush=True)
                    result = dict(result, action=action, decision=f"release_for_balance:{result.get('decision')}")
    except Exception as error:
        print(f"[reserve] step {step} taper failed: {error!r}", flush=True)
    try:
        result = _redispatch_feasibility_guard(env, obs, result, step)
    except Exception as error:
        print(f"[guard] step {step} failed: {error!r}", flush=True)
    if STEP_LOG:
        try:
            status = np.asarray(obs.line_status, dtype=bool)
            b_up, b_au, b_down, b_ad = _balancing_requirement(env, obs, env.action_space({}))
            print(
                f"[step] {int(obs.current_step)} rho={maximum_rho(obs):.3f} decision={result.get('decision')} "
                f"offline={np.flatnonzero(~status).tolist()} blocked={sorted(map(int, opponent_blocked))} "
                f"ratio2h={_demand_ratio(obs, NEED_HORIZON_H):.3f} ratio8h={_peak_factor(obs):.3f} "
                f"bal=up{b_up:+.0f}/{b_au:.0f} down{b_down:+.0f}/{b_ad:.0f} fleet={(_FLEET_MOVE['est'] if _FLEET_MOVE['est'] is not None else float('nan')):+.0f} "
                f"cg={_BALANCE_DIAG.get('cg', float('nan')):+.0f} st={_BALANCE_DIAG.get('storage', float('nan')):+.0f} "
                f"m={_BALANCE_DIAG.get('margin', float('nan')):.0f}(q={(_BALANCE_DIAG.get('quant') or float('nan')):.0f}) dren={_BALANCE_DIAG.get('dren', float('nan')):+.0f} bias={_renewable_forecast_bias():+.0f} "
                f"pv={_preview_need()[0]:+.0f}/{-_preview_need()[1]:+.0f} long={(_preview_long() if _preview_long() is not None else float('nan')):+.0f}{' SPIKE' if _FLEET_MOVE.get('forecast_rejected') else ''} "
                f"disp={float(np.sum(np.abs(np.asarray(obs.actual_dispatch, dtype=float)))):.0f} "
                f"curt={float(np.sum(np.maximum(np.asarray(obs.gen_p_before_curtail, dtype=float) - np.asarray(obs.gen_p, dtype=float), 0.0)[np.asarray(type(env).gen_renewable, dtype=bool)])):.0f} "
                f"stor={float(np.sum(np.asarray(getattr(obs, 'storage_power', np.zeros(0)), dtype=float))):+.0f}",
                flush=True,
            )
        except Exception as error:
            print(f"[step] {int(obs.current_step)} decision={result.get('decision')} (audit failed: {error!r})", flush=True)
    return result


ADAPTIVE_SCREEN = os.environ.get("ACOPF_ADAPTIVE_SCREEN", "0") == "1"
SCREEN_MIN_INTERVAL = int(os.environ.get("ACOPF_SCREEN_MIN_INTERVAL", "3"))
SCREEN_RATE_FLOOR = float(os.environ.get("ACOPF_SCREEN_RATE_FLOOR", "0.004"))
SCREEN_RATE_WINDOW = int(os.environ.get("ACOPF_SCREEN_RATE_WINDOW", "6"))
SCREEN_MARGIN_OFFSET = float(os.environ.get("ACOPF_SCREEN_MARGIN_OFFSET", "0.0"))
SCREEN_REACT_DELTA = float(os.environ.get("ACOPF_SCREEN_REACT_DELTA", "0.015"))
_SCREEN_STATE = {"step": -1, "next_due": 0, "history": [], "status": None, "screens": 0, "forced": 0,
                 "react_worst": None, "react_unsolved": None, "reactions": 0, "skipped": 0}


PREVENTIVE_HAZARDS_ONLY = os.environ.get("ACOPF_PREVENTIVE_HAZARDS_ONLY", "1") == "1"
_HAZARD_STATS = {"screens_failed": 0, "reacted": 0, "skipped_no_hazard": 0}


def _preventive_hazards(env, obs):
    from ACOPF.ACOPF_mod import posture_mod
    status = np.asarray(obs.line_status, dtype=bool)
    hazards = set(_maintenance_within(obs, MAINTENANCE_RISK_HORIZON_STEPS))
    if PREVENTIVE_ATTACK_DISPATCH:
        for group in posture_mod.attackable_groups(env, obs):
            hazards.update(int(l) for l in group if status[int(l)])
    security = dynamic_security_set(env, obs)
    hazards |= {int(l) for l in security["overflow"]} | {int(l) for l in security["protection"]}
    return hazards


def _screen_failure_is_a_hazard(env, obs, screen):
    if not PREVENTIVE_HAZARDS_ONLY:
        return True
    _HAZARD_STATS["screens_failed"] += 1
    hazards = _preventive_hazards(env, obs)
    hit = any(not it["ok"] and int(it["line"]) in hazards for it in screen["results"])
    if hit:
        _HAZARD_STATS["reacted"] += 1
    else:
        _HAZARD_STATS["skipped_no_hazard"] += 1
    return hit


def _screen_reset(obs):
    step = int(obs.current_step)
    if step <= _SCREEN_STATE["step"]:
        _SCREEN_STATE.update({"next_due": 0, "history": [], "status": None, "screens": 0, "forced": 0,
                              "react_worst": None, "react_unsolved": None, "reactions": 0, "skipped": 0})
    _SCREEN_STATE["step"] = step


def _screen_summary(screen):
    finite = [float(it["rho"]) for it in screen["results"] if np.isfinite(float(it.get("rho", float("inf"))))]
    unsolved = sum(1 for it in screen["results"] if not np.isfinite(float(it.get("rho", float("inf")))))
    return (max(finite) if finite else 0.0), unsolved


def _screen_feedback(screen, step, min_interval, max_interval):
    if not ADAPTIVE_SCREEN or max_interval >= 10 ** 6:
        return
    worst, _unsolved = _screen_summary(screen)
    history = _SCREEN_STATE["history"]
    base = history[-1] if history else 0.0
    rate = SCREEN_RATE_FLOOR
    if len(history) >= 2:
        rate = max(rate, float(np.max(np.abs(np.diff(history)))))
    rate *= max(1.0, worst / max(base, 1e-3))
    margin = N1_RHO_LIMIT - worst
    interval = int(math.floor(margin / rate)) if margin > 0 else min_interval
    interval = max(min_interval, min(max_interval, interval))
    _SCREEN_STATE["next_due"] = min(_SCREEN_STATE["next_due"], step + interval)


def _reaction_due(screen, step, max_interval):
    worst, unsolved = _screen_summary(screen)
    scheduled = step % max_interval == 0
    due = scheduled or _SCREEN_STATE["react_worst"] is None
    if not due:
        due = unsolved > (_SCREEN_STATE["react_unsolved"] or 0) or worst >= _SCREEN_STATE["react_worst"] + SCREEN_REACT_DELTA
    if due:
        _SCREEN_STATE["react_worst"] = worst
        _SCREEN_STATE["react_unsolved"] = unsolved
        _SCREEN_STATE["reactions"] += 1
    else:
        _SCREEN_STATE["skipped"] += 1
    return due


def _screen_due(obs, step, hold, activation_rho, min_interval, max_interval):
    if not ADAPTIVE_SCREEN:
        return step % max_interval == 0
    if max_interval >= 10 ** 6:
        return False
    projected = float(maximum_rho(obs))
    forecast = hold.get("forecast_obs") if isinstance(hold, dict) and hold.get("ok") else None
    if forecast is not None:
        projected = max(projected, float(maximum_rho(forecast)))
    history = _SCREEN_STATE["history"]
    history.append(projected)
    del history[:-max(SCREEN_RATE_WINDOW + 1, 2)]
    status = np.asarray(obs.line_status, dtype=bool).tobytes()
    changed = _SCREEN_STATE["status"] is not None and status != _SCREEN_STATE["status"]
    _SCREEN_STATE["status"] = status
    due = step >= _SCREEN_STATE["next_due"] or changed
    if not due:
        return False
    rate = SCREEN_RATE_FLOOR
    if len(history) >= 2:
        rate = max(rate, float(np.max(np.abs(np.diff(history)))))
    margin = (activation_rho - SCREEN_MARGIN_OFFSET) - projected
    interval = int(math.floor(margin / rate)) if margin > 0 else min_interval
    interval = max(min_interval, min(max_interval, interval))
    _SCREEN_STATE["next_due"] = step + interval
    _SCREEN_STATE["screens"] += 1
    if changed:
        _SCREEN_STATE["forced"] += 1
    return True


RECOVERY_HOLD_FRACTION = float(os.environ.get("ACOPF_RECOVERY_HOLD_FRACTION", "0.6"))
RESERVE_BALANCE_FIRST = os.environ.get("ACOPF_RESERVE_BALANCE_FIRST", "1") == "1"


def _balancing_comfortable(env, obs, action):
    up, avail_up, down, avail_down = _balancing_requirement(env, obs, action)
    limit = min(1.0, RECOVERY_HOLD_FRACTION) * BALANCING_HEADROOM_FRACTION
    return up <= limit * avail_up and down <= limit * avail_down


def _balance_is_tight(env, obs, step):
    if RECOVERY_HOLD_FRACTION >= 1.0:
        return False
    try:
        up, avail_up, down, avail_down = _balancing_requirement(env, obs, env.action_space({}))
    except Exception:
        return False
    limit = RECOVERY_HOLD_FRACTION * BALANCING_HEADROOM_FRACTION
    tight = down > limit * avail_down or up > limit * avail_up
    if tight:
        print(f"[reserve] step {step} recovery held: needs {up:+.1f}/{-down:+.1f} MW against fleet ramp "
              f"{avail_up:.1f}/{avail_down:.1f} MW", flush=True)
    return tight


def _choose_action(env, obs, step, managed_records, opponent_blocked=()):
    rho = maximum_rho(obs)
    _track_renewable_forecast(env, obs)
    _track_fleet_move(env, obs)
    try:
        _track_preview(env, obs)
    except Exception as error:
        print(f"[preview] step {step} failed: {error!r}", flush=True)
    _track_spend(env, obs)
    _track_load(obs)
    try:
        from ACOPF.ACOPF_mod import posture_mod as _posture_mod
        _posture_mod.observe_opponent(env, obs)
    except Exception as error:
        print(f"[opponent] step {step} observation failed: {error!r}", flush=True)
    _disarm_reset(obs)
    _screen_reset(obs)
    do_nothing = env.action_space({})
    pace = pace_mod.mode(step)
    activation_rho = ACTIVATION_RHO
    screen_interval = PREVENTIVE_SCREEN_INTERVAL
    if pace == "economy":
        activation_rho = max(ACTIVATION_RHO, pace_mod.ECONOMY_ACTIVATION_RHO)
        screen_interval = pace_mod.ECONOMY_SCREEN_INTERVAL
    elif pace == "minimal":
        activation_rho = 1.0
        screen_interval = 10 ** 9
    _OUTAGE_POSTURE["active"] = bool(OUTAGE_POSTURE and pace != "minimal" and _planned_outage_in_progress(obs))
    if _OUTAGE_POSTURE["active"]:
        activation_rho = min(activation_rho, OUTAGE_ACTIVATION_RHO)
    horizons = validation_horizons(env, obs)
    screen_step = max(horizons)
    hold = validate_action(
        env, obs, do_nothing, require_imminent_reset=True, horizons=horizons
    )
    maint_terms = None
    if MAINT_STEP_TERMS:
        try:
            from ACOPF.ACOPF_mod import maint_context as _mctx_step
            _mctx_step.begin_step(obs)
            maint_terms = _mctx_step.step_terms(env, obs, _demand_ratio, _outage_time_observation)
        except Exception as error:
            print(f"[maint-ctx] step {step} context failed: {error!r}", flush=True)
    screen_due = _screen_due(
        obs, step, hold, activation_rho,
        1 if pace == "normal" else SCREEN_MIN_INTERVAL, screen_interval,
    )
    protection_active = bool(dynamic_security_set(env, obs)["protection"])
    maintenance_soon = bool(dynamic_security_set(env, obs)["maintenance"])
    bridges = outage_induced_bridges(env, obs)
    observed_rho = np.asarray(obs.rho, dtype=float)
    stressed_bridge = any(
        bool(obs.line_status[int(line)])
        and np.isfinite(observed_rho[int(line)])
        and observed_rho[int(line)] >= CRITICAL_BRIDGE_ACTIVATION_RHO
        for line in bridges
    )
    forecast_bridge_stress = False
    if hold["ok"]:
        forecast_obs = hold["forecast_obs"]
        forecast_rho = np.asarray(forecast_obs.rho, dtype=float)
        _connected, all_bridges = topology_bridges(env, obs)
        forecast_bridge_stress = any(
            bool(forecast_obs.line_status[int(line)])
            and np.isfinite(forecast_rho[int(line)])
            and forecast_rho[int(line)] >= FORECAST_BRIDGE_ACTIVATION_RHO
            for line in all_bridges
        )

    if pace == "minimal":
        forecast_bridge_stress = False
        if not MINIMAL_BRIDGE_ACTIVATION:
            stressed_bridge = stressed_bridge and rho >= 1.0
    reserve_ok, reserve_reason = _balancing_feasible(env, obs, do_nothing) if RESERVE_WATCH else (True, "")
    if not reserve_ok:
        print(f"[reserve] step {step} standing dispatch short: {reserve_reason}", flush=True)
        balance_only = (rho <= activation_rho and not protection_active and not stressed_bridge
                        and not forecast_bridge_stress and hold["ok"])
        if balance_only and RESERVE_BALANCE_FIRST:
            remedy = _storage_absorb_action(env, obs)
            if remedy is not None:
                action, mw = remedy
                print(f"[reserve] step {step} storage takes {mw:+.1f} MW (order now {float(np.asarray(action.storage_p).sum()):+.1f} MW)", flush=True)
                return {"action": action, "dictionary": action.as_dict(), "decision": "reserve_storage",
                        "acopf_status": "not_activated", "record": None, "replace_index": None}
            taper = _storage_taper_action(env, obs)
            if taper is not None:
                action, factor = taper
                return {"action": action, "dictionary": action.as_dict(), "decision": f"reserve_storage_taper_{factor:.1f}",
                        "acopf_status": "not_activated", "record": None, "replace_index": None}
    calm = CALM_MODE and pace != "minimal" and _calm_week(env, obs, hold, horizons)
    if rho < 1.0 and not protection_active and hold["ok"]:
        prepared = _maintenance_preparation(env, obs, horizons, managed_records)
        if prepared is not None:
            return prepared
    if (
        rho > activation_rho
        or protection_active
        or stressed_bridge
        or forecast_bridge_stress
        or not hold["ok"]
        or not reserve_ok
    ):
        activation_screen = None
        if (
            maintenance_soon
            or screen_due
            or (rho > activation_rho and not managed_records and pace == "normal")
        ):
            activation_screen = screen_dynamic_contingencies(
                env,
                obs,
                do_nothing,
                rho_limit=N1_RHO_LIMIT,
                time_step=screen_step,
            )
        return solve_current_step(
            env,
            obs,
            managed_records,
            opponent_blocked,
            preventive_screen=activation_screen,
            contingency_terms=maint_terms,
        )

    if managed_records and step % RECORD_AUDIT_INTERVAL == 0 and rho <= RECOVERY_RHO_LIMIT:
        audit = _audit_records(env, obs, managed_records, horizons)
        if audit is not None:
            return audit

    pending = _pending_contingency(env, obs, managed_records, horizons)
    if pending is not None:
        screen = screen_dynamic_contingencies(
            env, obs, do_nothing, rho_limit=N1_RHO_LIMIT, time_step=screen_step
        )
        preventive_result = _preventive_split(
            env, obs, screen, managed_records, horizons, only_lines=(pending,)
        )
        if preventive_result is not None:
            return preventive_result

    emergency_records = [
        index for index, record in enumerate(managed_records)
        if str(record.get("label", "")).startswith("emergency")
    ]
    if (
        managed_records
        and rho <= RESTORE_RHO_LIMIT
        and step % RESTORATION_CHECK_INTERVAL == 0
    ):
        action, index, verdict = select_topology_restoration(
            env, obs, managed_records, rho_cap=RESTORE_RHO_LIMIT
        )
        if index is not None:
            return {
                "action": action,
                "dictionary": {},
                "decision": "restore_topology",
                "acopf_status": "not_activated",
                "record": None,
                "replace_index": index,
            }
    if (
        emergency_records
        and rho <= EMERGENCY_RESTORE_RHO_LIMIT
        and step % 2 == 0
    ):
        subset = [managed_records[index] for index in emergency_records]
        action, index, verdict = select_topology_restoration(
            env, obs, subset,
            rho_cap=(
                min(EMERGENCY_RESTORE_RHO_LIMIT, 0.97, max(0.85, float(rho) + 0.02))
                if any(managed_records[i].get("lines") for i in emergency_records)
                else min(EMERGENCY_RESTORE_RHO_LIMIT, 0.85)
            ),
            release_under_outage=RELEASE_UNDER_OUTAGE, n1_tolerance=0.05 if RELEASE_UNDER_OUTAGE else 0.02,
        )
        if index is not None:
            return {
                "action": action,
                "dictionary": {},
                "decision": "restore_emergency_topology",
                "acopf_status": "not_activated",
                "record": None,
                "replace_index": emergency_records[index],
            }

    if (
        AREA_DISARM
        and not calm
        and not (CALM_MODE and CALM_DISARM_INTERLOCK and step <= _CALM_STATE["calm_until"])
        and rho <= 1.0
        and not protection_active
        and _hardening_due(obs, step)
    ):
        action, label, verdict = _attack_survivability_dispatch(env, obs, horizons, step)
        if label is None:
            topo = _attack_survivability_topology(env, obs, horizons, step, managed_records)
            if topo is not None:
                _hardening_done(step)
                return topo
            action, label, verdict = _area_disarm(env, obs, horizons)
        _hardening_done(step)
        if label is not None:
            return {
                "action": action,
                "dictionary": action.as_dict() if label.startswith("attack survivability") else {},
                "decision": label,
                "acopf_status": "hardening" if label.startswith("attack survivability") else "not_activated",
                "record": None,
                "replace_index": None,
            }

    if (
        (rho <= RECOVERY_RHO_LIMIT and step % RESTORATION_CHECK_INTERVAL == 0
         or ATTACK_AWARE_RESTORE and rho <= 1.0 and step % NEED_CHECK_INTERVAL == 0)
        and not np.all(np.asarray(obs.line_status, dtype=bool))
    ):
        degraded = bool(emergency_records)
        own_lines = {
            int(line)
            for record in managed_records
            for line in record.get("lines", ())
        }
        if calm:
            action, label, verdict = _calm_restoration(
                env, obs, set(map(int, opponent_blocked)) | own_lines, horizons
            )
        elif ATTACK_AWARE_RESTORE:
            action, label, verdict = _attack_aware_restoration(
                env, obs, set(map(int, opponent_blocked)) | own_lines, horizons
            )
        else:
          action, label, verdict = select_line_restoration(
              env,
              obs,
              set(map(int, opponent_blocked)) | own_lines,
              rho_cap=RECOVERY_RHO_LIMIT,
              horizons=horizons,
              minimum_outages=1 if degraded else RESTORE_MIN_OUTAGES,
              require_robustness_gain=not degraded,
              require_favourable_tradeoff=True,
          )
        if label is not None:
            return {
                "action": action,
                "dictionary": {},
                "decision": f"restore_line:{label}",
                "acopf_status": "not_activated",
                "record": None,
                "replace_index": None,
            }

    recovery_cap = min(RECOVERY_RHO_LIMIT, OUTAGE_RHO_TARGET) if _OUTAGE_POSTURE["active"] else RECOVERY_RHO_LIMIT
    if rho <= recovery_cap and not _balance_is_tight(env, obs, step) and step >= _SURVIVE_STATE["hold_until"]:
        action, label, verdict = select_continuous_recovery(
            env, obs, rho_cap=recovery_cap
        )
        if label is not None:
            return {
                "action": action,
                "dictionary": action.as_dict(),
                "decision": label,
                "acopf_status": "recovery",
                "record": None,
                "replace_index": None,
            }

    if maintenance_soon or screen_due:
        screen = screen_dynamic_contingencies(
            env,
            obs,
            do_nothing,
            rho_limit=N1_RHO_LIMIT,
            time_step=screen_step,
        )
        _screen_feedback(screen, step, 1 if pace == "normal" else SCREEN_MIN_INTERVAL, screen_interval)
        if not screen["passed"] and not (calm and CALM_NO_PREVENTIVE) and _screen_failure_is_a_hazard(env, obs, screen):
            if maintenance_soon or not ADAPTIVE_SCREEN or _reaction_due(screen, step, screen_interval):
                result = solve_current_step(
                    env,
                    obs,
                    managed_records,
                    opponent_blocked,
                    preventive_screen=screen,
                    attack_lines=_attack_posture_lines(env, obs, screen),
                    contingency_terms=((_risk_contingency_terms(env, obs, screen) or []) + (maint_terms or [])) or None,
                )
                result["decision"] = f"preventive_{result['decision']}"
                if ENABLE_SCOPF_STEP and result["decision"].startswith("preventive_fallback_do_nothing"):
                    scopf = _contingency_dispatch_for_screen(env, obs, screen, horizons)
                    if scopf is not None:
                        return scopf
                return result
        else:
            preventive_result = _preventive_split(
                env, obs, screen, managed_records, horizons
            )
            if preventive_result is not None:
                return preventive_result

    return {
        "action": do_nothing,
        "dictionary": {},
        "decision": "do_nothing",
        "acopf_status": "not_activated",
        "record": None,
        "replace_index": None,
    }
