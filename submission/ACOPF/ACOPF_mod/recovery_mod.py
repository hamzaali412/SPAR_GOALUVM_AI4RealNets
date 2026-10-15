import os
import numpy as np

from ACOPF.ACOPF_mod.interventions import build_restore_action, cooldown_cleared

from ACOPF.ACOPF_mod.security_mod import (
    contingency_not_worse,
    dynamic_security_set,
    maximum_rho,
    screen_dynamic_contingencies,
    topology_bridges,
    topology_protection_safe,
    validate_action,
)


MAINT_RECONNECT_GUARD = int(os.environ.get("ACOPF_MAINT_RECONNECT_GUARD", "12"))


def maintenance_blocks_reconnection(obs, line, guard=None):
    g = MAINT_RECONNECT_GUARD if guard is None else int(guard)
    try:
        t = int(obs.time_next_maintenance[int(line)])
    except Exception:
        return False
    return t == 0 or (0 < t <= g)

def _balanced_redispatch_recovery(env, obs, fraction=0.35, tolerance=0.05):
    cls = type(env)
    target = np.asarray(obs.target_dispatch, dtype=float)
    redispatchable = np.asarray(cls.gen_redispatchable, dtype=bool)
    command = np.zeros(cls.n_gen, dtype=float)
    command[redispatchable] = -float(fraction) * target[redispatchable]
    command = np.minimum(command, 0.999 * np.asarray(cls.gen_max_ramp_up))
    command = np.maximum(command, -0.999 * np.asarray(cls.gen_max_ramp_down))

    for _ in range(cls.n_gen + 1):
        residue = float(np.sum(command))
        if abs(residue) <= 1e-6:
            break
        if residue > 0.0:
            room = np.where(
                redispatchable,
                command + 0.999 * np.asarray(cls.gen_max_ramp_down),
                0.0,
            )
            sign = -1.0
        else:
            room = np.where(
                redispatchable,
                0.999 * np.asarray(cls.gen_max_ramp_up) - command,
                0.0,
            )
            sign = 1.0
        ids = np.flatnonzero(room > 1e-9)
        if ids.size == 0:
            break
        amount = min(abs(residue), float(np.sum(room[ids])))
        command[ids] += sign * amount * room[ids] / float(np.sum(room[ids]))

    return [
        (int(g), float(command[g]))
        for g in np.flatnonzero(redispatchable)
        if abs(command[g]) > tolerance
    ]


CURTAIL_RELEASE_STEP = float(os.environ.get("ACOPF_CURTAIL_RELEASE_STEP", "0.25"))
CURTAIL_RELEASE_MAX_RHO = float(os.environ.get("ACOPF_CURTAIL_RELEASE_MAX_RHO", "1.0"))


def continuous_recovery_candidates(env, obs):
    redispatch = _balanced_redispatch_recovery(env, obs)
    storage = [
        (int(k), 0.0)
        for k, power in enumerate(np.asarray(obs.storage_power, dtype=float))
        if abs(power) > 0.05
    ]
    renewable = np.asarray(type(env).gen_renewable, dtype=bool)
    limits = np.asarray(obs.curtailment_limit, dtype=float)
    curtail = [
        (int(g), float(min(1.0, limits[g] + CURTAIL_RELEASE_STEP)))
        for g in np.flatnonzero(renewable & (limits < 1.0 - 1e-6))
    ] if float(np.nanmax(np.asarray(obs.rho, dtype=float))) <= CURTAIL_RELEASE_MAX_RHO else []

    pieces = []
    if redispatch:
        pieces.append(("recover_redispatch", {"redispatch": redispatch}))
    if storage:
        pieces.append(("neutralize_storage", {"set_storage": storage}))
    if curtail:
        pieces.append(("release_curtailment", {"curtail": curtail}))
    combined = {}
    if redispatch:
        combined["redispatch"] = redispatch
    if storage:
        combined["set_storage"] = storage
    if curtail:
        combined["curtail"] = curtail
    if len(combined) > 1:
        pieces.insert(0, ("recover_all_controls", combined))
    return [(label, env.action_space(dictionary)) for label, dictionary in pieces]


def select_continuous_recovery(env, obs, rho_cap=0.95):
    before = float(np.sum(np.abs(obs.target_dispatch)))
    best = None
    from ACOPF.ACOPF_mod import main_try as _controller

    for label, action in continuous_recovery_candidates(env, obs):
        feasible, _why = _controller._balancing_feasible(env, obs, action)
        if not feasible:
            continue
        verdict = validate_action(
            env, obs, action, rho_cap=rho_cap, require_imminent_reset=True
        )
        if not verdict["ok"]:
            continue
        forecast = verdict["forecast_obs"]
        after = float(np.sum(np.abs(forecast.target_dispatch)))
        improvement = before - after
        if improvement <= 1e-6 and before > 0.1 and "redispatch" not in label:
            continue
        key = (improvement, -verdict["reports"][-1]["rho"])
        if best is None or key > best[0]:
            best = (key, action, label, verdict)
    if best is None:
        return env.action_space({}), None, None
    return best[1], best[2], best[3]


def select_bridge_storage_relief(env, obs, bridge_lines):
    lines = {
        int(line)
        for line in bridge_lines
        if 0 <= int(line) < type(env).n_line
        and bool(obs.line_status[int(line)])
    }
    if not lines or type(env).n_storage == 0:
        return None, None

    hold = validate_action(
        env, obs, env.action_space({}), require_imminent_reset=False
    )
    if not hold["ok"]:
        return None, None
    hold_obs = hold["forecast_obs"]
    hold_priority = max(float(hold_obs.rho[line]) for line in lines)
    hold_global = hold["reports"][-1]["rho"]

    best = None
    max_discharge = np.asarray(type(env).storage_max_p_prod, dtype=float)
    max_charge = np.asarray(type(env).storage_max_p_absorb, dtype=float)
    command_sets = []
    for storage in range(type(env).n_storage):
        for command in (-max_discharge[storage], max_charge[storage]):
            command_sets.append([(int(storage), float(command))])
    for commands in command_sets:
        action = env.action_space({"set_storage": commands})
        verdict = validate_action(
            env,
            obs,
            action,
            require_imminent_reset=True,
            require_critical_improvement=False,
        )
        if not verdict["ok"]:
            continue
        forecast = verdict["forecast_obs"]
        priority_rho = max(float(forecast.rho[line]) for line in lines)
        if priority_rho >= hold_priority - 1e-4:
            continue
        if hold_priority >= 1.0 and priority_rho >= 1.0 - 1e-6:
            continue
        global_rho = verdict["reports"][-1]["rho"]
        if global_rho > max(1.10, hold_global + 0.03):
            continue
        key = (
            priority_rho,
            global_rho,
            sum(abs(command) for _storage, command in commands),
        )
        if best is None or key < best[0]:
            best = (key, action, verdict)
    if best is None:
        return None, None
    return best[1], best[2]


OUTAGE_GUARD_MAX_AGE = 120


def outage_still_present(obs, record):
    own = set(int(l) for l in record.get("lines", ()))
    status = np.asarray(obs.line_status, dtype=bool)
    age = int(obs.current_step) - int(record.get("step", obs.current_step))
    if age > OUTAGE_GUARD_MAX_AGE:
        return False
    return any(
        not status[int(line)]
        for line in record.get("outages_at_creation", ())
        if int(line) not in own
    )


def select_topology_restoration(
    env, obs, managed_records, rho_cap=0.90, release_under_outage=False, n1_tolerance=0.02
):
    best = None
    for index, record in enumerate(managed_records):
        if not cooldown_cleared(obs, record):
            continue
        if outage_still_present(obs, record) and not release_under_outage:
            continue
        contingency_line = record.get("contingency_line")
        if contingency_line is None and "contingency" in str(
            record.get("label", "")
        ).lower():
            continue
        if contingency_line is not None:
            contingency_line = int(contingency_line)
            if not bool(obs.line_status[contingency_line]):
                continue
            restore = build_restore_action(env, record, obs=obs)
            forced = restore + env.action_space(
                {"set_line_status": [(contingency_line, -1)]}
            )
            secure = True
            for horizon in (0, 1):
                sim_obs, _reward, done, info = obs.simulate(
                    forced, time_step=horizon
                )
                rho = maximum_rho(sim_obs) if not done else float("nan")
                if (
                    done
                    or info.get("is_illegal", False)
                    or info.get("is_ambiguous", False)
                    or info.get("exception", [])
                    or not np.isfinite(rho)
                    or rho > 0.95
                ):
                    secure = False
                    break
            if not secure:
                continue
        maintenance_duration = np.asarray(
            obs.duration_next_maintenance, dtype=int
        )
        if any(
            maintenance_blocks_reconnection(obs, int(line))
            for line in record.get("lines", ())
        ):
            continue
        action = build_restore_action(env, record, obs=obs)
        verdict = validate_action(env, obs, action, rho_cap=rho_cap)
        if not verdict["ok"]:
            continue
        baseline_security = screen_dynamic_contingencies(
            env, obs, env.action_space({}), rho_limit=1.20
        )
        restored_security = screen_dynamic_contingencies(
            env, obs, action, rho_limit=1.20
        )
        if not contingency_not_worse(
            restored_security, baseline_security, rho_tolerance=n1_tolerance
        ):
            continue
        key = (0 if record.get("lines") else 1, verdict["reports"][-1]["rho"])
        if best is None or key < best[0]:
            best = (key, action, index, record, verdict)
    if best is None:
        return env.action_space({}), None, None
    return best[1], best[2], best[4]


RESTORE_DEGRADED_RHO_SLACK = float(os.environ.get("ACOPF_RESTORE_DEGRADED_RHO_SLACK", "0.02"))


def select_emergency_topology_restoration(env, obs, managed_records):
    rho_before = maximum_rho(obs)
    _connected, bridges_before = topology_bridges(env, obs)
    critical_bridges = (
        set(bridges_before) & dynamic_security_set(env, obs)["critical"]
    )
    best = None
    for index, record in enumerate(managed_records):
        if not cooldown_cleared(obs, record):
            continue
        if outage_still_present(obs, record):
            continue
        action = build_restore_action(env, record, obs=obs)
        structural_ok, _ = topology_protection_safe(env, obs, action)
        if not structural_ok:
            continue
        _connected_after, bridges_after = topology_bridges(env, obs, action)
        removed_critical_bridges = critical_bridges - set(bridges_after)
        if removed_critical_bridges:
            verdict = validate_action(
                env,
                obs,
                action,
                require_imminent_reset=False,
                require_critical_improvement=False,
            )
            if not verdict["ok"]:
                continue
            worst_after = max(float(r["rho"]) for r in verdict["reports"])
            if worst_after > max(1.0, rho_before) + RESTORE_DEGRADED_RHO_SLACK:
                continue
            key = (
                -len(removed_critical_bridges),
                verdict["reports"][-1]["rho"],
                verdict["reports"][0]["rho"],
            )
            if best is None or key < best[0]:
                best = (key, action, index, verdict)
            continue
        verdict = validate_action(
            env,
            obs,
            action,
            require_imminent_reset=True,
            require_critical_improvement=True,
        )
        if not verdict["ok"]:
            continue
        if verdict["reports"][0]["rho"] >= rho_before - 1e-5:
            continue
        key = (
            0,
            verdict["reports"][-1]["rho"],
            verdict["reports"][0]["rho"],
        )
        if best is None or key < best[0]:
            best = (key, action, index, verdict)
    if best is None:
        return None, None, None
    return best[1], best[2], best[3]


def line_restoration_candidates(env, obs, blocked_lines=()):
    cls = type(env)
    blocked_lines = set(map(int, blocked_lines))
    out = []
    for line in np.flatnonzero(~np.asarray(obs.line_status, dtype=bool)):
        line = int(line)
        if line in blocked_lines:
            continue
        if int(obs.time_before_cooldown_line[line]) > 0:
            continue
        if maintenance_blocks_reconnection(obs, line):
            continue
        out.append((
            line,
            env.action_space({
                "set_line_status": [(line, 1)],
                "set_bus": {
                    "lines_or_id": [(line, 1)],
                    "lines_ex_id": [(line, 1)],
                },
            }),
        ))
    return out


def select_line_restoration(
    env,
    obs,
    blocked_lines=(),
    rho_cap=0.95,
    horizons=(0, 1),
    minimum_relief=0.01,
    minimum_outages=3,
    bridge_tolerance=0.005,
    require_robustness_gain=True,
    require_favourable_tradeoff=True,
):
    offline = int(np.count_nonzero(~np.asarray(obs.line_status, dtype=bool)))
    if offline < int(minimum_outages):
        return None, None, None
    rho_now = maximum_rho(obs)
    best = None
    for line, action in line_restoration_candidates(env, obs, blocked_lines):
        safe, _structural = topology_protection_safe(env, obs, action)
        if not safe:
            continue
        verdict = validate_action(
            env,
            obs,
            action,
            rho_cap=rho_cap,
            require_imminent_reset=True,
            horizons=horizons,
        )
        if not verdict["ok"]:
            continue
        relief = rho_now - verdict["reports"][0]["rho"]
        if relief < minimum_relief:
            continue
        if best is None or relief > best[0]:
            best = (relief, action, line, verdict)
    if best is None:
        return None, None, None

    _relief, action, line, verdict = best
    time_step = max(horizons)

    def unsolved(screen):
        return sum(
            1 for item in screen["results"] if not np.isfinite(item["rho"])
        )

    def worst_bridge(screen):
        values = [
            float(item.get("bridge_rho", 0.0))
            for item in screen["results"]
            if np.isfinite(float(item.get("bridge_rho", float("inf"))))
        ]
        return max(values) if values else 0.0

    baseline = screen_dynamic_contingencies(
        env, obs, env.action_space({}), rho_limit=1.0, time_step=time_step
    )
    candidate = screen_dynamic_contingencies(
        env, obs, action, rho_limit=1.0, time_step=time_step
    )
    def failing(screen):
        return sum(1 for item in screen["results"] if not item["ok"])

    robustness_gain = unsolved(baseline) - unsolved(candidate)
    overload_cost = failing(candidate) - failing(baseline)
    if require_robustness_gain:
        if robustness_gain <= 0:
            return None, None, None
    elif robustness_gain < 0:
        return None, None, None
    if require_favourable_tradeoff:
        if robustness_gain <= 0 and overload_cost > 0:
            return None, None, None
        if robustness_gain > 0 and robustness_gain <= overload_cost:
            return None, None, None
    if worst_bridge(candidate) > worst_bridge(baseline) + float(bridge_tolerance):
        return None, None, None
    return action, f"reconnect line {line}", verdict
