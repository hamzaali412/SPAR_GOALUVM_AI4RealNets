
from collections import defaultdict
import itertools

import networkx as nx
import numpy as np


MAINTENANCE_LOOKAHEAD = 12

_STEP_CACHE = {"key": None, "holder": None, "data": {}}


def step_cache(obs):
    # The key alone is not enough: a freed observation's ``id`` can be reused by a new one with the
    # same step number, so the holder is kept and compared by identity.
    key = (id(obs), int(getattr(obs, "current_step", -1)))
    if _STEP_CACHE["key"] != key or _STEP_CACHE.get("holder") is not obs:
        _STEP_CACHE["key"] = key
        _STEP_CACHE["holder"] = obs
        _STEP_CACHE["data"] = {}
    return _STEP_CACHE["data"]


def is_do_nothing(action):
    try:
        return not action.can_affect_something()
    except Exception:
        return False


def _topology_multigraph(env, obs, action=None):
    cls = type(env)
    buses = np.asarray(obs.topo_vect, dtype=int).copy()
    status = np.asarray(obs.line_status, dtype=bool).copy()
    if action is not None:
        set_bus = np.asarray(action.set_bus, dtype=int)
        buses[set_bus != 0] = set_bus[set_bus != 0]
        set_status = np.asarray(action.set_line_status, dtype=int)
        status[set_status == 1] = True
        status[set_status == -1] = False

    graph = nx.MultiGraph()
    for line in range(cls.n_line):
        if not status[line]:
            continue
        pos_or = int(cls.line_or_pos_topo_vect[line])
        pos_ex = int(cls.line_ex_pos_topo_vect[line])
        bus_or, bus_ex = int(buses[pos_or]), int(buses[pos_ex])
        if bus_or <= 0 or bus_ex <= 0:
            continue
        u = (int(cls.line_or_to_subid[line]), bus_or)
        v = (int(cls.line_ex_to_subid[line]), bus_ex)
        graph.add_edge(u, v, key=int(line), line=int(line))
    return graph


def maximum_rho(obs):
    rho = np.asarray(obs.rho, dtype=float)
    finite = rho[np.isfinite(rho)]
    return float(np.max(finite)) if finite.size else float("nan")


def valid_transition(done, info, rho):
    return bool(
        not done
        and not info.get("is_illegal", False)
        and not info.get("is_ambiguous", False)
        and not info.get("exception", [])
        and np.isfinite(rho)
    )


def topology_bridges(env, obs, action=None):
    if action is None:
        cache = step_cache(obs)
        if "topology_bridges" in cache:
            return cache["topology_bridges"]
        result = _topology_bridges(env, obs, None)
        cache["topology_bridges"] = result
        return result
    return _topology_bridges(env, obs, action)


_TOPOLOGY_MEMO_MAX = 50000
import os as _os
SCREEN_CHAINED = _os.environ.get("ACOPF_SCREEN_CHAINED", "1") == "1"
_BRIDGE_MEMO = {}
_SINGLE_FAILURE_MEMO = {}


def _topology_key(env, obs, action=None):
    buses = np.asarray(obs.topo_vect, dtype=np.int32).copy()
    status = np.asarray(obs.line_status, dtype=bool).copy()
    if action is not None:
        set_bus = np.asarray(action.set_bus, dtype=int)
        buses[set_bus != 0] = set_bus[set_bus != 0]
        set_status = np.asarray(action.set_line_status, dtype=int)
        status[set_status == 1] = True
        status[set_status == -1] = False
    return (status.tobytes(), buses.tobytes()), buses, status


def _topology_bridges(env, obs, action=None):
    key, buses, status = _topology_key(env, obs, action)
    hit = _BRIDGE_MEMO.get(key)
    if hit is not None:
        return hit[0], set(hit[1])
    connected, bridges = _topology_bridges_uncached(env, buses, status)
    if len(_BRIDGE_MEMO) >= _TOPOLOGY_MEMO_MAX:
        _BRIDGE_MEMO.clear()
    _BRIDGE_MEMO[key] = (connected, frozenset(bridges))
    return connected, set(bridges)


def _topology_bridges_uncached(env, buses, status):
    cls = type(env)

    graph = nx.Graph()
    line_by_edge = defaultdict(list)
    for line in range(cls.n_line):
        if not status[line]:
            continue
        pos_or = int(cls.line_or_pos_topo_vect[line])
        pos_ex = int(cls.line_ex_pos_topo_vect[line])
        bus_or, bus_ex = int(buses[pos_or]), int(buses[pos_ex])
        if bus_or <= 0 or bus_ex <= 0:
            continue
        u = (int(cls.line_or_to_subid[line]), bus_or)
        v = (int(cls.line_ex_to_subid[line]), bus_ex)
        graph.add_edge(u, v)
        line_by_edge[frozenset((u, v))].append(int(line))

    if graph.number_of_nodes() and not nx.is_connected(graph):
        return False, set()
    bridges = {
        line_by_edge[frozenset((u, v))][0]
        for u, v in nx.bridges(graph)
        if len(line_by_edge[frozenset((u, v))]) == 1
    }
    return True, bridges


def outage_induced_bridges(env, obs):
    cache = step_cache(obs)
    if "outage_induced_bridges" in cache:
        return cache["outage_induced_bridges"]
    result = _outage_induced_bridges(env, obs)
    cache["outage_induced_bridges"] = result
    return result


def _outage_induced_bridges(env, obs):
    _connected, live_bridges = topology_bridges(env, obs)
    offline = np.flatnonzero(~np.asarray(obs.line_status, dtype=bool))
    if offline.size == 0:
        return set()
    hypothetical_restore = env.action_space(
        {"set_line_status": [(int(line), 1) for line in offline]}
    )
    _reference_connected, reference_bridges = topology_bridges(
        env, obs, hypothetical_restore
    )
    return set(live_bridges) - set(reference_bridges)


def structural_double_outage_failures(env, obs, action=None):
    graph = _topology_multigraph(env, obs, action)
    if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
        return {("base", "disconnected")}
    security = dynamic_security_set(env, obs)
    hazards = sorted(security["opponent"] | security["maintenance"])
    present = {
        int(key)
        for _u, _v, key in graph.edges(keys=True)
    }
    hazards = [line for line in hazards if line in present]
    failures = set()
    for first, second in itertools.combinations(hazards, 2):
        trial = graph.copy()
        for u, v, key in list(trial.edges(keys=True)):
            if key == first or key == second:
                trial.remove_edge(u, v, key=key)
        if trial.number_of_nodes() and not nx.is_connected(trial):
            failures.add((int(first), int(second)))
    return failures


def structural_outage_failures_for(env, obs, hazards, action=None):
    hazards = sorted(int(l) for l in hazards)
    memo_key, _buses, _status = _topology_key(env, obs, action)
    memo_key = (memo_key, tuple(hazards))
    hit = _SINGLE_FAILURE_MEMO.get(memo_key)
    if hit is not None:
        return set(hit)
    result = _structural_single_outage_failures(env, obs, action, hazards)
    if len(_SINGLE_FAILURE_MEMO) >= _TOPOLOGY_MEMO_MAX:
        _SINGLE_FAILURE_MEMO.clear()
    _SINGLE_FAILURE_MEMO[memo_key] = frozenset(result)
    return set(result)


def structural_single_outage_failures(env, obs, action=None):
    hazards = sorted(int(l) for l in dynamic_security_set(env, obs)["contingencies"])
    memo_key, _buses, _status = _topology_key(env, obs, action)
    memo_key = (memo_key, tuple(hazards))
    hit = _SINGLE_FAILURE_MEMO.get(memo_key)
    if hit is not None:
        return set(hit)
    result = _structural_single_outage_failures(env, obs, action, hazards)
    if len(_SINGLE_FAILURE_MEMO) >= _TOPOLOGY_MEMO_MAX:
        _SINGLE_FAILURE_MEMO.clear()
    _SINGLE_FAILURE_MEMO[memo_key] = frozenset(result)
    return set(result)


def _structural_single_outage_failures(env, obs, action, hazards):
    # Removing one edge from a connected multigraph disconnects it exactly when that edge is a
    # bridge with no parallel line; ``_topology_bridges`` computes that set (memoised on the
    # topology), so no per-hazard graph copy is needed.
    connected, bridges = _topology_bridges(env, obs, action)
    _key, buses, status = _topology_key(env, obs, action)
    cls = type(env)
    live = [
        line for line in np.flatnonzero(status)
        if buses[int(cls.line_or_pos_topo_vect[line])] > 0 and buses[int(cls.line_ex_pos_topo_vect[line])] > 0
    ]
    if not live or not connected:
        return {"base"}
    return {int(line) for line in hazards if int(line) in bridges}


def dynamic_security_set(env, obs):
    cache = step_cache(obs)
    if "dynamic_security_set" in cache:
        return cache["dynamic_security_set"]
    result = _dynamic_security_set(env, obs)
    cache["dynamic_security_set"] = result
    return result


def _dynamic_security_set(env, obs):
    connected = np.asarray(obs.line_status, dtype=bool)
    rho = np.asarray(obs.rho, dtype=float)
    counters = np.asarray(obs.timestep_overflow, dtype=int)
    overflow = set(np.flatnonzero(connected & (rho > 1.0)).astype(int))
    protection = set(np.flatnonzero(connected & (counters > 0)).astype(int))
    _, bridges = topology_bridges(env, obs)
    opponent = set(map(int, getattr(env, "alertable_line_ids", ())))
    time_to_maintenance = np.asarray(obs.time_next_maintenance, dtype=int)
    maintenance = set(
        np.flatnonzero(
            connected
            & (time_to_maintenance >= 0)
            & (time_to_maintenance <= MAINTENANCE_LOOKAHEAD)
        ).astype(int)
    )
    return {
        "opponent": opponent,
        "overflow": overflow,
        "protection": protection,
        "bridges": bridges,
        "maintenance": maintenance,
        "critical": overflow | protection,
        "contingencies": (
            opponent | overflow | protection | bridges | maintenance
        ),
    }


def imminent_lines(env, obs, remaining_steps=1):
    allowed = int(env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED)
    connected = np.asarray(obs.line_status, dtype=bool)
    counters = np.asarray(obs.timestep_overflow, dtype=int)
    return {
        int(line)
        for line in np.flatnonzero(connected & (counters > 0))
        if allowed - int(counters[line]) <= remaining_steps
    }


def forecast_usable(env, obs):
    try:
        sim_obs, _reward, done, info = obs.simulate(
            env.action_space({}), time_step=1
        )
    except Exception:
        return False
    rho = maximum_rho(sim_obs) if not done else float("nan")
    return valid_transition(done, info, rho)


def validation_horizons(env, obs):
    cache = step_cache(obs)
    if "validation_horizons" not in cache:
        cache["validation_horizons"] = (0, 1) if forecast_usable(env, obs) else (0,)
    return cache["validation_horizons"]


def validate_action(
    env,
    obs,
    action,
    *,
    rho_cap=None,
    require_imminent_reset=True,
    require_critical_improvement=False,
    horizons=(0, 1),
):
    memo_key = None
    if is_do_nothing(action):
        memo_key = (
            "validate_hold",
            rho_cap,
            bool(require_imminent_reset),
            bool(require_critical_improvement),
            tuple(horizons),
        )
        cache = step_cache(obs)
        if memo_key in cache:
            return cache[memo_key]
    result = _validate_action(
        env,
        obs,
        action,
        rho_cap=rho_cap,
        require_imminent_reset=require_imminent_reset,
        require_critical_improvement=require_critical_improvement,
        horizons=horizons,
    )
    if memo_key is not None:
        step_cache(obs)[memo_key] = result
    return result


def _validate_action(
    env,
    obs,
    action,
    *,
    rho_cap=None,
    require_imminent_reset=True,
    require_critical_improvement=False,
    horizons=(0, 1),
):
    reports = []
    critical = dynamic_security_set(env, obs)["critical"]
    imminent = imminent_lines(env, obs)
    last_horizon = max(horizons)
    for horizon in horizons:
        sim_obs, _reward, done, info = obs.simulate(action, time_step=horizon)
        rho = maximum_rho(sim_obs) if not done else float("nan")
        ok = valid_transition(done, info, rho)
        reason = "ok" if ok else "invalid_transition"
        if ok and rho_cap is not None and rho > rho_cap + 1e-6:
            ok, reason = False, "rho_cap"
        if ok and require_imminent_reset and horizon == last_horizon:
            for line in imminent:
                if bool(sim_obs.line_status[line]) and float(sim_obs.rho[line]) >= 1.0:
                    ok, reason = False, f"imminent_line_{line}_not_reset"
                    break
        if (
            ok
            and require_critical_improvement
            and horizon == last_horizon
            and critical
        ):
            improved = any(
                bool(sim_obs.line_status[line])
                and float(sim_obs.rho[line]) < float(obs.rho[line]) - 1e-5
                for line in critical
            )
            if not improved:
                ok, reason = False, "no_critical_improvement"
        reports.append(
            {"horizon": horizon, "ok": ok, "rho": rho, "reason": reason}
        )
        if not ok:
            return {"ok": False, "reports": reports, "forecast_obs": sim_obs}
    return {"ok": True, "reports": reports, "forecast_obs": sim_obs}


def topology_protection_safe(env, obs, action):
    security = dynamic_security_set(env, obs)
    connected, bridges = topology_bridges(env, obs, action)
    new_bridges = bridges - security["bridges"]
    unsafe = new_bridges & security["critical"]
    return connected and not unsafe, {
        "connected": connected,
        "bridges": bridges,
        "new_bridges": new_bridges,
        "critical_bridges": unsafe,
    }


def relevant_contingencies(env, obs, baseline_screen, rho_threshold=0.9):
    security = dynamic_security_set(env, obs)
    keep = set(security["bridges"]) | set(security["critical"])
    for item in baseline_screen["results"]:
        rho = float(item.get("rho", 0.0))
        bridge = float(item.get("bridge_rho", 0.0))
        if not np.isfinite(rho) or rho > rho_threshold or not item.get("ok", True):
            keep.add(int(item["line"]))
        elif np.isfinite(bridge) and bridge > 0.8:
            keep.add(int(item["line"]))
    return keep


def screen_dynamic_contingencies(
    env, obs, action, rho_limit=1.0, bridge_rho_limit=0.95, time_step=1, lines=None, extra_lines=()
):
    memo_key = None
    if is_do_nothing(action) and lines is None:
        memo_key = ("screen_hold", float(rho_limit), float(bridge_rho_limit), int(time_step), tuple(sorted(int(l) for l in extra_lines)))
        cache = step_cache(obs)
        if memo_key in cache:
            return cache[memo_key]
    result = _screen_dynamic_contingencies(
        env, obs, action, rho_limit=rho_limit, bridge_rho_limit=bridge_rho_limit,
        time_step=time_step, lines=lines, extra_lines=extra_lines,
    )
    if memo_key is not None:
        step_cache(obs)[memo_key] = result
    return result



# ---------------------------------------------------------------------------
# Fast N-1 screen via lightsim2grid ContingencyAnalysis (ACOPF_SCREEN_CA=1)
# ---------------------------------------------------------------------------
# obs.simulate costs ~9 ms per contingency, almost all of it grid2op overhead
# (action copy, redispatch solver, Observation build); the AC power flow itself
# is ~0.25 ms. ContingencyAnalysis runs every N-1 case in C++ on the backend
# that already holds the forecast state. Its result equals simulate's whenever
# simulate would not do anything *after* the power flow, so a case is taken
# from ContingencyAnalysis only when
#   * the power flow converged,
#   * no live line exceeds the hard-overflow threshold (protection would trip it
#     and re-solve inside simulate),
#   * no line already at the soft-overflow counter limit is above 1.0,
#   * the tripped line is not under cooldown (simulate would flag it illegal).
# Every other case falls back to obs.simulate, so results stay identical.
SCREEN_CA = _os.environ.get("ACOPF_SCREEN_CA", "1") == "1"
SCREEN_CA_MARGIN = float(_os.environ.get("ACOPF_SCREEN_CA_MARGIN", "0.98"))
SCREEN_CA_AUDIT = _os.environ.get("ACOPF_SCREEN_CA_AUDIT", "0") == "1"
_SCREEN_STATS = {"screens": 0, "ca_cases": 0, "island_cases": 0, "sim_cases": 0, "ca_unavailable": 0,
                 "fallback": {"cooldown": 0, "unsolved": 0, "hard_overflow": 0, "soft_overflow": 0, "state_protection": 0,
                              "bus_conflict": 0}}


def screen_stats():
    out = dict(_SCREEN_STATS)
    out["fallback"] = dict(_SCREEN_STATS["fallback"])
    return out


class _FlowObs:
    """Minimal observation-like record for one contingency solved by ContingencyAnalysis."""

    __slots__ = ("topo_vect", "line_status", "rho", "p_or")

    def __init__(self, topo_vect, line_status, rho, p_or):
        self.topo_vect = topo_vect
        self.line_status = line_status
        self.rho = rho
        self.p_or = p_or


def _simulate_for_backend(obs, action, time_step):
    """Run the forecast for real, never from the per-decision memo: the contingency analysis reads
    the grid the forecast environment is left holding, which a memoised result would not refresh."""
    try:
        from ACOPF.ACOPF_mod import fast_simulate_mod as _fsm

        return _fsm.simulate_fresh(obs, action, time_step=time_step)
    except Exception:
        return obs.simulate(action, time_step=time_step)


def _ca_backend_for(state_obs):
    """The LightSimBackend that currently holds `state_obs` (the last simulate), or None."""
    try:
        from lightsim2grid import LightSimBackend
    except ImportError:
        return None
    obs_env = getattr(state_obs, "_obs_env", None)
    backend = getattr(obs_env, "backend", None)
    if backend is None or not isinstance(backend, LightSimBackend):
        return None
    try:
        a_or = np.asarray(backend.lines_or_info()[3], dtype=float)
    except Exception:
        return None
    ref = np.asarray(state_obs.a_or, dtype=float)
    if a_or.shape != ref.shape or not np.allclose(a_or, ref, rtol=1e-4, atol=1e-2):
        return None
    return backend


def _audit_fast(env, base_obs, action, status_change, horizon, fast, step):
    """Re-simulate every ContingencyAnalysis case and print any disagreement (ACOPF_SCREEN_CA_AUDIT=1)."""
    for line, f in fast.items():
        trip = env.action_space({"set_line_status": [(int(line), -1)]})
        contingency = trip if status_change else action + trip
        sim_obs, _r, done, info = base_obs.simulate(contingency, time_step=horizon)
        rho = maximum_rho(sim_obs) if not done else float("inf")
        ok = valid_transition(done, info, rho)
        if f is None:
            if ok:
                print(f"[screen-audit] step {step} line {line}: CA unsolved/island but simulate solved rho={rho:.4f}", flush=True)
            continue
        if not ok:
            print(f"[screen-audit] step {step} line {line}: CA solved rho={maximum_rho(f):.4f} but simulate invalid "
                  f"done={done} illegal={info.get('is_illegal')} exc={info.get('exception')}", flush=True)
            continue
        d_all = float(np.max(np.abs(np.asarray(f.rho) - np.asarray(sim_obs.rho, dtype=float))))
        d_p = float(np.max(np.abs(np.asarray(f.p_or) - np.asarray(sim_obs.p_or, dtype=float))))
        sdiff = np.flatnonzero(np.asarray(f.line_status) != np.asarray(sim_obs.line_status, dtype=bool))
        _SCREEN_STATS["audit_max_drho"] = max(_SCREEN_STATS.get("audit_max_drho", 0.0), d_all)
        _SCREEN_STATS["audit_max_dp"] = max(_SCREEN_STATS.get("audit_max_dp", 0.0), d_p)
        if d_all > 1e-3 or d_p > 1e-2 or sdiff.size:
            print(f"[screen-audit] step {step} line {line}: max_rho {maximum_rho(f):.4f} vs {rho:.4f} "
                  f"max|drho|={d_all:.4f} max|dp_or|={d_p:.3f}MW status_diff={sdiff.tolist()}", flush=True)
            if _os.environ.get("ACOPF_SCREEN_CA_AUDIT_VERBOSE") == "1":
                cd = np.asarray(base_obs.time_before_cooldown_line, dtype=int)
                sub_cd = np.asarray(base_obs.time_before_cooldown_sub, dtype=int)
                cls = type(env)
                print(f"    action_do_nothing={is_do_nothing(action)} status_change={status_change} horizon={horizon} "
                      f"action.set_line_status_nonzero={np.flatnonzero(np.asarray(action.set_line_status)).tolist()} "
                      f"action.set_bus_nonzero={np.flatnonzero(np.asarray(action.set_bus)).tolist()[:8]} "
                      f"redispatch_nonzero={int(np.count_nonzero(np.asarray(action.redispatch)))} curtail_nonzero={int(np.count_nonzero(np.asarray(action.curtail) != -1))}", flush=True)
                print(f"    line {line}: base status={bool(base_obs.line_status[line])} cooldown={cd[line]} sub_cd(or,ex)=({sub_cd[cls.line_or_to_subid[line]]},{sub_cd[cls.line_ex_to_subid[line]]}) "
                      f"maint_in={int(base_obs.time_next_maintenance[line])} sim status={bool(sim_obs.line_status[line])} sim cooldown={int(sim_obs.time_before_cooldown_line[line])} "
                      f"info: illegal={info.get('is_illegal')} ambiguous={info.get('is_ambiguous')} exc={info.get('exception')} disc={np.flatnonzero(np.asarray(info.get('disc_lines', [-1])) >= 0).tolist()} "
                      f"contingency.set_line_status[line]={int(contingency.set_line_status[line])}", flush=True)


def _bus_conflict_lines(env, action, lines):
    """Lines whose end is re-bused by `action` (set_bus > 0 or change_bus).

    grid2op applies `action + set_line_status(-1)` by tripping the line and then
    reconnecting it through the bus assignment, so simulate returns the untripped
    case for these; they must go through simulate to reproduce that."""
    cls = type(env)
    set_bus = np.asarray(action.set_bus, dtype=int)
    try:
        change_bus = np.asarray(action.change_bus, dtype=bool)
    except Exception:
        change_bus = np.zeros_like(set_bus, dtype=bool)
    if not np.any(set_bus > 0) and not np.any(change_bus):
        return set()
    out = set()
    for line in lines:
        p_or, p_ex = int(cls.line_or_pos_topo_vect[line]), int(cls.line_ex_pos_topo_vect[line])
        if set_bus[p_or] > 0 or set_bus[p_ex] > 0 or change_bus[p_or] or change_bus[p_ex]:
            out.add(int(line))
    return out


def _cut_isolates_injection(env, state_obs, line):
    """True when opening `line` in the topology of `state_obs` leaves a load, generator
    or storage unit on each side of the cut. grid2op then ends the episode (lightsim2grid
    reports a non-connected grid), so simulate is guaranteed to fail the case."""
    cls = type(env)
    buses = np.asarray(state_obs.topo_vect, dtype=int)
    status = np.asarray(state_obs.line_status, dtype=bool)
    graph = nx.Graph()
    def node(sub, pos):
        b = int(buses[int(pos)])
        return (int(sub), b) if b > 0 else None
    for k in range(cls.n_line):
        if k == line or not status[k]:
            continue
        u, v = node(cls.line_or_to_subid[k], cls.line_or_pos_topo_vect[k]), node(cls.line_ex_to_subid[k], cls.line_ex_pos_topo_vect[k])
        if u is not None and v is not None:
            graph.add_edge(u, v)
    injections = set()
    for sub_ids, pos in ((cls.load_to_subid, cls.load_pos_topo_vect), (cls.gen_to_subid, cls.gen_pos_topo_vect),
                         (getattr(cls, "storage_to_subid", ()), getattr(cls, "storage_pos_topo_vect", ()))):
        for sub, p in zip(sub_ids, pos):
            n = node(sub, p)
            if n is not None:
                injections.add(n)
                graph.add_node(n)
    u, v = node(cls.line_or_to_subid[line], cls.line_or_pos_topo_vect[line]), node(cls.line_ex_to_subid[line], cls.line_ex_pos_topo_vect[line])
    if u is None or v is None:
        return False
    graph.add_node(u); graph.add_node(v)
    side_u = nx.node_connected_component(graph, u)
    if v in side_u:
        return False
    side_v = nx.node_connected_component(graph, v)
    return bool(injections & side_u) and bool(injections & side_v)


def _ca_screen(env, state_obs, rule_obs, lines):
    """Solve the N-1 cases `lines` on the state of `state_obs` with ContingencyAnalysis.

    Returns {line: _FlowObs | None}: a _FlowObs for the cases whose result is provably
    what simulate would return, None for bridge trips that island part of the grid
    (simulate would end the episode, i.e. the case is unsolved). Missing lines must be
    simulated. `rule_obs` is the observation simulate would start from (its
    cooldowns and overflow counters apply).
    """
    try:
        from lightsim2grid import ContingencyAnalysis
    except ImportError:
        return {}
    backend = _ca_backend_for(state_obs)
    if backend is None:
        return {}
    cls = type(env)
    params = getattr(getattr(state_obs, "_obs_env", None), "parameters", None) or env.parameters
    hard = float(params.HARD_OVERFLOW_THRESHOLD) * SCREEN_CA_MARGIN
    soft = float(getattr(params, "SOFT_OVERFLOW_THRESHOLD", 1.0)) * SCREEN_CA_MARGIN
    nb_allowed = int(params.NB_TIMESTEP_OVERFLOW_ALLOWED)
    no_disconnection = bool(params.NO_OVERFLOW_DISCONNECTION)
    limit = np.asarray(backend.get_thermal_limit(), dtype=float)
    base_status = np.asarray(state_obs.line_status, dtype=bool)
    base_topo = np.asarray(state_obs.topo_vect, dtype=int)
    cooldown = np.asarray(rule_obs.time_before_cooldown_line, dtype=int)
    counters = np.asarray(rule_obs.timestep_overflow, dtype=int)
    at_soft_limit = counters >= max(nb_allowed - 1, 0)
    fb = _SCREEN_STATS["fallback"]
    todo = [int(l) for l in lines if base_status[l] and cooldown[l] <= 0]
    fb["cooldown"] += sum(1 for l in lines if base_status[l] and cooldown[l] > 0)
    if not todo:
        return {}
    ca = None
    try:
        ca = ContingencyAnalysis(backend)
        ca.add_multiple_contingencies(*todo)
        p_or, a_or, v = ca.get_flows()
        order = [int(c[0]) for c in ca.all_contingencies]
    except Exception:
        return {}
    finally:
        if ca is not None:
            try:
                ca.close()
            except Exception:
                pass
    _connected, bridges = _topology_bridges(env, state_obs, None)
    out = {}
    for k, line in enumerate(order):
        vk, ak = v[k], np.asarray(a_or[k], dtype=float)
        if not np.any(vk) or not np.all(np.isfinite(ak)):
            # Opening a bridge that leaves injections on both sides islands part of
            # the grid; grid2op ends the episode for those, so the case is unsolved
            # without simulating it. A cut that only strands an empty bus can still
            # be solved by simulate (lightsim2grid drops the bus), so it falls back.
            if line in bridges and _cut_isolates_injection(env, state_obs, line):
                out[line] = None
            else:
                fb["unsolved"] += 1
            continue
        live = base_status.copy()
        live[line] = False
        rho = np.where(live, ak / limit, 0.0)
        if not no_disconnection:
            if np.any(rho > hard):
                fb["hard_overflow"] += 1
                continue
            if np.any(at_soft_limit & (rho > soft)):
                fb["soft_overflow"] += 1
                continue
        topo = base_topo.copy()
        topo[int(cls.line_or_pos_topo_vect[line])] = -1
        topo[int(cls.line_ex_pos_topo_vect[line])] = -1
        out[line] = _FlowObs(topo, live, rho, np.asarray(p_or[k], dtype=float).copy())
    return out


def _screen_dynamic_contingencies(
    env, obs, action, rho_limit=1.0, bridge_rho_limit=0.95, time_step=1, lines=None, extra_lines=()
):
    security = dynamic_security_set(env, obs)
    results = []
    worst = 0.0
    passed = True
    status = np.asarray(obs.line_status, dtype=bool)
    contingencies = security["contingencies"] if lines is None else (
        set(security["contingencies"]) & {int(l) for l in lines}
    )
    if extra_lines:
        contingencies = set(contingencies) | {int(l) for l in extra_lines}
    status_change = SCREEN_CHAINED and bool(np.any(np.asarray(action.set_line_status, dtype=int) != 0))
    base_obs = obs
    contingency_horizon = time_step
    if status_change:
        first_obs, _r0, done0, info0 = obs.simulate(action, time_step=time_step)
        if not valid_transition(done0, info0, maximum_rho(first_obs) if not done0 else float("nan")):
            return {"passed": False, "worst_rho": float("inf"), "results": [
                {"line": int(l), "ok": False, "rho": float("inf"), "max_rho_line": None,
                 "bridge_line": None, "bridge_rho": float("inf")}
                for l in sorted(contingencies) if 0 <= l < type(env).n_line and status[l]
            ]}
        base_obs = first_obs
        contingency_horizon = 1
    to_screen = [int(l) for l in sorted(contingencies) if 0 <= l < type(env).n_line and status[l]]
    fast = {}
    if SCREEN_CA and to_screen:
        # State whose N-1 cases we need: the forecast after `action` at the horizon
        # simulate would use for the contingencies (one extra simulate, then C++).
        if status_change:
            state_obs, _rs, done_s, info_s = _simulate_for_backend(base_obs, env.action_space({}), 1)
        else:
            state_obs, _rs, done_s, info_s = _simulate_for_backend(obs, action, time_step)
        state_ok = valid_transition(done_s, info_s, maximum_rho(state_obs) if not done_s else float("nan"))
        # Protection may have tripped a line inside that simulate (its flows depend on
        # the hold case, not on the contingency): the state is then not the one the
        # contingency simulates would see, so every case goes through simulate.
        protection_fired = bool(np.any(np.asarray(info_s.get("disc_lines", -1)) >= 0))
        conflict = set() if status_change else _bus_conflict_lines(env, action, to_screen)
        _SCREEN_STATS["fallback"]["bus_conflict"] += len(conflict)
        if state_ok and not protection_fired:
            fast = _ca_screen(env, state_obs, base_obs, [l for l in to_screen if l not in conflict])
        elif state_ok:
            _SCREEN_STATS["fallback"]["state_protection"] += len(to_screen)
        else:
            _SCREEN_STATS["ca_unavailable"] += 1
        _SCREEN_STATS["screens"] += 1
        if SCREEN_CA_AUDIT and fast:
            _audit_fast(env, base_obs, action, status_change, contingency_horizon, fast, int(getattr(obs, "current_step", -1)))
        _SCREEN_STATS["island_cases"] += sum(1 for f in fast.values() if f is None)
        _SCREEN_STATS["ca_cases"] += sum(1 for f in fast.values() if f is not None)
        _SCREEN_STATS["sim_cases"] += len(to_screen) - len(fast)
    for line in to_screen:
        if line in fast and fast[line] is None:
            sim_obs, rho, transition_ok = None, float("inf"), False
        elif line in fast:
            sim_obs = fast[line]
            rho = maximum_rho(sim_obs)
            transition_ok = np.isfinite(rho)
        else:
            if status_change:
                contingency = env.action_space({"set_line_status": [(int(line), -1)]})
            else:
                contingency = action + env.action_space(
                    {"set_line_status": [(int(line), -1)]}
                )
            sim_obs, _reward, done, info = base_obs.simulate(
                contingency, time_step=contingency_horizon
            )
            rho = maximum_rho(sim_obs) if not done else float("inf")
            transition_ok = valid_transition(done, info, rho)
        bridge_rho = float("inf")
        bridge_line = None
        max_rho_line = None
        if transition_ok:
            live_rho = np.asarray(sim_obs.rho, dtype=float)
            live_status = np.asarray(sim_obs.line_status, dtype=bool)
            finite_live = np.flatnonzero(live_status & np.isfinite(live_rho))
            if finite_live.size:
                max_rho_line = int(
                    finite_live[np.argmax(live_rho[finite_live])]
                )
            _connected, bridges = _topology_bridges(env, sim_obs, None)
            live = [
                int(item)
                for item in bridges
                if bool(sim_obs.line_status[int(item)])
                and np.isfinite(float(sim_obs.rho[int(item)]))
            ]
            if live:
                bridge_line = max(live, key=lambda item: float(sim_obs.rho[item]))
                bridge_rho = float(sim_obs.rho[bridge_line])
            else:
                bridge_rho = 0.0
        ok = (
            transition_ok
            and rho <= rho_limit + 1e-6
            and bridge_rho <= bridge_rho_limit + 1e-6
        )
        results.append(
            {
                "line": int(line),
                "ok": ok,
                "rho": rho,
                "max_rho_line": max_rho_line,
                "bridge_line": bridge_line,
                "bridge_rho": bridge_rho,
                "p_or": np.asarray(sim_obs.p_or, dtype=float).copy() if transition_ok else None,
                "rho_all": np.asarray(sim_obs.rho, dtype=float).copy() if transition_ok else None,
            }
        )
        passed &= ok
        worst = max(worst, rho)
    return {"passed": bool(passed), "worst_rho": worst, "results": results}


def contingency_not_worse(candidate, baseline, rho_tolerance=0.02):
    base_by_line = {item["line"]: item for item in baseline["results"]}
    for item in candidate["results"]:
        previous = base_by_line.get(item["line"])
        if previous is None:
            if not item["ok"]:
                return False
            continue
        if np.isfinite(previous["rho"]) and not np.isfinite(item["rho"]):
            return False
        if (
            np.isfinite(previous["rho"])
            and np.isfinite(item["rho"])
            and item["rho"] > previous["rho"] + rho_tolerance
        ):
            return False
        previous_bridge = float(previous.get("bridge_rho", 0.0))
        candidate_bridge = float(item.get("bridge_rho", 0.0))
        if (
            np.isfinite(previous_bridge)
            and np.isfinite(candidate_bridge)
            and candidate_bridge > previous_bridge + rho_tolerance
        ):
            return False
        if previous_bridge <= 0.95 and candidate_bridge > 0.95:
            return False
    return True
