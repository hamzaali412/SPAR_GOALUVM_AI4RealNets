import hashlib
import os

import numpy as np

from ACOPF.ACOPF_mod.ac_lodf import ac_lodf

ENABLED = os.environ.get("ACOPF_MAINT_CONTEXT", "1") == "1"
HORIZON_STEPS = int(os.environ.get("ACOPF_MAINT_CONTEXT_STEPS", "48"))
MONITOR_ENTER = float(os.environ.get("ACOPF_MAINT_MONITOR_ENTER", "1.0"))
MONITOR_EXIT = float(os.environ.get("ACOPF_MAINT_MONITOR_EXIT", "0.9"))
FACTOR_MAX_AGE = int(os.environ.get("ACOPF_MAINT_FACTOR_MAX_AGE", "48"))
VMIN_FACTOR = float(os.environ.get("ACOPF_MAINT_VMIN_FACTOR", "0.97"))
Q_FACTOR = float(os.environ.get("ACOPF_MAINT_Q_FACTOR", "1.2"))
WEIGHT = float(os.environ.get("ACOPF_MAINT_CT_WEIGHT", "10.0"))
SECURE_RHO = float(os.environ.get("ACOPF_MAINT_SECURE_RHO", "0.85"))
DISPATCH_CAP_MW = float(os.environ.get("ACOPF_MAINT_DISPATCH_CAP_MW", "250"))
PROGRESS_MIN = float(os.environ.get("ACOPF_MAINT_PROGRESS_MIN", "0.005"))

_STATE = {"episode": 0, "last_step": None, "factors": {}, "monitored": {}, "checked": False, "progress": {}}
STEP_TERMS_HORIZON = int(os.environ.get("ACOPF_MAINT_STEP_TERMS_HORIZON", "12"))
INEFFECTIVE_STREAK = int(os.environ.get("ACOPF_MAINT_INEFFECTIVE_STREAK", "3"))
INEFFECTIVE_MIN_GAIN = float(os.environ.get("ACOPF_MAINT_INEFFECTIVE_MIN_GAIN", "0.01"))


def begin_step(obs):
    step = int(obs.current_step)
    if _STATE["last_step"] is None or step < _STATE["last_step"] or (step == _STATE["last_step"] and step == 0):
        _STATE["episode"] += 1
        _STATE["factors"].clear()
        _STATE["monitored"].clear()
        _STATE["progress"].clear()
    _STATE["last_step"] = step
    return _STATE["episode"]


def _episode(obs):
    if _STATE["last_step"] is None:
        return begin_step(obs)
    return _STATE["episode"]


def _signature(obs):
    return hashlib.md5(np.asarray(obs.topo_vect, dtype=np.int8).tobytes()
                       + np.asarray(obs.line_status, dtype=bool).tobytes()).hexdigest()


def announced_groups(obs, horizon=HORIZON_STEPS):
    tnm = np.asarray(obs.time_next_maintenance, dtype=int)
    status = np.asarray(obs.line_status, dtype=bool)
    soon = [int(l) for l in np.flatnonzero(status & (tnm > 0) & (tnm <= int(horizon)))]
    groups = {}
    for line in soon:
        groups.setdefault(int(tnm[line]), []).append(line)
    return [(first, sorted(lines)) for first, lines in sorted(groups.items())]


def outage_islands(obs, group):
    try:
        cls = type(obs)
        status = np.asarray(obs.line_status, dtype=bool)
        topo = np.asarray(obs.topo_vect, dtype=int)
        n_sub = int(cls.n_sub)
        def gbus(sub, pos):
            b = int(topo[int(pos)])
            return -1 if b <= 0 else int(sub) + (0 if b == 1 else n_sub)
        parent = {}
        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[ra] = rb
        grp = set(int(l) for l in group)
        for l in range(int(cls.n_line)):
            if not status[l] or l in grp: continue
            a = gbus(cls.line_or_to_subid[l], cls.line_or_pos_topo_vect[l]); b = gbus(cls.line_ex_to_subid[l], cls.line_ex_pos_topo_vect[l])
            if a >= 0 and b >= 0: union(a, b)
        loads = [gbus(cls.load_to_subid[i], cls.load_pos_topo_vect[i]) for i in range(int(cls.n_load))]
        gens = [gbus(cls.gen_to_subid[i], cls.gen_pos_topo_vect[i]) for i in range(int(cls.n_gen))]
        buses = [b for b in loads + gens if b >= 0]
        roots = {find(b) for b in buses}
        return len(roots) > 1
    except Exception:
        return None


def event_key(obs, first, group):
    return (_episode(obs), int(obs.current_step) + int(first), tuple(sorted(int(l) for l in group)))


def outage_factors(env, obs, group, state_obs=None):
    state_obs = obs if state_obs is None else state_obs
    islands = outage_islands(state_obs, group)
    if islands is None:
        return {"status": "UNKNOWN_CONNECTIVITY", "structural": "connectivity could not be evaluated"}
    if islands:
        return {"status": "ISLANDING", "structural": "the outage disconnects load or generation: no dispatch remedy, topology must restore connectivity"}
    key = (_episode(obs), _signature(state_obs), tuple(sorted(int(l) for l in group)))
    step = int(obs.current_step)
    cached = _STATE["factors"].get(key)
    if cached is not None and 0 <= step - cached["step"] <= FACTOR_MAX_AGE:
        return cached
    p_or = np.asarray(state_obs.p_or, dtype=float)
    try:
        table = ac_lodf(env, state_obs, [int(l) for l in group])
    except np.linalg.LinAlgError:
        return {"status": "NUMERICAL_FAILURE", "structural": "singular compensation (checked connected: numerical, not physical)"}
    except ValueError as error:
        return {"status": "ALREADY_APPLIED", "structural": f"outage line not in service: {error}"}
    except Exception as error:
        return {"status": "MODEL_UNAVAILABLE", "structural": f"factor computation failed: {error!r}"}
    ref = max((int(l) for l in group), key=lambda l: abs(p_or[l]))
    if abs(p_or[ref]) < 1.0:
        return {"status": "LOW_REFERENCE_FLOW", "structural": "reference outage flow below 1 MW"}
    coeff, p_pre, p_post = {}, {}, {}
    for k, row in table.items():
        k = int(k)
        if "p_pre_mw" not in row:
            continue
        coeff[k] = float((row["p_post_mw"] - row["p_pre_mw"]) / p_or[ref])
        p_pre[k] = float(row["p_pre_mw"])
        p_post[k] = float(row["p_post_mw"])
    if not _STATE["checked"] and p_pre:
        dev = max(abs(p_pre[k] - p_or[k]) for k in p_pre)
        print(f"[maint-ctx] step {step} factor model check: max |model p_pre - obs p_or| = {dev:.2f} MW over {len(p_pre)} lines", flush=True)
        _STATE["checked"] = True
    result = {"status": "CONNECTED", "ref": ref, "coeff": coeff, "p_pre": p_pre, "p_post": p_post, "step": step}
    _STATE["factors"][key] = result
    return result


def thermal_bounds(env, obs):
    i_max = np.asarray(obs.thermal_limit, dtype=float)
    v_or = np.asarray(obs.v_or, dtype=float)
    v_ex = np.asarray(obs.v_ex, dtype=float)
    q_or = np.asarray(obs.q_or, dtype=float)
    q_ex = np.asarray(obs.q_ex, dtype=float)
    v_min = VMIN_FACTOR * np.where((v_or > 0) & (v_ex > 0), np.minimum(v_or, v_ex), np.maximum(v_or, v_ex))
    s_limit = np.sqrt(3.0) * v_min * i_max / 1000.0
    q_bound = Q_FACTOR * np.maximum(np.abs(q_or), np.abs(q_ex))
    ok = np.isfinite(s_limit) & (s_limit > 0) & (q_bound < s_limit)
    p_bound = np.full(i_max.shape, np.nan)
    p_bound[ok] = np.sqrt(s_limit[ok] ** 2 - q_bound[ok] ** 2)
    return p_bound, ~ok & np.isfinite(s_limit) & (s_limit > 0)


def predicted_loading(obs, factors, p_bound, growth=1.0):
    p_or = np.asarray(obs.p_or, dtype=float)
    status = np.asarray(obs.line_status, dtype=bool)
    ref = factors["ref"]
    out = {}
    for k, c in factors["coeff"].items():
        if not status[k] or not np.isfinite(p_bound[k]) or p_bound[k] <= 0:
            continue
        out[k] = abs(p_or[k] + c * p_or[ref]) * float(growth) / float(p_bound[k])
    return out


def terms_for_group(env, obs, group, growth, state_obs=None, first=None):
    if not ENABLED:
        return None, "disabled"
    factors = outage_factors(env, obs, group, state_obs)
    if factors.get("status") != "CONNECTED":
        return None, f"{factors.get('status')}: {factors.get('structural')}"
    within = state_obs is not None
    ref_obs = state_obs if within else obs
    g = 1.0 if within else max(1.0, float(growth))
    p_bound, flagged = thermal_bounds(env, ref_obs)
    loading = predicted_loading(ref_obs, factors, p_bound, g)
    gkey = event_key(obs, first, group) if first is not None else tuple(sorted(int(l) for l in group))
    previous = _STATE["monitored"].get(gkey, set())
    monitored = {k for k, v in loading.items() if v >= MONITOR_ENTER} | {k for k in previous if loading.get(k, 0.0) >= MONITOR_EXIT}
    monitored -= set(int(l) for l in group)
    _STATE["monitored"][gkey] = monitored
    if not monitored:
        return None, f"secure by the surrogate (worst {max(loading.values()) if loading else 0.0:.3f}); {int(flagged.sum())} lines with a reactive bound above their rating"
    term = {"outage": int(factors["ref"]), "weight": WEIGHT,
            "monitored": {int(k): {"lodf": float(factors["coeff"][k]), "pmax_mw": float(p_bound[k]) / g} for k in monitored}}
    worst = max(loading.values())
    return [term], f"worst predicted post-outage loading {worst:.3f} on {sorted(monitored)[:6]}, growth x{g:.2f}"


def surrogate_worst(env, obs, group, growth, state_obs, factors_obs=None):
    factors = outage_factors(env, obs, group, factors_obs)
    if factors.get("status") != "CONNECTED":
        return float("nan")
    p_bound, _flagged = thermal_bounds(env, state_obs)
    loading = predicted_loading(state_obs, factors, p_bound, max(1.0, float(growth)))
    return max(loading.values()) if loading else float("nan")


RETRY_AFTER_STEPS = int(os.environ.get("ACOPF_MAINT_RETRY_AFTER_STEPS", "12"))


def record_progress(obs, first, group, worst_hold, worst_cand):
    ekey = event_key(obs, first, group) + (_signature(obs),)
    rec = _STATE["progress"].setdefault(ekey, {"streak": 0, "ineffective_since": None})
    if not (np.isfinite(worst_hold) and np.isfinite(worst_cand)):
        return False
    gain = worst_hold - worst_cand
    rec["streak"] = rec["streak"] + 1 if gain < PROGRESS_MIN else 0
    if rec["streak"] >= INEFFECTIVE_STREAK and rec["ineffective_since"] is None:
        rec["ineffective_since"] = int(obs.current_step)
    return dispatch_ineffective(obs, first, group)


def dispatch_ineffective(obs, first, group):
    ekey = event_key(obs, first, group) + (_signature(obs),)
    rec = _STATE["progress"].get(ekey)
    if not rec or rec["ineffective_since"] is None:
        return False
    if int(obs.current_step) - rec["ineffective_since"] >= RETRY_AFTER_STEPS:
        rec["ineffective_since"] = None; rec["streak"] = 0
        return False
    return True


def step_terms(env, obs, demand_ratio, outage_time_observation=None):
    if not ENABLED:
        return None
    terms = []
    for first, group in announced_groups(obs, STEP_TERMS_HORIZON):
        if dispatch_ineffective(obs, first, group):
            continue
        try:
            growth = float(demand_ratio(obs, first / 12.0))
        except Exception:
            growth = 1.0
        state_obs = None
        if outage_time_observation is not None:
            try:
                state_obs = outage_time_observation(env, obs, first)
            except Exception:
                state_obs = None
        t, _note = terms_for_group(env, obs, group, growth, state_obs=state_obs, first=first)
        if t:
            terms.extend(t)
    return terms or None


FLEX_MIN_UP_MW = float(os.environ.get("ACOPF_MAINT_FLEX_MIN_UP_MW", "60"))
FLEX_MIN_DOWN_MW = float(os.environ.get("ACOPF_MAINT_FLEX_MIN_DOWN_MW", "60"))


def flexibility(env, state_obs):
    cls = type(env)
    p = np.asarray(state_obs.gen_p, dtype=float)
    red = np.asarray(cls.gen_redispatchable, dtype=bool) & (p > 0.5)
    up = np.minimum(np.asarray(cls.gen_pmax, dtype=float) - p, np.asarray(cls.gen_max_ramp_up, dtype=float))
    down = np.minimum(p - np.asarray(cls.gen_pmin, dtype=float), np.asarray(cls.gen_max_ramp_down, dtype=float))
    return float(np.sum(np.clip(up[red], 0, None))), float(np.sum(np.clip(down[red], 0, None)))


def flexibility_ok(env, state_obs, need_up=0.0, need_down=0.0):
    up, down = flexibility(env, state_obs)
    return up >= max(FLEX_MIN_UP_MW, float(need_up)) and down >= max(FLEX_MIN_DOWN_MW, float(need_down)), up, down
