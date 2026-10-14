"""v110: the announced-outage context for the dispatch (design review "v109 Astra", P0 items).

Factors: the compensation (Z-bus) AC line-outage factors of ``ac_lodf`` -- they need only the intact
state, handle a joint outage group exactly, and fail loudly on a bridge outage (singular compensation)
instead of returning a number.  Factors are cached by (episode, bus topology, line status, group) with a
non-negative age; ratings, forecasts and the monitored set are refreshed every step.  Ratings come from
the backend's current limit with a voltage lower bound and a reactive bound, never from ``|p|/rho``.
"""
import hashlib
import os

import numpy as np

from ACOPF.ACOPF_mod.ac_lodf import ac_lodf

ENABLED = os.environ.get("ACOPF_MAINT_CONTEXT", "1") == "1"
HORIZON_STEPS = int(os.environ.get("ACOPF_MAINT_CONTEXT_STEPS", "48"))
MONITOR_ENTER = float(os.environ.get("ACOPF_MAINT_MONITOR_ENTER", "1.0"))    # v111: only predicted overloads
MONITOR_EXIT = float(os.environ.get("ACOPF_MAINT_MONITOR_EXIT", "0.9"))
FACTOR_MAX_AGE = int(os.environ.get("ACOPF_MAINT_FACTOR_MAX_AGE", "48"))
VMIN_FACTOR = float(os.environ.get("ACOPF_MAINT_VMIN_FACTOR", "0.97"))    # lower voltage bound as a fraction of the observed voltage
Q_FACTOR = float(os.environ.get("ACOPF_MAINT_Q_FACTOR", "1.2"))           # upper bound on |Q| as a multiple of the observed one
WEIGHT = float(os.environ.get("ACOPF_MAINT_CT_WEIGHT", "10.0"))          # v111: x 200 = 2,000 per unit of slack; the slack is used where dispatch cannot help
SECURE_RHO = float(os.environ.get("ACOPF_MAINT_SECURE_RHO", "0.85"))   # v112: the first-order surrogate is not conservative
DISPATCH_CAP_MW = float(os.environ.get("ACOPF_MAINT_DISPATCH_CAP_MW", "250"))   # v112: no constrained solve beyond this standing dispatch
PROGRESS_MIN = float(os.environ.get("ACOPF_MAINT_PROGRESS_MIN", "0.005"))

_STATE = {"episode": 0, "last_step": None, "factors": {}, "monitored": {}, "checked": False, "progress": {}}
STEP_TERMS_HORIZON = int(os.environ.get("ACOPF_MAINT_STEP_TERMS_HORIZON", "12"))   # v111: ordinary solves constrained only inside the forecast window
INEFFECTIVE_STREAK = int(os.environ.get("ACOPF_MAINT_INEFFECTIVE_STREAK", "3"))
INEFFECTIVE_MIN_GAIN = float(os.environ.get("ACOPF_MAINT_INEFFECTIVE_MIN_GAIN", "0.01"))


def begin_step(obs):
    """v113: the lifecycle advances here, once per real decision (called from the controller's step
    entry).  A step at or below the last one is a new episode; every other query of this module is
    read-only with respect to episode identity, so same-step calls are idempotent."""
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
    """Outage groups announced within the horizon: lines in service whose maintenance starts at the same step."""
    tnm = np.asarray(obs.time_next_maintenance, dtype=int)
    status = np.asarray(obs.line_status, dtype=bool)
    soon = [int(l) for l in np.flatnonzero(status & (tnm > 0) & (tnm <= int(horizon)))]
    groups = {}
    for line in soon:
        groups.setdefault(int(tnm[line]), []).append(line)
    return [(first, sorted(lines)) for first, lines in sorted(groups.items())]


def outage_islands(obs, group):
    """True when removing the group's lines (from the current bus topology) disconnects a bus that carries
    load or generation: the first-order factors mean nothing then and the outage is a structural event."""
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
        return None          # unknown, not "connected"


def event_key(obs, first, group):
    """(episode, outage start step, sorted lines): a line group alone does not identify an event."""
    return (_episode(obs), int(obs.current_step) + int(first), tuple(sorted(int(l) for l in group)))


def outage_factors(env, obs, group, state_obs=None):
    """Typed outcome.  status: CONNECTED (with 'ref', 'coeff', 'p_pre', 'p_post', 'step'), ISLANDING,
    UNKNOWN_CONNECTIVITY, NUMERICAL_FAILURE, ALREADY_APPLIED, LOW_REFERENCE_FLOW, MODEL_UNAVAILABLE.
    Factors from the intact state ``state_obs`` (the current observation, or the outage-time forecast)."""
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
        # one-time check that the network model's flows are the observation's flows
        dev = max(abs(p_pre[k] - p_or[k]) for k in p_pre)
        print(f"[maint-ctx] step {step} factor model check: max |model p_pre - obs p_or| = {dev:.2f} MW over {len(p_pre)} lines", flush=True)
        _STATE["checked"] = True
    result = {"status": "CONNECTED", "ref": ref, "coeff": coeff, "p_pre": p_pre, "p_post": p_post, "step": step}
    _STATE["factors"][key] = result
    return result


def thermal_bounds(env, obs):
    """Active-power bound per line in MW from the backend's current limit, a voltage lower bound and a
    reactive upper bound; nan where the reactive bound alone exhausts the rating (flagged)."""
    try:
        i_max = np.asarray(env.get_thermal_limit(), dtype=float)
    except Exception:
        i_max = np.asarray(env.backend.get_thermal_limit(), dtype=float)
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
    """Predicted post-outage active loading |p_k + c_k * p_ref| * growth / P_bound_k per surviving line."""
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
    """(terms, note).  terms is None when unavailable; note carries the diagnosis.  Ratings and loading
    are evaluated on the same state as the factors (the outage-time forecast inside the window, no
    growth; the current state with growth beyond it)."""
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
    """Worst predicted post-outage loading at ``state_obs`` (a hold or candidate forecast), with the
    factors of the current topology; nan when unavailable."""
    factors = outage_factors(env, obs, group, factors_obs)
    if factors.get("status") != "CONNECTED":
        return float("nan")
    p_bound, _flagged = thermal_bounds(env, state_obs)
    loading = predicted_loading(state_obs, factors, p_bound, max(1.0, float(growth)))
    return max(loading.values()) if loading else float("nan")


RETRY_AFTER_STEPS = int(os.environ.get("ACOPF_MAINT_RETRY_AFTER_STEPS", "12"))


def record_progress(obs, first, group, worst_hold, worst_cand):
    """Dispatch-effectiveness bookkeeping per event and topology.  A numerical failure (nan) is not
    evidence of no physical effect and does not count.  Marked ineffective after INEFFECTIVE_STREAK
    consecutive solves below PROGRESS_MIN (the acceptance threshold, so accepted small steps never
    trigger it); the mark expires after RETRY_AFTER_STEPS or on a topology change."""
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
    """Terms for the ordinary dispatch solves: only groups inside the forecast window, evaluated on the
    outage-time forecast state (T-1 intact proxy) when the caller provides it, and not marked ineffective."""
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


# --- Stage B: the flexibility gate on a candidate's next state (replaces the 250 MW cap as the decision)
FLEX_MIN_UP_MW = float(os.environ.get("ACOPF_MAINT_FLEX_MIN_UP_MW", "60"))
FLEX_MIN_DOWN_MW = float(os.environ.get("ACOPF_MAINT_FLEX_MIN_DOWN_MW", "60"))


def flexibility(env, state_obs):
    """Deliverable one-step up and down room of the redispatchable fleet at ``state_obs``: for each unit
    min(pmax - p, ramp_up) and min(p - pmin, ramp_down), off units excluded (they cannot be redispatched)."""
    cls = type(env)
    p = np.asarray(state_obs.gen_p, dtype=float)
    red = np.asarray(cls.gen_redispatchable, dtype=bool) & (p > 0.5)
    up = np.minimum(np.asarray(cls.gen_pmax, dtype=float) - p, np.asarray(cls.gen_max_ramp_up, dtype=float))
    down = np.minimum(p - np.asarray(cls.gen_pmin, dtype=float), np.asarray(cls.gen_max_ramp_down, dtype=float))
    return float(np.sum(np.clip(up[red], 0, None))), float(np.sum(np.clip(down[red], 0, None)))


def flexibility_ok(env, state_obs, need_up=0.0, need_down=0.0):
    up, down = flexibility(env, state_obs)
    return up >= max(FLEX_MIN_UP_MW, float(need_up)) and down >= max(FLEX_MIN_DOWN_MW, float(need_down)), up, down
