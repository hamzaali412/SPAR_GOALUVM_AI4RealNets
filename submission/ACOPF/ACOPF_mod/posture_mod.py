
import math
import os

import numpy as np


PMAX_PMIN_RATIO = float(os.environ.get("ACOPF_PMAX_PMIN_RATIO", "4.0"))

SEVERITY_START_RHO = float(os.environ.get("ACOPF_SEVERITY_START_RHO", "1.0"))
SEVERITY_FULL_RHO = float(os.environ.get("ACOPF_SEVERITY_FULL_RHO", "1.5"))

#
#     ["26_31_106", "21_22_93", "17_18_88", "4_10_162", "12_14_68", "29_37_117"]
#     ["62_58_180", "62_63_160", "48_50_136", "48_53_141", "41_48_131",
#      "39_41_121", "43_44_125", "44_45_126", "34_35_110", "54_58_154"]
#     ["74_117_81", "93_95_43", "88_91_33", "91_92_37", "99_105_62", "102_104_61"]
AREA_LINE_NAMES = ()
_AREA_CACHE = {}
_RATIO_CACHE = {}
AREA_VERBOSE = os.environ.get("ACOPF_AREA_VERBOSE", "1") == "1"


def alertable_lines(env):
    lines = getattr(env, "alertable_line_ids", ())
    try:
        return {int(l) for l in lines}
    except Exception:
        return set()


def _env_key(env):
    n_line = int(getattr(type(env), "n_line", -1))
    return (id(env), n_line, tuple(sorted(alertable_lines(env))))


def _clean(groups, n_line):
    out = []
    seen = set()
    for group in groups:
        ids = []
        for line in group:
            line = int(line)
            if 0 <= line < n_line and line not in seen:
                seen.add(line)
                ids.append(line)
        if ids:
            out.append(ids)
    return out


def _from_live_opponent(env):
    opponent = getattr(env, "_opponent", None)
    if opponent is None:
        return None
    subs = getattr(opponent, "list_opponents", None) or getattr(opponent, "_list_opponents", None)
    if not subs:
        ids = getattr(opponent, "_lines_ids", None)
        return [list(ids)] if ids is not None and len(ids) else None
    groups = []
    for sub in subs:
        ids = getattr(sub, "_lines_ids", None)
        if ids is None or not len(ids):
            return None
        groups.append(list(ids))
    return groups or None


def _from_kwargs(env):
    try:
        names = env._kwargs_opponent["lines_attacked"]
    except Exception:
        return None
    if not names:
        return None
    if isinstance(names[0], (str, bytes)):     # a flat list of names = one area
        names = [names]
    index = {str(n): i for i, n in enumerate(env.name_line)}
    groups = [[index[str(n)] for n in group if str(n) in index] for group in names]
    return groups or None


def _from_constant(env):
    index = {str(n): i for i, n in enumerate(env.name_line)}
    groups = [[index[str(n)] for n in group if str(n) in index] for group in AREA_LINE_NAMES]
    resolved = {l for group in groups for l in group}
    alertable = alertable_lines(env)
    if alertable and resolved != alertable:
        return None                              # this grid is not the one these names describe
    return groups or None


def areas(env):
    key = _env_key(env)
    if key in _AREA_CACHE:
        return _AREA_CACHE[key][0]
    n_line = int(getattr(type(env), "n_line", 0))
    result, source = None, "none"
    for name, finder in (("opponent", _from_live_opponent), ("env config", _from_kwargs),
                         ("built-in names", _from_constant)):
        try:
            candidate = _clean(finder(env) or [], n_line)
        except Exception:
            candidate = []
        if candidate:
            result, source = candidate, name
            break
    if not result:
        alertable = sorted(alertable_lines(env))
        result, source = ([alertable] if alertable else []), "alertable lines (no grouping found)"
    _AREA_CACHE[key] = (result, source)
    if AREA_VERBOSE:
        alertable = alertable_lines(env)
        covered = {l for group in result for l in group}
        note = ""
        if alertable and covered != alertable:
            note = (f"; WARNING: {len(alertable - covered)} alertable line(s) in no area, "
                    f"{len(covered - alertable)} area line(s) not alertable")
        print(f"[areas] {len(result)} opponent area(s) from {source}: "
              f"{[len(g) for g in result]} lines{note}", flush=True)
    return result


def area_source(env):
    areas(env)
    return _AREA_CACHE[_env_key(env)][1]


def _ratio_from_live_opponent(env, n_areas):
    opponent = getattr(env, "_opponent", None)
    if opponent is None:
        return None
    subs = getattr(opponent, "list_opponents", None) or getattr(opponent, "_list_opponents", None)
    if subs and len(subs) == n_areas:
        values = [getattr(sub, "_pmax_pmin_ratio", None) for sub in subs]
        if all(v is not None and float(v) > 1.0 for v in values):
            return [float(v) for v in values]
    single = getattr(opponent, "_pmax_pmin_ratio", None)
    if single is not None and float(single) > 1.0:
        return [float(single)] * n_areas
    return None


def _ratio_from_kwargs(env, n_areas):
    try:
        value = env._kwargs_opponent["pmax_pmin_ratio"]
    except Exception:
        return None
    if isinstance(value, (list, tuple)) and len(value) == n_areas:
        return [float(v) for v in value]
    try:
        value = float(value)
    except Exception:
        return None
    return [value] * n_areas if value > 1.0 else None


def area_ratios(env):
    """The ratio between the most and the least likely target of each area, as the environment's
    opponent will use it (``GeometricOpponent._pmax_pmin_ratio``)."""
    key = _env_key(env)
    cached = _RATIO_CACHE.get(key)
    if cached is not None:
        return cached[0]
    n_areas = len(areas(env))
    result, source = None, "built-in default"
    for name, finder in (("opponent", _ratio_from_live_opponent), ("env config", _ratio_from_kwargs)):
        candidate = None
        try:
            candidate = finder(env, n_areas)
        except Exception:
            candidate = None
        if candidate:
            result, source = candidate, name
            break
    if result is None:
        result = [PMAX_PMIN_RATIO] * n_areas
    _RATIO_CACHE[key] = (result, source)
    if AREA_VERBOSE:
        shown = result[0] if len(set(result)) == 1 else result
        print(f"[areas] attack-probability ratio from {source}: {shown}", flush=True)
    return result


def ratio_source(env):
    area_ratios(env)
    return _RATIO_CACHE[_env_key(env)][1]


def attackable_groups(env, obs):
    """Areas that can still attack: every line live under the abort model, at least one otherwise."""
    status = np.asarray(obs.line_status, dtype=bool)
    if OPPONENT_ABORT_MODEL:
        return [g for g in areas(env) if status[g].all()]
    return [g for g in areas(env) if status[g].any()]


def area_armed(env, obs):
    status = np.asarray(obs.line_status, dtype=bool)
    return [bool(status[g].all()) for g in areas(env)]



_ABORT_ENV = os.environ.get("ACOPF_OPPONENT_ABORT_MODEL", "auto").strip().lower()
ABORT_EVIDENCE_EXPECTED = float(os.environ.get("ACOPF_ABORT_EVIDENCE_EXPECTED", "3.0"))

ABORT_PRIOR_LOGODDS = {"any_out": float(os.environ.get("ACOPF_ABORT_PRIOR_ANYOUT", "2.2")),
                       "all_out": float(os.environ.get("ACOPF_ABORT_PRIOR_ALLOUT", "-2.2")),
                       "schedule": float(os.environ.get("ACOPF_ABORT_PRIOR_SCHEDULE", "-2.2")),
                       None: 0.0}
ABORT_STRIKE_EPS = float(os.environ.get("ACOPF_ABORT_STRIKE_EPS", "0.05"))
ABORT_LOGODDS_CAP = float(os.environ.get("ACOPF_ABORT_LOGODDS_CAP", "6.0"))


def abort_probability():
    if _ABORT_ENV == "1":
        return 1.0
    if _ABORT_ENV != "auto":
        return 0.0
    x = float(_ABORT_STATE.get("logodds", 0.0))
    return 1.0 / (1.0 + math.exp(-x))
FATAL_NEEDS_SIMULATION = os.environ.get("ACOPF_FATAL_NEEDS_SIMULATION", "1") == "1"
_ABORT_STATE = {"exposed_steps": 0.0, "attacks_exposed": 0, "attacks_live": 0, "confirmed": False, "falsified": False,
                "step": -1, "since": None, "hazard": None, "announced": False, "status_prev": None, "identified": False,
                # v81: the class's documented rule ("any_out", "all_out", "schedule" or None), the number of
                # exposed-area attacks needed to falsify it, and the exposed area-steps seen since the last one
                "rule": None, "strikes_needed": 1, "quiet_exposed_since_strike": 0.0, "class_names": None,
                "logodds": 0.0}

REIDENTIFY_OPPONENT = os.environ.get("ACOPF_OPPONENT_REIDENTIFY", "1") == "1"

ABORT_FALSIFY_STRIKES = int(os.environ.get("ACOPF_ABORT_FALSIFY_STRIKES", "2"))
ABORT_RECONFIRM_EXPECTED = float(os.environ.get("ACOPF_ABORT_RECONFIRM_EXPECTED", "3.0"))

IDENTIFY_OPPONENT = os.environ.get("ACOPF_OPPONENT_IDENTIFY", "1") == "1"
_GEOMETRIC_NAMES = ("GeometricOpponent", "GeometricOpponentMultiArea")

_CLASS_RULES = {
    "GeometricOpponent": "any_out", "GeometricOpponentMultiArea": "any_out",
    "RandomLineOpponent": "all_out", "WeightedRandomOpponent": "all_out",
    "FromEpisodeDataOpponent": "schedule",
}


def _opponent_classes(env):
    opponent = getattr(env, "_opponent", None)
    if opponent is None:
        return []
    names = [type(opponent).__name__]
    subs = getattr(opponent, "list_opponents", None) or getattr(opponent, "_list_opponents", None) or []
    names += [type(o).__name__ for o in subs]
    return list(dict.fromkeys(names))


def _opponent_declared(env):
    """Attack duration, cooldown and budget per step as the environment declares them (diagnostic)."""
    out = {}
    for key, attr in (("duration", "_opponent_attack_duration"), ("cooldown", "_opponent_attack_cooldown"),
                      ("budget_per_ts", "_opponent_budget_per_ts"), ("init_budget", "_opponent_init_budget")):
        try:
            v = getattr(env, attr, None)
            if v is not None:
                out[key] = float(v)
        except Exception:
            pass
    return out


def _identify_opponent(env):
    """(rule, names): the documented rule of the environment's opponent class, or None when unknown."""
    names = _opponent_classes(env)
    if not names:
        return None, "no opponent"
    rules = [_CLASS_RULES[n] for n in names if n in _CLASS_RULES]
    rule = None
    if rules:
        # the wrapper's own name may not be in the table; the sub-opponents decide, and a mixed list
        # of classes is treated as unknown
        rule = rules[0] if all(r == rules[0] for r in rules) else None
    return rule, "/".join(names)


def _hazard_rate(env):
    if _ABORT_STATE["hazard"] is None:
        rate = 0.0
        try:
            for o in getattr(env._opponent, "list_opponents", []):
                rate = max(rate, float(getattr(o, "_attack_hazard_rate", 0.0) or 0.0))
        except Exception:
            rate = 0.0
        _ABORT_STATE["hazard"] = rate if rate > 0.0 else 1.0 / 360.0
    return _ABORT_STATE["hazard"]


def _adopt_class(env, step, changed_from=None):
    """Take the documented rule of the environment's opponent class as the current prior.

    Called at the first observation of the process and, in v82, at the first observation of every
    episode whose opponent class differs from the one in force.  A change resets the evidence: the
    exposure counts, the strike count and the hazard rate all belong to the class that produced them."""
    st = _ABORT_STATE
    rule, names = _identify_opponent(env)
    st["identified"] = True
    if changed_from is not None:
        st.update(exposed_steps=0.0, attacks_exposed=0, attacks_live=0, confirmed=False, falsified=False,
                  quiet_exposed_since_strike=0.0, hazard=None)
    st["rule"], st["class_names"] = rule, names
    st["logodds"] = float(ABORT_PRIOR_LOGODDS.get(rule, 0.0))
    declared = _opponent_declared(env)
    note = f" (opponent changed from {changed_from}; evidence reset)" if changed_from is not None else ""
    if rule == "any_out":
        st["confirmed"] = True
        st["attacks_exposed"] = 0
        st["strikes_needed"] = max(1, ABORT_FALSIFY_STRIKES)
        print(f"[opponent] step {step}: {names} identified from the environment{note} -- its attack() aborts when any "
              f"attackable line of the area is disconnected; taken as the adversary model, dropped only after "
              f"{st['strikes_needed']} observed attacks on exposed areas, re-confirmed after "
              f"{ABORT_RECONFIRM_EXPECTED:g} expected ones with none; declared {declared}", flush=True)
    else:
        st["strikes_needed"] = 1
        print(f"[opponent] step {step}: {names} identified from the environment{note} -- documented rule "
              f"{rule or 'unknown'}: no disarm prior; the honest model runs and the behaviour is tested "
              f"({ABORT_EVIDENCE_EXPECTED:g} expected attacks on exposed areas with none observed confirm the "
              f"abort behaviour); declared {declared}", flush=True)


def observe_opponent(env, obs):
    """Called once per step (from the controller): update the abort evidence."""
    if _ABORT_ENV != "auto":
        return
    st = _ABORT_STATE
    step = int(obs.current_step)
    if IDENTIFY_OPPONENT and not st["identified"]:
        _adopt_class(env, step, changed_from=None)
    elif REIDENTIFY_OPPONENT and IDENTIFY_OPPONENT and (step == 0 or step < st["step"]):
        # first observation of an episode: has the opponent changed under us?
        _rule, names = _identify_opponent(env)
        if names != st["class_names"]:
            _adopt_class(env, step, changed_from=st["class_names"])
    if step == st["step"]:
        return
    st["step"] = step
    try:
        status = np.asarray(obs.line_status, dtype=bool)
        ids = np.asarray(obs.alertable_line_ids, dtype=int)
        since = np.asarray(obs.time_since_last_attack, dtype=int)
    except Exception:
        return
    started = {int(ids[p]) for p in range(min(ids.size, since.size)) if since[p] == 0}
    prev = st["since"]
    st["since"] = since.copy()
    status_prev = st["status_prev"]
    st["status_prev"] = status.copy()
    if prev is None or step == 0 or status_prev is None:
        return
    for group in areas(env):
        lines = [int(l) for l in group]
        # exposure is judged on the state the opponent saw, the previous step's observation: a line
        # attacked now was live, and a line whose maintenance starts this very step was live too
        # (v75; 06-11_56 step 44 falsified the model on such a line and 12 later weeks ran without it)
        out_before = [l for l in lines if not status_prev[l] and l not in started]
        hit = [l for l in lines if l in started]
        if out_before:
            st["exposed_steps"] += 1.0
            st["quiet_exposed_since_strike"] += 1.0
            before_p = abort_probability()
            if hit:
                st["logodds"] = max(-ABORT_LOGODDS_CAP, st["logodds"] + math.log(ABORT_STRIKE_EPS))
            else:
                st["logodds"] = min(ABORT_LOGODDS_CAP, st["logodds"] + _hazard_rate(env))
            after_p = abort_probability()
            if (before_p >= 0.5) != (after_p >= 0.5):
                print(f"[opponent] step {step}: P(abort) {before_p:.2f} -> {after_p:.2f}: the abort model is now "
                      f"{'on' if after_p >= 0.5 else 'off'} (posterior crossed 0.5)", flush=True)
            if hit:
                st["quiet_exposed_since_strike"] = 0.0
                if st["confirmed"]:
                    st["attacks_exposed"] += 1          # v82: a strike counts only against a model in force
                else:
                    st["attacks_exposed"] = 0
                if st["confirmed"] and st["attacks_exposed"] >= st["strikes_needed"]:
                    st["falsified"] = True; st["confirmed"] = False
                    strike = st["attacks_exposed"]; st["attacks_exposed"] = 0
                    print(f"[opponent] step {step}: an attack started on line {hit[0]} while its area had {out_before} out "
                          f"(strike {strike} of {st['strikes_needed']}) -- the abort behaviour is falsified; "
                          f"honest model until {ABORT_RECONFIRM_EXPECTED:g} expected attacks pass unobserved", flush=True)
                elif st["confirmed"]:
                    print(f"[opponent] step {step}: an attack started on line {hit[0]} while its area had {out_before} out "
                          f"(strike {st['attacks_exposed']} of {st['strikes_needed']}) -- model kept", flush=True)
                elif not st["falsified"]:
                    # never confirmed: an observed exposed attack resets the evidence for the honest start
                    st["exposed_steps"] = 0.0
        elif hit:
            st["attacks_live"] += 1
    h = _hazard_rate(env)
    if st["falsified"] and st["quiet_exposed_since_strike"] * h >= ABORT_RECONFIRM_EXPECTED:
        # layer 2: reversible.  Enough exposed area-steps have passed without an attack since the last strike.
        st["falsified"] = False; st["confirmed"] = True; st["attacks_exposed"] = 0
        st["quiet_exposed_since_strike"] = 0.0
        print(f"[opponent] step {step}: {ABORT_RECONFIRM_EXPECTED:g} attacks expected on exposed areas since the last strike, "
              f"none observed -- the abort behaviour is taken as observed again", flush=True)
    elif not st["confirmed"] and not st["falsified"] and st["attacks_live"] >= 1 \
            and st["exposed_steps"] * h >= ABORT_EVIDENCE_EXPECTED:
        st["confirmed"] = True
        st["attacks_exposed"] = 0
        st["strikes_needed"] = max(1, ABORT_FALSIFY_STRIKES) if st.get("rule") == "any_out" else 1
        print(f"[opponent] step {step}: {st['exposed_steps']:.0f} exposed area-steps ({st['exposed_steps'] * h:.1f} attacks expected "
              f"under the honest model) with none observed, {st['attacks_live']} attacks on live areas -- the abort behaviour is taken as observed", flush=True)


def abort_model_active():
    if _ABORT_ENV == "1":
        return True
    if _ABORT_ENV == "auto":
        return abort_probability() >= 0.5
    return False


class _AbortFlag:
    """Backwards-compatible truthy flag: `posture_mod.OPPONENT_ABORT_MODEL` reads the live decision."""
    def __bool__(self):
        return abort_model_active()
    __nonzero__ = __bool__
    def __repr__(self):
        return f"OPPONENT_ABORT_MODEL({_ABORT_ENV}: {'on' if abort_model_active() else 'off'})"


OPPONENT_ABORT_MODEL = _AbortFlag()


def attack_response(env, forecast_obs):
    """The opponent's target distribution: within an area the live lines are ranked by loading
    (0 = least loaded) and drawn with probability proportional to ``exp(beta * rank)``,
    ``beta = ln(pmax_pmin_ratio) / (n_attackable - 1)``, ratio and grouping from the environment.
    With the abort model an area holding an out-of-service line attacks nothing; without it the
    area attacks among its remaining live lines."""
    status = np.asarray(forecast_obs.line_status, dtype=bool)
    rho = np.asarray(forecast_obs.rho, dtype=float)
    ratios = area_ratios(env)
    q = {}
    for group, ratio in zip(areas(env), ratios):
        for l in group:
            q[int(l)] = 0.0
        exposed = not status[group].all()
        scale = (1.0 - abort_probability()) if (exposed and _ABORT_ENV == "auto") else (0.0 if (exposed and OPPONENT_ABORT_MODEL) else 1.0)
        if scale <= 1e-6:
            continue
        live = [int(l) for l in group if status[int(l)]]
        n_attackable = len(live)
        if n_attackable == 0:
            continue
        if n_attackable == 1:
            q[live[0]] = float(scale)
            continue
        r = np.where(np.isfinite(rho[live]), rho[live], 0.0)
        order = np.argsort(r)
        ranks = np.empty(n_attackable, dtype=int)
        ranks[order] = np.arange(n_attackable)
        beta = math.log(float(ratio)) / (n_attackable - 1)
        weights = np.exp(beta * ranks)
        weights /= weights.sum()
        for l, w in zip(live, weights):
            q[int(l)] = float(w) * float(scale)
    return q


def severity(item, demand_scale=1.0):
    rho = float(item.get("rho", float("inf")))
    if not np.isfinite(rho):
        return 1.0
    rho *= float(demand_scale)
    if rho <= SEVERITY_START_RHO:
        return 0.0
    return float(min(1.0, (rho - SEVERITY_START_RHO) / (SEVERITY_FULL_RHO - SEVERITY_START_RHO)))


def expected_loss(env, forecast_obs, screen, demand_scale=1.0, fatal=(), certain=()):
    q = attack_response(env, forecast_obs)
    status = np.asarray(forecast_obs.line_status, dtype=bool)
    for line in certain:
        if 0 <= int(line) < status.size and status[int(line)]:
            q[int(line)] = 1.0
    by_line = {int(item["line"]): item for item in screen["results"]}
    terms = {}
    for line, prob in q.items():
        item = by_line.get(line)
        # The capacity certificate is a bound, not a simulation: it called line 81 fatal in states
        # where the N-1 simulation showed 0.8 and where the agent itself opened 81 and ran for
        # hours.  With the certificate overriding the simulation, every risk ranking was led by 81
        # (S 1.00) and the splits and openings chased it while the line that actually killed the
        # week (136, S 0.6) went untreated.  The override now holds only where the simulation
        # agrees: the contingency diverged / islanded, or its loading is at or above 1.0.
        sim_rho = float(item.get("rho", float("nan"))) if item is not None else float("nan")
        certified = (line in fatal and prob > 0.0) and (
            not FATAL_NEEDS_SIMULATION or item is None or not np.isfinite(sim_rho) or sim_rho >= 1.0)
        if certified:
            sev = 1.0
        elif item is None and prob > 0.0:
            sev = max([severity(it, demand_scale) for it in screen["results"]] or [0.0])
        else:
            sev = severity(item, demand_scale) if item is not None else 0.0
        terms[line] = (prob, sev, prob * sev)
    return float(sum(t[2] for t in terms.values())), terms


def base_severity(forecast_obs, demand_scale=1.0):
    rho = np.asarray(forecast_obs.rho, dtype=float)
    status = np.asarray(forecast_obs.line_status, dtype=bool)
    live = rho[status]
    if live.size == 0 or not np.all(np.isfinite(live)):
        return 1.0
    return severity({"rho": float(live.max())}, demand_scale)


def format_terms(terms, top=3):
    worst = sorted(terms.items(), key=lambda kv: -kv[1][2])[:top]
    return ", ".join(f"{l}: q {p:.3f} S {s:.2f}" for l, (p, s, _r) in worst if s > 0)
