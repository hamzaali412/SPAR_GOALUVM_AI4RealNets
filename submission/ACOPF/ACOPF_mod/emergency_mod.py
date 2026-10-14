
import os
import time

import numpy as np

from ACOPF.ACOPF_mod import split_search_mod as S
from ACOPF.ACOPF_mod.security_mod import (
    maximum_rho,
    valid_transition,
)


_LAST_CANDIDATES = []
_PENDING_PLAN = {}


def reset_episode_state():
    _PENDING_PLAN.clear()
    _PREVENTIVE_MEMO.clear()
MAX_OUTAGES_FOR_OPENING = int(os.environ.get("ACOPF_MAX_OUTAGES_FOR_OPENING", "4"))
SELF_OPENED = set()
REFINE_SPLITS = os.environ.get("ACOPF_REFINE", "1") == "1"
WATCH_GRIDWIDE = os.environ.get("ACOPF_WATCH_GRIDWIDE", "1") == "1"
AC_DEPTH_SMALL = int(os.environ.get("ACOPF_AC_DEPTH_SMALL", "30"))
AC_DEPTH_LARGE = int(os.environ.get("ACOPF_AC_DEPTH_LARGE", "16"))
URGENCY_FIRST = os.environ.get("ACOPF_URGENCY_FIRST", "1") == "1"
MAX_WORST_LOADING_INCREASE = 0.05
MAX_ACCEPTED_LOADING = 1.5


def sanitize_opening(env, action, opened):
    cls = type(env)
    set_bus = np.asarray(action.set_bus, dtype=int).copy()
    set_status = np.asarray(action.set_line_status, dtype=int).copy()
    for line in opened:
        set_bus[int(cls.line_or_pos_topo_vect[int(line)])] = 0
        set_bus[int(cls.line_ex_pos_topo_vect[int(line)])] = 0
        set_status[int(line)] = -1
    dictionary = {"set_bus": set_bus, "set_line_status": set_status}
    redispatch = np.asarray(action.redispatch, dtype=float)
    if np.any(redispatch != 0.0):
        dictionary["redispatch"] = redispatch
    storage = np.asarray(action.storage_p, dtype=float)
    if storage.size and np.any(storage != 0.0):
        dictionary["set_storage"] = storage
    curtail = np.asarray(action.curtail, dtype=float)
    if curtail.size and np.any(curtail >= 0.0):
        dictionary["curtail"] = curtail
    try:
        return env.action_space(dictionary)
    except Exception:
        return None


def evaluate(env, obs, action, horizons=(0, 1), opened=(), outage=None):
    from ACOPF.ACOPF_mod.security_mod import is_do_nothing, step_cache

    memo_key = None
    if outage is None and not opened and is_do_nothing(action):
        memo_key = ("evaluate_hold", tuple(horizons))
        cache = step_cache(obs)
        if memo_key in cache:
            return cache[memo_key]
    result = _evaluate(env, obs, action, horizons=horizons, opened=opened, outage=outage)
    if memo_key is not None:
        step_cache(obs)[memo_key] = result
    return result


def _evaluate(env, obs, action, horizons=(0, 1), opened=(), outage=None):
    cls = type(env)
    status0 = np.asarray(obs.line_status, dtype=bool)
    opened = {int(l) for l in opened}
    outage_lines = _outage_tuple(outage)
    tnm = np.asarray(obs.time_next_maintenance, dtype=int)
    worst_n, worst_rho, last = 0, 0.0, None
    for horizon in horizons:
        # v83: a horizon that reaches an announced maintenance sees the line go out in the forecast
        # itself; only the outage lines the forecast does not take out by then are forced open in
        # the action.  grid2op allows one line-status change per action, so a group outage beyond
        # the forecast is approximated by its first line (the others are still tolerated as lost).
        # v83b kit-208: this tolerance belongs to the outage evaluation only.  Applied to every
        # evaluation it changed posture, disarm and restoration decisions grid-wide (04-16_59
        # diverged hundreds of steps before its outage), so a plain evaluation keeps v81's rule:
        # a line lost in the forecast disqualifies the candidate.
        by_forecast = ({int(l) for l in np.flatnonzero((tnm >= 0) & (tnm <= int(horizon)))}
                       if outage is not None else set())
        force = sorted(opened) + [l for l in outage_lines if l not in by_forecast and l not in opened]
        if len(force) > 1:
            force = force[:1] if not opened else sorted(opened)[:1]
        allowed = opened | set(outage_lines) | by_forecast
        combined = action
        if force:
            combined = sanitize_opening(env, action, force)
            if combined is None:
                return None
        try:
            sim_obs, _reward, done, info = obs.simulate(combined, time_step=horizon)
        except Exception:
            return None
        rho_max = maximum_rho(sim_obs) if not done else float("nan")
        if not valid_transition(done, info, rho_max):
            return None
        status = np.asarray(sim_obs.line_status, dtype=bool)
        lost = [int(l) for l in np.flatnonzero(status0 & ~status) if int(l) not in allowed]
        if lost:
            return None
        rho = np.where(status, np.asarray(sim_obs.rho, dtype=float), 0.0)
        rho = np.where(np.isfinite(rho), rho, 0.0)
        if outage is not None and _RHO_SCALE["k"] != 1.0:
            rho = rho * _RHO_SCALE["k"]
        worst_n = max(worst_n, int(np.sum(rho > 1.0)))
        worst_rho = max(worst_rho, float(rho.max()))
        last = sim_obs
    return {"n_over": worst_n, "max_rho": worst_rho, "obs": last}


def _key(evaluation):
    if evaluation is None:
        return (10 ** 6, float("inf"))
    return (evaluation["n_over"], round(evaluation["max_rho"], 4))


def _better(candidate_key, reference_key, rho_margin=0.01):
    n = min(len(candidate_key), len(reference_key))
    for i in range(n - 1):
        if candidate_key[i] != reference_key[i]:
            return candidate_key[i] < reference_key[i]
    return candidate_key[n - 1] < reference_key[n - 1] - rho_margin


def _reconnects_offline_line(env, obs, action):
    cls = type(env)
    set_bus = np.asarray(action.set_bus, dtype=int)
    status = np.asarray(obs.line_status, dtype=bool)
    set_status = np.asarray(action.set_line_status, dtype=int)
    for line in np.flatnonzero(~status):
        if set_status[line] == 1:
            return True
        if (
            set_bus[int(cls.line_or_pos_topo_vect[line])] > 0
            or set_bus[int(cls.line_ex_pos_topo_vect[line])] > 0
        ):
            return True
    return False


def refine_split(env, obs, action, sub, score, deadline, max_sweeps=4):
    cls = type(env)
    topo = np.asarray(obs.topo_vect, dtype=int)
    status = np.asarray(obs.line_status, dtype=bool)
    positions = [int(p) for p in np.flatnonzero(cls.grid_objects_types[:, 0] == int(sub))]
    frozen = set()
    for line in np.flatnonzero(~status):
        frozen.add(int(cls.line_or_pos_topo_vect[line]))
        frozen.add(int(cls.line_ex_pos_topo_vect[line]))
    base_bus = np.asarray(action.set_bus, dtype=int)
    current = {}
    for p in positions:
        bus = int(base_bus[p]) if int(base_bus[p]) != 0 else int(topo[p])
        current[p] = bus
    movable = [p for p in positions if p not in frozen and current[p] in (1, 2)]
    if len(movable) < 2:
        return None

    def build(assign):
        set_bus = np.zeros(cls.dim_topo, dtype=int)
        for p in positions:
            if p in frozen or assign[p] not in (1, 2):
                continue
            set_bus[p] = assign[p]
        return env.action_space({"set_bus": set_bus})

    best_key, best_eval = score(action)
    if best_key is None:
        return None
    best_action, best_assign = action, dict(current)
    for _sweep in range(max_sweeps):
        improved = False
        for p in movable:
            if time.perf_counter() > deadline:
                return best_action, best_key, best_eval
            trial = dict(best_assign)
            trial[p] = 3 - trial[p]
            if sum(1 for q in movable if trial[q] == trial[p]) < 2:
                continue
            if sum(1 for q in movable if trial[q] != trial[p]) < 1:
                continue
            candidate = build(trial)
            key, evaluation = score(candidate)
            if key is None or not key < best_key:
                continue
            best_key, best_eval, best_action, best_assign = key, evaluation, candidate, trial
            improved = True
        if not improved:
            break
    return best_action, best_key, best_eval


def _split_candidates(env, obs, sub, monitored, outage_lines=(), top=10, max_candidates=None):
    cls = type(env)
    n = int(cls.sub_info[sub])
    if max_candidates is None:
        max_candidates = 1500 if n > 12 else S.DEFAULT_MAX_CANDIDATES
    ranked = S.dc_rank_splits(
        env,
        obs,
        sub,
        monitored_lines=monitored,
        outage_lines=outage_lines,
        max_candidates=max_candidates,
        focus_lines=list(monitored) + list(outage_lines),
    )
    out = []
    seen = set()
    for r in ranked:
        if len(out) >= top:
            break
        action = S.assignment_action(env, obs, r)
        if action is None:
            continue
        signature = action.set_bus.tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        out.append((action, r))
    return out


def _openable_lines(env, obs, region, critical, blocked, limit=6):
    cls = type(env)
    status = np.asarray(obs.line_status, dtype=bool)
    rho = np.asarray(obs.rho, dtype=float)
    region = set(region)
    out = []
    for line in range(cls.n_line):
        if not status[line] or line in blocked:
            continue
        if int(cls.line_or_to_subid[line]) not in region and int(
            cls.line_ex_to_subid[line]
        ) not in region:
            continue
        if 0 <= int(obs.time_next_maintenance[line]) <= 12:
            continue
        if int(obs.time_before_cooldown_line[line]) > 0:
            continue
        if not np.isfinite(rho[line]) or rho[line] < 0.25:
            continue
        out.append(line)
    out.sort(key=lambda l: -rho[l])
    return out[:limit]


def _ordered_substations(env, obs, lines, region, limit, exclude=()):
    cls = type(env)
    endpoints = set(S.candidate_substations(env, obs, lines, hops=0))
    usable = [
        int(s) for s in region
        if int(obs.time_before_cooldown_sub[s]) == 0
        and int(cls.sub_info[s]) >= 3
        and int(s) not in set(exclude)
    ]
    usable.sort(key=lambda s: (0 if s in endpoints else 1, int(cls.sub_info[s])))
    return usable[:limit]


def _critical_lines(obs):
    status = np.asarray(obs.line_status, dtype=bool)
    rho = np.asarray(obs.rho, dtype=float)
    counters = np.asarray(obs.timestep_overflow, dtype=int)
    rho = np.where(np.isfinite(rho), rho, 0.0)
    return [int(l) for l in np.flatnonzero(status & ((rho > 1.0) | (counters > 0)))]


def _pair_plans(
    env,
    obs,
    subs,
    monitored,
    horizons,
    budget_s,
    started,
    outage=None,
    base_cap=None,
    per_sub=40,
    top_pairs=40,
    per_pair_cap=8,
    max_pair_configs=8000,
):
    import itertools

    cls = type(env)
    outage_lines = _outage_tuple(outage)
    small = [s for s in subs if int(cls.sub_info[s]) <= 10][:14]
    singles = {}
    for sub in small:
        if time.perf_counter() - started > budget_s:
            break
        ranked = S.dc_rank_splits(
            env, obs, sub, monitored_lines=monitored, outage_lines=outage_lines,
            focus_lines=list(monitored) + list(outage_lines),
        )
        configs = []
        seen = set()
        keep = len(ranked) if int(cls.sub_info[sub]) <= 8 else per_sub
        for r in ranked[:keep]:
            config = dict(zip(r["positions"], r["assignment"]))
            key = tuple(sorted(config.items()))
            if key in seen:
                continue
            seen.add(key)
            configs.append(config)
        if configs:
            singles[sub] = configs
    pair_configs, meta = [], []
    for a_sub, b_sub in itertools.combinations(sorted(singles, key=small.index), 2):
        if len(pair_configs) >= max_pair_configs:
            break
        for a in singles[a_sub]:
            for b in singles[b_sub]:
                pair_configs.append({**a, **b})
                meta.append((a_sub, a, b_sub, b))
    if not pair_configs or time.perf_counter() - started > budget_s:
        return []
    scores = S.dc_evaluate_configs(
        env, obs, pair_configs, monitored_lines=monitored, outage_lines=outage_lines
    )
    order = sorted(range(len(scores)), key=lambda i: (round(scores[i][0], 3), scores[i][1]))
    plans = []
    tried = 0
    per_pair = {}
    for i in order:
        if tried >= top_pairs or time.perf_counter() - started > budget_s:
            break
        a_sub, a, b_sub, b = meta[i]
        if per_pair.get((a_sub, b_sub), 0) >= per_pair_cap:
            continue
        per_pair[(a_sub, b_sub)] = per_pair.get((a_sub, b_sub), 0) + 1
        tried += 1
        for first_cfg, first_sub, second_cfg, second_sub in ((a, a_sub, b, b_sub), (b, b_sub, a, a_sub)):
            first = S.config_action(env, obs, first_cfg)
            if first is None or _reconnects_offline_line(env, obs, first):
                continue
            ev1 = evaluate(env, obs, first, horizons)
            if ev1 is None:
                continue
            if base_cap is not None and ev1["max_rho"] > base_cap:
                continue
            sim1 = ev1["obs"]
            if int(sim1.time_before_cooldown_sub[second_sub]) > 0:
                continue
            second = S.config_action(env, sim1, second_cfg)
            if second is None:
                continue
            if base_cap is not None:
                ev2_base = evaluate(env, sim1, second, (1,))
                if ev2_base is None or ev2_base["max_rho"] > base_cap:
                    continue
            ev2 = evaluate(env, sim1, second, (1,), outage=outage)
            if ev2 is None:
                continue
            plans.append((_key(ev2), _key(ev1), first, first_sub, second_sub, second_cfg))
    plans.sort(key=lambda p: (p[0], p[1]))
    return plans


def emergency_topology_action(
    env,
    obs,
    blocked_lines=(),
    horizons=(0, 1),
    allow_openings=True,
    budget_s=30.0,
    max_split_subs=12,
    max_combo_subs=4,
):
    started = time.perf_counter()
    cls = type(env)
    status = np.asarray(obs.line_status, dtype=bool)
    rho = np.asarray(obs.rho, dtype=float)
    counters = np.asarray(obs.timestep_overflow, dtype=int)
    critical = [
        int(l) for l in np.flatnonzero(status & ((rho > 1.0) | (counters > 0)))
    ]
    if not critical:
        return None, None, {"reason": "no_critical_line"}
    blocked = {int(l) for l in blocked_lines}
    hold = evaluate(env, obs, env.action_space({}), horizons)
    imminent = [
        int(l) for l in critical
        if int(counters[l]) >= int(env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED) - 1
    ]

    def cascade_flag(action, evaluation):
        if not imminent or evaluation is None:
            return 0
        final = evaluation["obs"]
        still = [
            l for l in imminent
            if bool(final.line_status[l]) and float(final.rho[l]) >= 1.0
        ]
        if not still:
            return 0
        worst = max(still, key=lambda l: float(final.rho[l]))
        try:
            post = evaluate(env, final, env.action_space({}), (1,), outage=worst)
        except Exception:
            post = None
        return 0 if post is not None else 1

    hold_key = _key(hold)
    hold_cascade = cascade_flag(env.action_space({}), hold)
    hold_key = (hold_key[0], hold_cascade, hold_key[1])

    pending = _PENDING_PLAN.get("plan")
    _PENDING_PLAN.clear()
    if pending is not None and pending["step"] == int(obs.current_step) - 1:
        sub = int(pending["sub"])
        if int(obs.time_before_cooldown_sub[sub]) == 0:
            second = S.config_action(env, obs, pending["config"])
            if second is not None and not _reconnects_offline_line(env, obs, second):
                ev = evaluate(env, obs, second, horizons)
                if ev is not None and (
                    ev["n_over"] < hold_key[0] and ev["max_rho"] <= hold_key[-1] + 1e-6
                    or ev["n_over"] <= hold_key[0] and ev["max_rho"] < hold_key[-1] - 0.005
                ):
                    return (
                        second,
                        f"emergency split @ sub {sub} (plan follow-up)",
                        {"hold_key": hold_key, "best_key": _key(ev), "kind": "plan-followup",
                         "elapsed_s": round(time.perf_counter() - started, 2)},
                    )

    region = S.candidate_substations(env, obs, critical, hops=1)
    subs = _ordered_substations(env, obs, critical, region, max_split_subs)
    pair_pool = _ordered_substations(env, obs, critical, region, 18)
    watch = sorted(
        set(critical)
        | {
            int(l) for l in range(cls.n_line)
            if status[l] and np.isfinite(rho[l]) and rho[l] >= 0.6
        }
    ) if WATCH_GRIDWIDE else list(critical)

    candidates = []

    def consider(action, label, kind, opened=()):
        if opened:
            action = sanitize_opening(env, action, opened)
            if action is None:
                return
        if _reconnects_offline_line(env, obs, action):
            return
        evaluation = evaluate(env, obs, action, horizons, opened=opened)
        if evaluation is None:
            return
        if evaluation["n_over"] > 0:
            if evaluation["max_rho"] > hold_key[-1] + MAX_WORST_LOADING_INCREASE:
                return
            if evaluation["max_rho"] >= MAX_ACCEPTED_LOADING:
                return
        key = _key(evaluation)
        key = (key[0], cascade_flag(action, evaluation), key[1])
        candidates.append((key, kind, action, label, evaluation))

    def _score(action):
        if _reconnects_offline_line(env, obs, action):
            return None, None
        evaluation = evaluate(env, obs, action, horizons)
        if evaluation is None:
            return None, None
        if evaluation["n_over"] > 0 and (
            evaluation["max_rho"] > hold_key[-1] + MAX_WORST_LOADING_INCREASE
            or evaluation["max_rho"] >= MAX_ACCEPTED_LOADING
        ):
            return None, None
        key = _key(evaluation)
        return (key[0], cascade_flag(action, evaluation), key[1]), evaluation

    split_quality = {}
    for sub in subs:
        if time.perf_counter() - started > budget_s:
            break
        top = AC_DEPTH_SMALL if int(cls.sub_info[sub]) <= 12 else AC_DEPTH_LARGE
        for action, ranked in _split_candidates(env, obs, sub, watch, top=top):
            consider(action, f"emergency split @ sub {sub}", "split")
        mine = [c for c in candidates if c[1] == "split" and f"sub {sub}" in c[3]]
        split_quality[sub] = min((c[0] for c in mine), default=(10 ** 6, float("inf")))
        mine.sort(key=lambda c: c[0])
        for seed_key, _kind, seed_action, _label, _ev in (mine[:1] if REFINE_SPLITS else []):
            if time.perf_counter() - started > budget_s:
                break
            refined = refine_split(
                env, obs, seed_action, sub, _score, started + budget_s
            )
            if refined is None:
                continue
            r_action, r_key, r_eval = refined
            if r_key < seed_key:
                candidates.append((r_key, "split", r_action, f"emergency split @ sub {sub} (refined)", r_eval))
                split_quality[sub] = min(split_quality[sub], r_key)

    already_out = int(np.count_nonzero(~status)) - len(SELF_OPENED & set(np.flatnonzero(~status).tolist()))
    if already_out >= MAX_OUTAGES_FOR_OPENING:
        allow_openings = False
    if allow_openings and time.perf_counter() - started < budget_s:
        from ACOPF.ACOPF_mod.security_mod import (
            dynamic_security_set,
            structural_single_outage_failures,
        )

        security = dynamic_security_set(env, obs)
        hazards = {int(l) for l in security["opponent"]} | {int(l) for l in security["maintenance"]}
        base_failures = structural_single_outage_failures(env, obs) & hazards

        def structurally_safe(action):
            return not (
                (structural_single_outage_failures(env, obs, action) & hazards) - base_failures
            )

        lines = _openable_lines(env, obs, region, critical, blocked)
        combo_subs = sorted(subs, key=lambda s: split_quality.get(s, (10 ** 6, float("inf"))))
        combo_subs = [s for s in combo_subs if int(cls.sub_info[s]) <= 12][:max_combo_subs]
        for line in lines:
            if time.perf_counter() - started > budget_s:
                break
            opener = env.action_space({"set_line_status": [(int(line), -1)]})
            if not structurally_safe(opener):
                continue
            consider(opener, f"emergency open line {line}", "open", opened=(line,))
            for sub in combo_subs:
                if time.perf_counter() - started > budget_s:
                    break
                for action, ranked in _split_candidates(
                    env, obs, sub, watch, outage_lines=(line,), top=6, max_candidates=600
                ):
                    combined = opener + action
                    if not structurally_safe(combined):
                        continue
                    consider(
                        combined,
                        f"emergency open line {line} + split @ sub {sub}",
                        "open+split",
                        opened=(line,),
                    )

    meta = {
        "hold_key": hold_key,
        "n_candidates": len(candidates),
    }
    if not candidates:
        meta["reason"] = "no_valid_candidate"
        meta["elapsed_s"] = round(time.perf_counter() - started, 2)
        return None, None, meta
    counting = [int(l) for l in critical if int(counters[int(l)]) >= 1]

    def resets_countdowns(evaluation):
        if not counting or evaluation is None:
            return True
        final = evaluation["obs"]
        return all(float(final.rho[l]) < 1.0 for l in counting if bool(final.line_status[l]))

    order = {"split": 0, "open+split": 1, "open": 2}
    candidates.sort(
        key=lambda c: ((0 if resets_countdowns(c[4]) else 1) if URGENCY_FIRST else 0, c[0], order[c[1]])
    )
    _LAST_CANDIDATES[:] = candidates
    best = candidates[0]
    for c in candidates[1:]:
        if (
            c[0][:-1] == best[0][:-1]
            and c[0][-1] <= best[0][-1] + 0.02
            and order[c[1]] < order[best[1]]
            and resets_countdowns(c[4]) == resets_countdowns(best[4])
        ):
            best = c

    counters = np.asarray(obs.timestep_overflow, dtype=int)
    trip_imminent = any(int(counters[l]) >= 2 for l in critical)
    if (
        best[0][0] > 0
        and time.perf_counter() - started < budget_s * 0.6
        and (trip_imminent or not _better(best[0], hold_key))
    ):
        plans = _pair_plans(env, obs, pair_pool, watch, horizons, budget_s, started)
        meta["plans"] = [(p[0], p[3], p[4]) for p in plans[:3]]
        if plans:
            final_key, first_key, first_action, first_sub, second_sub, second_cfg = plans[0]
            best_plain = (best[0][0], best[0][-1])
            hold_plain = (hold_key[0], hold_key[-1])
            if (
                best[0][0] > 0
                and final_key[0] <= min(best_plain[0], hold_plain[0])
                and final_key[1] < min(best_plain[1], hold_plain[1]) - 0.01
                and first_key[1] <= hold_plain[1] + 0.02
            ):
                _PENDING_PLAN["plan"] = {
                    "step": int(obs.current_step), "sub": int(second_sub), "config": second_cfg,
                }
                meta["best_key"] = first_key
                meta["plan_key"] = final_key
                meta["kind"] = "plan"
                meta["elapsed_s"] = round(time.perf_counter() - started, 2)
                return (
                    first_action,
                    f"emergency split @ sub {first_sub} (then sub {second_sub})",
                    meta,
                )
    meta["elapsed_s"] = round(time.perf_counter() - started, 2)
    if meta["elapsed_s"] >= 0.8 * budget_s:
        print(f"[budget] step {int(obs.current_step)} emergency search used {meta['elapsed_s']}s of {budget_s}s", flush=True)
    if not _better(best[0], hold_key):
        meta["reason"] = "no_improvement_over_hold"
        meta["best_key"] = best[0]
        return None, None, meta
    meta["best_key"] = best[0]
    meta["kind"] = best[1]
    return best[2], best[3], meta


_PREVENTIVE_MEMO = {}
# v83: successive attempts for the same certain outage start from a rotated substation order, so a
# search that is repeated every few steps of a maintenance window explores new candidates instead
# of re-running the same first ones inside the same budget.
_SEARCH_OFFSET = {"k": 0}


def set_search_offset(k):
    _SEARCH_OFFSET["k"] = max(0, int(k))


# v83: post-outage loadings are judged at the projected demand of the outage time (rho * growth),
# so a maintenance outage 40 steps ahead that is secure at today's load and overloaded at the
# outage's load is searched, and a candidate is accepted only if it holds at that load.
_RHO_SCALE = {"k": 1.0}


def set_rho_scale(k):
    _RHO_SCALE["k"] = max(1.0, float(k)) if np.isfinite(k) else 1.0


def _outage_tuple(outage):
    """An outage is one line or a group of lines taken out together (a maintenance event)."""
    if outage is None:
        return ()
    if isinstance(outage, (int, np.integer)):
        return (int(outage),)
    return tuple(sorted({int(l) for l in outage}))


def _rotated(seq):
    k = _SEARCH_OFFSET["k"]
    if not k or len(seq) < 2:
        return list(seq)
    k = k % len(seq)
    return list(seq[k:]) + list(seq[:k])
PREVENTIVE_RETRY_STEPS = 12
PREVENTIVE_RETRY_RHO = 0.015


def _memo_key(obs, outage):
    lines = _outage_tuple(outage)
    return lines[0] if len(lines) == 1 else lines


def _memo_skip(obs, outage):
    step = int(obs.current_step)
    entry = _PREVENTIVE_MEMO.get(_memo_key(obs, outage))
    if entry is None:
        return False
    last_step, last_rho, last_topo = entry
    if step < last_step:
        _PREVENTIVE_MEMO.clear()
        return False
    if step - last_step >= PREVENTIVE_RETRY_STEPS:
        return False
    if abs(maximum_rho(obs) - last_rho) > PREVENTIVE_RETRY_RHO:
        return False
    if last_topo != np.asarray(obs.topo_vect, dtype=int).tobytes():
        return False
    return True


def _memo_record(obs, outage):
    _PREVENTIVE_MEMO[_memo_key(obs, outage)] = (
        int(obs.current_step),
        maximum_rho(obs),
        np.asarray(obs.topo_vect, dtype=int).tobytes(),
    )


def preventive_topology_action(
    env,
    obs,
    outage,
    horizons=(0, 1),
    base_cap=0.95,
    budget_s=20.0,
    max_subs=12,
    overload_threshold=0.95,
    partial_baseline_rho=1.3,
    partial_gain=0.3,
):
    started = time.perf_counter()
    cls = type(env)
    lines = _outage_tuple(outage)
    outage = lines[0] if len(lines) == 1 else lines
    if not all(bool(obs.line_status[l]) for l in lines):
        return None, None, {"reason": "outage_line_already_out"}
    if _memo_skip(obs, outage):
        return None, None, {"reason": "recently_searched_without_result"}
    baseline = evaluate(env, obs, env.action_space({}), horizons, outage=outage)
    baseline_key = _key(baseline)
    if baseline is not None and baseline["n_over"] == 0:
        return None, None, {"reason": "contingency_already_secure"}
    monitored = []
    if baseline is not None:
        post_rho = np.where(
            np.asarray(baseline["obs"].line_status, dtype=bool),
            np.asarray(baseline["obs"].rho, dtype=float),
            0.0,
        )
        monitored = [int(l) for l in np.flatnonzero(post_rho > overload_threshold)]
    region = S.candidate_substations(env, obs, monitored + list(lines), hops=1 if monitored else 0)
    subs = _rotated(_ordered_substations(env, obs, monitored + list(lines), region, max_subs))
    pair_pool = _rotated(_ordered_substations(env, obs, monitored + list(lines), region, 18))
    tag = lines[0] if len(lines) == 1 else "+".join(str(l) for l in lines)   # label token 1 stays a single int

    best = None
    accepted = []
    for sub in subs:
        if time.perf_counter() - started > budget_s:
            break
        for action, ranked in _split_candidates(
            env, obs, sub, monitored or list(lines), outage_lines=lines, top=8
        ):
            if _reconnects_offline_line(env, obs, action):
                continue
            base = evaluate(env, obs, action, horizons)
            if base is None or base["max_rho"] > base_cap:
                continue
            post = evaluate(env, obs, action, horizons, outage=outage)
            if post is None:
                continue
            key = (post["n_over"], round(post["max_rho"], 4), round(base["max_rho"], 4))
            accepted.append((key, action, sub, base, post))
            if best is None or key < best[0]:
                best = (key, action, sub, base, post)
        mine = sorted((a for a in accepted if a[2] == sub), key=lambda a: a[0])
        if mine and time.perf_counter() - started < budget_s:
            _base_store = {}

            def _pscore(action_):
                if _reconnects_offline_line(env, obs, action_):
                    return None, None
                base_ = evaluate(env, obs, action_, horizons)
                if base_ is None or base_["max_rho"] > base_cap:
                    return None, None
                post_ = evaluate(env, obs, action_, horizons, outage=outage)
                if post_ is None:
                    return None, None
                _base_store[id(post_)] = base_
                return (post_["n_over"], round(post_["max_rho"], 4), round(base_["max_rho"], 4)), post_

            refined = refine_split(env, obs, mine[0][1], sub, _pscore, started + budget_s) if REFINE_SPLITS else None
            if refined is not None and refined[1] < mine[0][0]:
                r_action, r_key, r_post = refined
                r_base = _base_store.get(id(r_post)) or evaluate(env, obs, r_action, horizons)
                accepted.append((r_key, r_action, sub, r_base, r_post))
                if best is None or r_key < best[0]:
                    best = (r_key, r_action, sub, r_base, r_post)
    meta = {"baseline_key": baseline_key}
    if time.perf_counter() - started >= 0.8 * budget_s:
        print(f"[budget] step {int(obs.current_step)} preventive search for {outage} used {time.perf_counter() - started:.1f}s of {budget_s}s", flush=True)
    if best is None:
        meta["reason"] = "no_valid_candidate"
        meta["elapsed_s"] = round(time.perf_counter() - started, 2)
        _memo_record(obs, outage)
        return None, None, meta
    key, action, sub, base, post = best
    secured = post["n_over"] == 0
    rescued = baseline is None and post is not None
    if secured or rescued:
        meta["best_key"] = key
        meta["base_rho"] = base["max_rho"]
        meta["elapsed_s"] = round(time.perf_counter() - started, 2)
        return action, f"contingency {lines[0]} preventive split @ sub {sub}" + (f" outage {tag}" if len(lines) > 1 else ""), meta

    plans = _pair_plans(
        env, obs, pair_pool, monitored or list(lines), horizons, budget_s, started,
        outage=outage, base_cap=base_cap,
    )
    plan = None
    for final_key, first_key, first_action, first_sub, second_sub, _second_cfg in plans:
        if final_key[0] == 0:
            plan = ((final_key, first_key), first_action, first_sub, second_sub, first_key)
            break

    if plan is None and baseline is not None and baseline["max_rho"] >= partial_baseline_rho:
        candidates_partial = []
        for key_, action_, sub_, base_, post_ in accepted:
            if post_["n_over"] <= 1 and post_["max_rho"] <= baseline["max_rho"] - partial_gain:
                candidates_partial.append(((post_["n_over"], post_["max_rho"]), action_, sub_, None, key_))
        for final_key, first_key, first_action, first_sub, second_sub, _second_cfg in plans:
            if final_key[0] <= 1 and final_key[1] <= baseline["max_rho"] - partial_gain:
                candidates_partial.append((final_key, first_action, first_sub, second_sub, first_key))
        if candidates_partial:
            candidates_partial.sort(key=lambda item: item[0])
            best_partial = candidates_partial[0]
            meta["partial"] = True
            meta["best_key"] = best_partial[4]
            meta["elapsed_s"] = round(time.perf_counter() - started, 2)
            second = best_partial[3]
            suffix = f" (then sub {second})" if second is not None else ""
            return (
                best_partial[1],
                f"contingency {lines[0]} partial split @ sub {best_partial[2]}{suffix}",
                meta,
            )
    meta["elapsed_s"] = round(time.perf_counter() - started, 2)
    if plan is None:
        meta["reason"] = "no_full_security"
        meta["best_key"] = key
        _memo_record(obs, outage)
        return None, None, meta
    _plan_key, first_action, first_sub, sub2, first_key = plan
    meta["best_key"] = first_key
    meta["plan"] = f"then split @ sub {sub2}"
    return (
        first_action,
        f"contingency {lines[0]} preventive split @ sub {first_sub} (then sub {sub2})",
        meta,
    )
