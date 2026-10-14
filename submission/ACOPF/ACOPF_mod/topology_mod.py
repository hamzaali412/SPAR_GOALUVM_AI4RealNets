from ACOPF.ACOPF_mod.recovery_mod import maintenance_blocks_reconnection

import itertools
import numpy as np

from ACOPF.ACOPF_mod.ac_lodf import ac_lodf
from ACOPF.ACOPF_mod.overload_graph import (
    _select_best,
    action_from_split,
    build_gov,
    classify_gov,
    identify_paths,
    label_upstream_downstream,
    path_c_branch_switch_candidates,
    relief_candidates,
    split,
    substation_ranking,
)
from ACOPF.ACOPF_mod.interventions import (
    build_restore_action,
    cooldown_cleared,
)

from ACOPF.ACOPF_mod.security_mod import (
    contingency_not_worse,
    dynamic_security_set,
    maximum_rho,
    screen_dynamic_contingencies,
    structural_double_outage_failures,
    structural_single_outage_failures,
    topology_bridges,
    topology_protection_safe,
    validate_action,
    validation_horizons,
    valid_transition,
)


MAX_MANAGED_TOPOLOGY = 1
ALLOW_BRANCH_DISCONNECT = False
MAX_BRIDGE_TOPOLOGY_CANDIDATES = 64
MAX_SECOND_OUTAGE_CANDIDATES = 6


def _same_action_effect(first, second):
    return bool(
        np.array_equal(
            np.asarray(first.set_bus, dtype=int),
            np.asarray(second.set_bus, dtype=int),
        )
        and np.array_equal(
            np.asarray(first.set_line_status, dtype=int),
            np.asarray(second.set_line_status, dtype=int),
        )
    )


def _filtered_lodf_pipeline(congested_line, env, obs, blocked_lines):
    blocked_lines = set(map(int, blocked_lines))
    lodf = ac_lodf(env, obs, [int(congested_line)])
    ranked_candidates = []
    seen = set()
    for threshold in (0.05, 0.01):
        graph = build_gov(
            env, obs, int(congested_line), threshold=threshold, lodf=lodf
        )
        try:
            paths = identify_paths(graph)
            upstream, downstream = label_upstream_downstream(env, obs, graph)
            gov_type, _ = classify_gov(
                env, obs, graph, paths, upstream
            )
        except (StopIteration, ValueError):
            continue
        if gov_type == "unmeshed":
            continue

        eligible = set(upstream) | set(downstream)
        ranking = substation_ranking(graph, paths["hubs"], obs)

        def split_candidates():
            tried = set()
            for substation in ranking:
                if (
                    substation not in eligible
                    or substation in tried
                    or obs.time_before_cooldown_sub[substation] > 0
                ):
                    continue
                tried.add(substation)
                try:
                    yield (
                        f"hub split @ sub {substation}",
                        action_from_split(
                            env,
                            substation,
                            split(
                                env,
                                obs,
                                graph,
                                substation,
                                upstream,
                                downstream,
                            ),
                        ),
                    )
                except ValueError:
                    continue
            series_nodes = {node for edge in paths["series_path"] for node in edge}
            remaining = sorted(
                (series_nodes & eligible) - tried,
                key=lambda substation: -int(type(env).sub_info[substation]),
            )
            for substation in remaining:
                if obs.time_before_cooldown_sub[substation] > 0:
                    continue
                try:
                    yield (
                        f"Path_c split @ sub {substation}",
                        action_from_split(
                            env,
                            substation,
                            split(
                                env,
                                obs,
                                graph,
                                substation,
                                upstream,
                                downstream,
                            ),
                        ),
                    )
                except ValueError:
                    continue

        def switches():
            if not ALLOW_BRANCH_DISCONNECT:
                return
            for line in path_c_branch_switch_candidates(env, obs, paths):
                maintenance_time = int(obs.time_next_maintenance[int(line)])
                if (
                    int(line) not in blocked_lines
                    and not (0 <= maintenance_time <= 12)
                ):
                    yield (
                        f"branch switch line {line}",
                        env.action_space(
                            {"set_line_status": [(int(line), -1)]}
                        ),
                    )

        def relief(node_set, prefix):
            for label, action in relief_candidates(env, obs, node_set):
                status = np.asarray(action.set_line_status, dtype=int)
                reconnects = set(np.flatnonzero(status == 1).astype(int))
                unavailable = {
                    line
                    for line in reconnects
                    if maintenance_blocks_reconnection(obs, line)
                }
                if reconnects & blocked_lines or unavailable:
                    continue
                yield f"{prefix}: {label}", action

        parallel_nodes = {node for edge in paths["parallel_path"] for node in edge}
        other_nodes = (set(upstream) | set(downstream)) - parallel_nodes
        candidates = itertools.chain(
            split_candidates(),
            switches(),
            relief(parallel_nodes, "Path_// relief"),
            relief(other_nodes, "Da/Ua relief"),
        )
        candidates = (
            (label, action)
            for label, action in candidates
            if topology_protection_safe(env, obs, action)[0]
        )
        for label, proposed in candidates:
            signature = str(proposed.as_dict())
            if signature in seen:
                continue
            seen.add(signature)
            action, accepted_label, rho = _select_best(
                env,
                obs,
                [(label, proposed)],
                time_step=0,
                verbose=False,
            )
            if action is not None:
                ranked_candidates.append((float(rho), accepted_label, action))
    ranked_candidates.sort(key=lambda item: item[0])
    return [(action, label) for _rho, label, action in ranked_candidates]


def _reconnects_offline_line(env, obs, action):
    cls = type(env)
    set_bus = np.asarray(action.set_bus, dtype=int)
    status = np.asarray(obs.line_status, dtype=bool)
    set_status = np.asarray(action.set_line_status, dtype=int)
    for line in np.flatnonzero(~status):
        if set_status[line] == 1:
            continue
        if (
            set_bus[int(cls.line_or_pos_topo_vect[line])] > 0
            or set_bus[int(cls.line_ex_pos_topo_vect[line])] > 0
        ):
            return True
    return False


def _topology_touched(action):
    return bool(
        np.any(np.asarray(action.set_bus, dtype=int) != 0)
        or np.any(np.asarray(action.set_line_status, dtype=int) != 0)
    )


def protected_bridge_relief_action(env, obs, bridge_lines):
    lines = {
        int(line)
        for line in bridge_lines
        if 0 <= int(line) < type(env).n_line
        and bool(obs.line_status[int(line)])
    }
    if not lines:
        return None, None, None

    hold = validate_action(
        env, obs, env.action_space({}), require_imminent_reset=False
    )
    hold_priority_rho = float("inf")
    if hold["ok"]:
        forecast = hold["forecast_obs"]
        hold_priority_rho = max(float(forecast.rho[line]) for line in lines)

    cls = type(env)
    substations = set()
    for line in lines:
        substations.add(int(cls.line_or_to_subid[line]))
        substations.add(int(cls.line_ex_to_subid[line]))

    baseline_n2 = structural_double_outage_failures(env, obs)
    seen = set()
    accepted = []
    for substation in sorted(substations):
        if int(obs.time_before_cooldown_sub[substation]) > 0:
            continue
        try:
            actions = list(
                env.action_space.get_all_unitary_topologies_set(
                    env.action_space, substation
                )
            )
        except Exception:
            continue
        if len(actions) > MAX_BRIDGE_TOPOLOGY_CANDIDATES:
            indices = np.linspace(
                0,
                len(actions) - 1,
                MAX_BRIDGE_TOPOLOGY_CANDIDATES,
                dtype=int,
            )
            actions = [actions[int(index)] for index in np.unique(indices)]
        for action in actions:
            signature = str(action.as_dict())
            if signature in seen or not _topology_touched(action):
                continue
            seen.add(signature)
            if _reconnects_offline_line(env, obs, action):
                continue
            safe, structural = topology_protection_safe(env, obs, action)
            if not safe:
                continue
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
            if priority_rho >= 1.0 - 1e-6:
                continue
            if priority_rho >= hold_priority_rho - 1e-5:
                continue
            key = (priority_rho, verdict["reports"][-1]["rho"])
            verdict["structural"] = structural
            accepted.append((key, action, substation, verdict))
    for _key, action, substation, verdict in sorted(
        accepted, key=lambda item: item[0]
    ):
        candidate_n2 = structural_double_outage_failures(env, obs, action)
        if candidate_n2 - baseline_n2:
            continue
        return action, f"critical bridge split @ sub {substation}", verdict
    return None, None, None


TEMPLATE_BRIDGE_BUDGET_S = 25.0


def protected_contingency_bridge_action(
    env, obs, contingency_report, prefix_action=None, budget_s=TEMPLATE_BRIDGE_BUDGET_S,
    return_all=False,
):
    outage = contingency_report.get("line")
    bridge = contingency_report.get("bridge_line")
    bridge_rho = float(contingency_report.get("bridge_rho", 0.0))
    if bridge is not None and bridge_rho > 0.95:
        monitored = int(bridge)
        baseline_monitored_rho = bridge_rho
        threat_kind = "bridge"
    else:
        monitored = contingency_report.get("max_rho_line")
        baseline_monitored_rho = float(contingency_report.get("rho", float("inf")))
        threat_kind = "overload"
    if outage is None or monitored is None:
        return None, None, None
    outage, monitored = int(outage), int(monitored)
    cls = type(env)
    substations = {
        int(cls.line_or_to_subid[outage]),
        int(cls.line_ex_to_subid[outage]),
        int(cls.line_or_to_subid[monitored]),
        int(cls.line_ex_to_subid[monitored]),
    }
    accepted = []
    horizons = validation_horizons(env, obs)
    baseline_single_failures = structural_single_outage_failures(env, obs)
    seen = set()
    import time as _time
    _started = _time.perf_counter()
    for substation in sorted(substations):
        if _time.perf_counter() - _started > budget_s:
            break
        if (
            int(obs.time_before_cooldown_sub[substation]) > 0
            or int(cls.sub_info[substation]) > 8
        ):
            continue
        try:
            actions = list(
                env.action_space.get_all_unitary_topologies_set(
                    env.action_space, substation
                )
            )
        except Exception:
            continue
        for action in actions:
            if _time.perf_counter() - _started > budget_s:
                break
            proposed = action if prefix_action is None else prefix_action + action
            signature = str(proposed.as_dict())
            if signature in seen or not _topology_touched(proposed):
                continue
            seen.add(signature)
            if _reconnects_offline_line(env, obs, proposed):
                continue
            safe, structural = topology_protection_safe(env, obs, proposed)
            if not safe:
                continue
            candidate_single_failures = structural_single_outage_failures(
                env, obs, proposed
            )
            if candidate_single_failures - baseline_single_failures:
                continue
            base_verdict = validate_action(
                env,
                obs,
                proposed,
                rho_cap=0.98,
                require_imminent_reset=True,
                horizons=horizons,
            )
            if not base_verdict["ok"]:
                continue
            forced = proposed + env.action_space(
                {"set_line_status": [(outage, -1)]}
            )
            sim_obs, _reward, done, info = obs.simulate(forced, time_step=1)
            contingency_rho = maximum_rho(sim_obs) if not done else float("inf")
            if not valid_transition(done, info, contingency_rho):
                continue
            if not bool(sim_obs.line_status[monitored]):
                candidate_monitored_rho = 0.0
            else:
                candidate_monitored_rho = float(sim_obs.rho[monitored])
            target = 0.98 if threat_kind == "bridge" else 1.0
            if candidate_monitored_rho >= min(
                target, baseline_monitored_rho - 0.01
            ):
                continue
            key = (
                candidate_monitored_rho,
                contingency_rho,
                base_verdict["reports"][-1]["rho"],
            )
            base_verdict["structural"] = structural
            accepted.append((key, proposed, substation, base_verdict))
    if not accepted:
        return None, None, None

    partner_ranking = []
    for partner in map(int, getattr(env, "alertable_line_ids", ())):
        if _time.perf_counter() - _started > 2 * budget_s:
            break
        if partner == outage or not bool(obs.line_status[partner]):
            continue
        forced_pair = env.action_space(
            {"set_line_status": [(outage, -1), (partner, -1)]}
        )
        pair_obs, _reward, done, info = obs.simulate(
            forced_pair, time_step=1
        )
        pair_rho = maximum_rho(pair_obs) if not done else float("inf")
        if not valid_transition(done, info, pair_rho):
            continue
        _connected, pair_bridges = topology_bridges(env, pair_obs)
        pair_bridge_rho = max(
            (
                float(pair_obs.rho[line])
                for line in pair_bridges
                if bool(pair_obs.line_status[line])
                and np.isfinite(float(pair_obs.rho[line]))
            ),
            default=0.0,
        )
        partner_ranking.append((pair_bridge_rho, pair_rho, partner))
    partners = [
        item[2]
        for item in sorted(partner_ranking, reverse=True)[
            :MAX_SECOND_OUTAGE_CANDIDATES
        ]
    ]

    robust = []
    for key, proposed, substation, verdict in sorted(
        accepted, key=lambda item: item[0]
    ):
        pair_worst_rho = 0.0
        pair_worst_bridge = 0.0
        invalid_pairs = 0
        for partner in partners:
            forced_pair = proposed + env.action_space(
                {
                    "set_line_status": [
                        (outage, -1),
                        (partner, -1),
                    ]
                }
            )
            pair_obs, _reward, done, info = obs.simulate(
                forced_pair, time_step=1
            )
            pair_rho = maximum_rho(pair_obs) if not done else float("inf")
            if not valid_transition(done, info, pair_rho):
                invalid_pairs += 1
                continue
            _connected, pair_bridges = topology_bridges(env, pair_obs)
            bridge_values = [
                float(pair_obs.rho[line])
                for line in pair_bridges
                if bool(pair_obs.line_status[line])
                and np.isfinite(float(pair_obs.rho[line]))
            ]
            pair_bridge_rho = max(bridge_values, default=0.0)
            pair_worst_rho = max(pair_worst_rho, pair_rho)
            pair_worst_bridge = max(pair_worst_bridge, pair_bridge_rho)
        robust.append(
            (
                (invalid_pairs, pair_worst_bridge, pair_worst_rho) + key,
                proposed,
                substation,
                verdict,
            )
        )
    if not robust:
        return [] if return_all else (None, None, None)
    if return_all:
        return [
            (item[1], f"contingency {outage} {threat_kind} {monitored} split @ sub {item[2]}", item[3])
            for item in sorted(robust, key=lambda item: item[0])
        ]
    selected = min(robust, key=lambda item: item[0])
    return (
        selected[1],
        f"contingency {outage} {threat_kind} {monitored} split @ sub {selected[2]}",
        selected[3],
    )


def _candidate_gate(env, obs, action, n1_rho_limit=1.20):
    safe, structural = topology_protection_safe(env, obs, action)
    if not safe:
        return False, "critical_bridge_or_island", None

    baseline_double_failures = structural_double_outage_failures(env, obs)
    candidate_double_failures = structural_double_outage_failures(
        env, obs, action
    )
    new_double_failures = candidate_double_failures - baseline_double_failures
    if new_double_failures:
        return False, "new_opponent_or_maintenance_n2_cut", {
            "structural": structural,
            "new_double_failures": sorted(new_double_failures),
        }

    verdict = validate_action(
        env,
        obs,
        action,
        require_imminent_reset=True,
        require_critical_improvement=bool(dynamic_security_set(env, obs)["critical"]),
    )
    if not verdict["ok"]:
        return False, verdict["reports"][-1]["reason"], verdict

    if verdict["reports"][0]["rho"] >= maximum_rho(obs) - 1e-5:
        return False, "no_global_rho_reduction", verdict

    hold = validate_action(
        env, obs, env.action_space({}), require_imminent_reset=False
    )
    if (
        hold["ok"]
        and verdict["reports"][-1]["rho"]
        > hold["reports"][-1]["rho"] + 0.005
    ):
        return False, "next_step_worse_than_hold", verdict

    contingency = screen_dynamic_contingencies(
        env, obs, action, rho_limit=n1_rho_limit
    )
    baseline = screen_dynamic_contingencies(
        env, obs, env.action_space({}), rho_limit=n1_rho_limit
    )
    if not contingency["passed"] and not contingency_not_worse(
        contingency, baseline, rho_tolerance=0.10
    ):
        return False, "dynamic_contingency_failure", verdict
    verdict["structural"] = structural
    verdict["contingency"] = contingency
    verdict["baseline_contingency"] = baseline
    verdict["new_double_failures"] = []
    return True, "validated", verdict


def protected_lodf_action(
    env,
    obs,
    congested_line,
    managed_records,
    *,
    n1_rho_limit=1.20,
    blocked_lines=(),
):
    candidates = _filtered_lodf_pipeline(
        int(congested_line), env, obs, blocked_lines
    )
    if not candidates:
        return None, None, None, "no_lodf_candidate"
    last_reason, last_verdict = "all_candidates_rejected", None
    for action, label in candidates:
        if not _topology_touched(action):
            continue
        if len(managed_records) < MAX_MANAGED_TOPOLOGY:
            ok, reason, verdict = _candidate_gate(
                env, obs, action, n1_rho_limit=n1_rho_limit
            )
            if ok:
                return action, label, verdict, None
            last_reason, last_verdict = reason, verdict
            continue

        replacements = [
            (index, record)
            for index, record in enumerate(managed_records)
            if cooldown_cleared(obs, record)
        ]
        accepted = []
        for index, record in replacements:
            restore_action = build_restore_action(env, record, obs=obs)
            if _same_action_effect(restore_action, action):
                ok, reason, verdict = _candidate_gate(
                    env, obs, action, n1_rho_limit=n1_rho_limit
                )
                if ok:
                    return (
                        action,
                        f"restore_managed:{label}",
                        verdict,
                        index,
                    )
                last_reason, last_verdict = reason, verdict
                continue
            compound = restore_action + action
            ok, reason, verdict = _candidate_gate(
                env, obs, compound, n1_rho_limit=n1_rho_limit
            )
            if ok:
                accepted.append(
                    (verdict["reports"][-1]["rho"], index, compound, verdict)
                )
            else:
                last_reason, last_verdict = reason, verdict
        if accepted:
            _, replaced_index, candidate, verdict = min(
                accepted, key=lambda item: item[0]
            )
            return (
                candidate,
                f"replace_then_{label}",
                verdict,
                replaced_index,
            )
        last_reason = "managed_topology_limit_or_replacement_rejected"
    return None, None, last_verdict, last_reason
