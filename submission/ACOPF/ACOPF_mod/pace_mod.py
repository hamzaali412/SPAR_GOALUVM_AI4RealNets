import os
import time

CODABENCH_TOTAL_BUDGET_S = 86400.0
CODABENCH_EPISODES = 208
EPISODE_BUDGET_S = float(
    os.environ.get(
        "ACOPF_EPISODE_BUDGET_S",
        str(0.9 * CODABENCH_TOTAL_BUDGET_S / CODABENCH_EPISODES),
    )
)
BASE_MODE = os.environ.get("ACOPF_BASE_MODE", "economy")
EPISODE_STEPS = 2017
STARTUP_ALLOWANCE_S = float(os.environ.get("ACOPF_PACE_STARTUP_S", "30"))
ECONOMY_RATIO = float(os.environ.get("ACOPF_PACE_ECONOMY_RATIO", "1.0"))
MINIMAL_RATIO = float(os.environ.get("ACOPF_PACE_MINIMAL_RATIO", "1.3"))
_RANK = {"normal": 0, "economy": 1, "minimal": 2}
RECOVER_RATIO = float(os.environ.get("ACOPF_PACE_RECOVER_RATIO", "0.85"))

ECONOMY_ACTIVATION_RHO = float(os.environ.get("ACOPF_ECONOMY_ACTIVATION_RHO", "0.98"))
ECONOMY_SCREEN_INTERVAL = int(os.environ.get("ACOPF_ECONOMY_SCREEN_INTERVAL", "12"))
FORCE_MODE = os.environ.get("ACOPF_PACE_MODE")

_STATE = {"spent": 0.0, "steps": 0, "mode": "normal", "by_mode": {}, "budget": EPISODE_BUDGET_S}
GLOBAL_BUDGETING = os.environ.get("ACOPF_GLOBAL_BUDGET", "1") == "1"
BUDGET_SHARE = float(os.environ.get("ACOPF_BUDGET_SHARE", "0.75"))
EPISODE_BUDGET_MIN_S = float(os.environ.get("ACOPF_EPISODE_BUDGET_MIN_S", "250"))
EPISODE_BUDGET_MAX_S = float(os.environ.get("ACOPF_EPISODE_BUDGET_MAX_S", "700"))
_GLOBAL = {"spent": 0.0, "episodes": 0}


def _episode_budget():
    if not GLOBAL_BUDGETING or "ACOPF_EPISODE_BUDGET_S" in os.environ:
        return EPISODE_BUDGET_S
    remaining_eps = max(CODABENCH_EPISODES - _GLOBAL["episodes"], 1)
    remaining_s = BUDGET_SHARE * CODABENCH_TOTAL_BUDGET_S - _GLOBAL["spent"]
    return float(min(EPISODE_BUDGET_MAX_S, max(EPISODE_BUDGET_MIN_S, remaining_s / remaining_eps)))


def reset():
    if _STATE["steps"] > 0:
        _GLOBAL["spent"] += _STATE["spent"]
        _GLOBAL["episodes"] += 1
    _STATE["spent"] = 0.0
    _STATE["steps"] = 0
    _STATE["mode"] = BASE_MODE
    _STATE["by_mode"] = {}
    _STATE["budget"] = _episode_budget()


def record(elapsed_s):
    _STATE["spent"] += float(elapsed_s)
    _STATE["steps"] += 1
    _STATE["by_mode"][_STATE["mode"]] = _STATE["by_mode"].get(_STATE["mode"], 0.0) + float(elapsed_s)


def allowed(step):
    budget = _STATE["budget"]
    return min(budget, STARTUP_ALLOWANCE_S + budget * (int(step) + 1) / EPISODE_STEPS)


def mode(step):
    if FORCE_MODE in ("normal", "economy", "minimal"):
        _STATE["mode"] = FORCE_MODE
        return FORCE_MODE
    ratio = _STATE["spent"] / max(allowed(step), 1e-9)
    current = _STATE["mode"]
    if ratio >= MINIMAL_RATIO:
        new = "minimal"
    elif ratio >= ECONOMY_RATIO:
        new = "economy" if current != "minimal" or ratio < MINIMAL_RATIO * RECOVER_RATIO else "minimal"
    elif ratio < ECONOMY_RATIO * RECOVER_RATIO:
        new = "normal"
    else:
        new = current if current != "minimal" else "economy"
    if _RANK[new] < _RANK[BASE_MODE]:
        new = BASE_MODE
    _STATE["mode"] = new
    return new


def spent():
    return _STATE["spent"]


def summary():
    parts = ", ".join(f"{k}={v:.0f}s" for k, v in sorted(_STATE["by_mode"].items()))
    return (
        f"agent_time={_STATE['spent']:.0f}s over {_STATE['steps']} steps "
        f"(budget {_STATE['budget']:.0f}s, evaluation so far {_GLOBAL['spent']:.0f}s over {_GLOBAL['episodes']} episodes; {parts})"
    )
