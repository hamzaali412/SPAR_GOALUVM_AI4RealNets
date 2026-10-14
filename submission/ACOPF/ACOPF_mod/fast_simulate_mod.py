
import copy
import hashlib
import inspect
import os

from grid2op.Observation.baseObservation import BaseObservation
from grid2op.Exceptions import NoForecastAvailable

_EXPECTED_SOURCE_SHA1 = {"a05e5145c3f53b47d219642eb8ab8faa874b12b6"}
_ORIGINAL = BaseObservation.simulate
INSTALLED = False

# Per-decision memo: the same observation simulated with the same action vector and
# horizon returns the same result (the forecast environment is deterministic).  The
# observation object itself is kept in the value so that a recycled ``id`` can never
# alias a different observation.  Cleared by ``reset_memo`` at the start of each decision.
MEMO_ENABLED = os.environ.get("ACOPF_SIMULATE_MEMO", "1") == "1"
_MEMO = {}
_MEMO_MAX = 50000
_MEMO_STATS = {"calls": 0, "hits": 0}


def reset_memo():
    _MEMO.clear()


def memo_stats():
    return dict(_MEMO_STATS)


def simulate_fresh(obs, action, time_step=1):
    """Run the forecast for real (no memo lookup) so that callers relying on the side
    effect of the simulation -- the forecast environment's backend holding the
    simulated grid -- see the state they expect."""
    global MEMO_ENABLED
    saved = MEMO_ENABLED
    MEMO_ENABLED = False
    try:
        return obs.simulate(action, time_step=time_step)
    finally:
        MEMO_ENABLED = saved


def _fast_simulate(self, action, time_step=1):
    if self.action_helper is None:
        raise NoForecastAvailable(
            "No forecasts are available for this instance of BaseObservation "
            "(no action_space and no simulated environment are set)."
        )
    if self._obs_env is None:
        raise NoForecastAvailable(
            'This observation has no "environment used for simulation" (_obs_env).'
        )
    if time_step < 0:
        raise NoForecastAvailable("Impossible to 'forecast' in the past at the moment.")
    if time_step >= len(self._forecasted_inj):
        raise NoForecastAvailable(
            f"Forecast for {time_step} timestep(s) ahead is not possible "
            f"with your chronics, max length {len(self._forecasted_inj) - 1} steps ahead."
        )
    memo_key = None
    if MEMO_ENABLED:
        memo_key = (id(self), int(self.current_step), action.to_vect().tobytes(), int(time_step))
        _MEMO_STATS["calls"] += 1
        hit = _MEMO.get(memo_key)
        if hit is not None and hit[0] is self:
            _MEMO_STATS["hits"] += 1
            return hit[1]
    if time_step not in self._forecasted_grid_act:
        timestamp, inj_forecasted = self._forecasted_inj[time_step]
        self._forecasted_grid_act[time_step] = {
            "timestamp": timestamp,
            "inj_action": self.action_helper(inj_forecasted),
        }
    timestamp = self._forecasted_grid_act[time_step]["timestamp"]
    inj_action = self._forecasted_grid_act[time_step]["inj_action"]
    self._obs_env.init(inj_action, time_stamp=timestamp, obs=self, time_step=time_step)
    raw, *rest = self._obs_env.simulate(action)
    raw_prev = raw._prev_conn
    raw._prev_conn = None
    try:
        sim_obs = copy.deepcopy(raw)
    finally:
        raw._prev_conn = raw_prev
    sim_obs._prev_conn = self._prev_conn
    if self._forecasted_inj:
        sim_obs.action_helper = self.action_helper
        sim_obs._obs_env = self._obs_env
        sim_obs._forecasted_inj = self._forecasted_inj[1:]
        sim_obs._update_internal_env_params(self._obs_env)
    result = (sim_obs, *rest)
    if memo_key is not None:
        if len(_MEMO) >= _MEMO_MAX:
            _MEMO.clear()
        _MEMO[memo_key] = (self, result)
    return result


def source_sha1():
    return hashlib.sha1(inspect.getsource(_ORIGINAL).encode()).hexdigest()


def install():
    global INSTALLED
    if INSTALLED:
        return True
    if os.environ.get("ACOPF_FAST_SIMULATE", "1") != "1":
        return False
    if source_sha1() not in _EXPECTED_SOURCE_SHA1:
        return False
    BaseObservation.simulate = _fast_simulate
    INSTALLED = True
    return True


def uninstall():
    global INSTALLED
    BaseObservation.simulate = _ORIGINAL
    INSTALLED = False
