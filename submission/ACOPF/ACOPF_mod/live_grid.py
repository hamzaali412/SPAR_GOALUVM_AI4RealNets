
from contextlib import contextmanager

import numpy as np

STALE_FLOW_TOLERANCE_MW = 1.0


def _grid_active_flows(grid):
    try:
        lines = np.asarray(grid.get_line_res1()[0], dtype=float)
        transformers = np.asarray(grid.get_trafo_res1()[0], dtype=float)
    except Exception:
        return None
    flows = np.concatenate([lines, transformers])
    return flows if flows.size else None


def grid_is_stale(grid, obs):
    if grid is None:
        return True
    observed = np.asarray(obs.p_or, dtype=float)
    flows = _grid_active_flows(grid)
    if flows is not None and flows.size == observed.size:
        both_finite = np.isfinite(flows) & np.isfinite(observed)
        if both_finite.any():
            drift = float(np.max(np.abs(flows[both_finite] - observed[both_finite])))
            return drift > STALE_FLOW_TOLERANCE_MW
    try:
        from ACOPF.ACOPF_mod.build_ybus import extract_live_branches

        connected = len(extract_live_branches(grid))
        return connected != int(np.asarray(obs.line_status, dtype=bool).sum())
    except Exception:
        return False


@contextmanager
def live_grid(env, obs):
    backend = getattr(env, "backend", None)
    saved = getattr(backend, "_grid", None)
    if not grid_is_stale(saved, obs):
        yield saved
        return

    grid = saved
    try:
        from ACOPF.ACOPF_mod import fast_simulate_mod as _fsm
        _fsm.simulate_fresh(obs, env.action_space({}), time_step=0)
        grid = obs._obs_env.backend._grid
    except Exception:
        grid = saved
    try:
        if backend is not None and grid is not None:
            backend._grid = grid
        yield grid
    finally:
        if backend is not None and saved is not None:
            backend._grid = saved
