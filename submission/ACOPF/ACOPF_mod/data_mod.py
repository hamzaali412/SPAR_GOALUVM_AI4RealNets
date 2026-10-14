
import numpy as np

from ACOPF.ACOPF_mod.data_extraction import extract_grid2op_data


def extract_grid2op_data_mod(env, obs, grid_obs=None):
    data = extract_grid2op_data(env, obs, grid_obs=grid_obs)
    current_net = float(np.sum(obs.load_p) - np.sum(obs.gen_p))
    try:
        forecast_gen, _forecast_v, forecast_load, _forecast_q = (
            obs.get_forecasted_inj(1)
        )
        forecast_gen = np.asarray(forecast_gen, dtype=float)
        forecast_load = np.asarray(forecast_load, dtype=float)
        if not np.all(np.isfinite(forecast_gen)) or not np.all(
            np.isfinite(forecast_load)
        ):
            raise ValueError("non-finite one-step forecast")
        forecast_net = float(np.sum(forecast_load) - np.sum(forecast_gen))
    except Exception:
        forecast_net = current_net

    change = forecast_net - current_net
    data.forecast_net_change_mw = change
    data.forecast_up_requirement_mw = max(0.0, change)
    data.forecast_down_requirement_mw = max(0.0, -change)
    return data
