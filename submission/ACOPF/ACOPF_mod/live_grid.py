from contextlib import contextmanager


@contextmanager
def live_grid(env, obs):
    from ACOPF.ACOPF_mod.obs_network import model_grid

    yield model_grid(obs)
