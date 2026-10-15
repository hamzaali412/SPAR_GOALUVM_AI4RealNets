import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ACOPF.ACOPF_mod.make_agent import make_agent as _make_agent


_QUIET = os.environ.get("ACOPF_QUIET", "1") == "1"
_SILENCED = set()


def _noop(*_args, **_kwargs):
    return None


def _silence_controller_output():
    if not _QUIET:
        return
    if not _SILENCED:
        import importlib
        for leaf in ("posture_mod", "emergency_mod", "capacity_mod", "security_mod", "recovery_mod",
                     "pace_mod", "topology_mod", "split_search_mod", "scopf_mod", "interventions",
                     "n1_check", "overload_graph", "obs_network", "live_grid", "fast_simulate_mod",
                     "data_mod", "data_extraction", "ac_lodf", "acopf_setup", "acopf_setup_twostage_mod",
                     "build_ybus"):
            try:
                importlib.import_module(f"ACOPF.ACOPF_mod.{leaf}")
            except Exception:
                pass
    for name, module in list(sys.modules.items()):
        if name in _SILENCED or module is None or not name.startswith("ACOPF"):
            continue
        try:
            setattr(module, "print", _noop)
        except Exception:
            pass
        _SILENCED.add(name)


def _hook_quiet(agent):
    original_act = agent.act

    def act(observation, reward, done):
        _silence_controller_output()
        return original_act(observation, reward, done)

    agent.act = act
    return agent


def make_agent(env, this_directory_path=None):
    agent = _make_agent(env, this_directory_path)
    if not _QUIET:
        return agent
    _silence_controller_output()
    return _hook_quiet(agent)


__all__ = ["make_agent"]
