"""Codabench entry point."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ACOPF.ACOPF_mod.make_agent import make_agent as _make_agent  # noqa: E402


_QUIET = os.environ.get("ACOPF_QUIET", "1") == "1"
_SILENCED = set()


def _noop(*_args, **_kwargs):
    return None


def _silence_controller_output():
    """ACOPF_QUIET=1 (default): the controller's diagnostic prints are turned off by shadowing
    ``print`` with a no-op in every ACOPF module (a module global takes precedence over the
    builtin), so nothing else in the process -- the scoring program included -- is affected and
    no control code changes.  Several controller modules are imported lazily inside the step, so
    this runs before every decision and patches whatever is new.  ACOPF_QUIET=0 keeps the logs."""
    if not _QUIET:
        return
    if not _SILENCED:
        # the controller imports several of its modules lazily inside the first decision; import
        # them here first so their prints are shadowed before that decision runs
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
    """Installs the silencer on the instance's ``act`` (delegating to the original bound method), so
    the agent stays the very object the controller built: the runner's ``isinstance(..., BaseAgent)``
    check passes and nothing else about it changes."""
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
