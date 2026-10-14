
import os
import time

import numpy as np
from grid2op.Agent import BaseAgent

from ACOPF.ACOPF_mod.ipopt_bootstrap import bootstrap

IPOPT_BINARY, IPOPT_OK = bootstrap()

from ACOPF.ACOPF_mod import main_try as controller
from ACOPF.ACOPF_mod.interventions import trim_record


STEP_BUDGET_S = float(os.environ.get("ACOPF_STEP_BUDGET_S", "90"))
SLOW_STEPS_BEFORE_DEGRADE = int(os.environ.get("ACOPF_SLOW_STEPS", "5"))
VERBOSE = os.environ.get("ACOPF_VERBOSE", "0") == "1"


class ACOPFAgent(BaseAgent):
    def __init__(self, action_space, env):
        super().__init__(action_space)
        self.env = env
        self._reset_state()

    def _reset_state(self):
        from ACOPF.ACOPF_mod import pace_mod

        pace_mod.reset()
        self.step = 0
        self.managed_records = []
        self.slow_steps = 0
        self.degraded = False

    def reset(self, obs):
        self._reset_state()
        try:
            from ACOPF.ACOPF_mod import emergency_mod

            emergency_mod.reset_episode_state()
        except Exception:
            pass

    def _blocked_lines(self, obs):
        blocked = set()
        try:
            since = np.asarray(obs.time_since_last_attack, dtype=int)
            ids = np.asarray(obs.alertable_line_ids, dtype=int)
            status = np.asarray(obs.line_status, dtype=bool)
            cooldown = np.asarray(obs.time_before_cooldown_line, dtype=int)
        except Exception:
            return blocked
        # v85: a line is blocked only while the opponent's attack on it lasts.  `time_since_last_attack`
        # never returns to -1 once a line has been attacked, so the previous rule ("attacked and still
        # out") kept every attacked line out of the restoration candidates for the rest of the week:
        # on the 832 quiet-death weeks line 110 stayed out 1,220 steps, 125 for 857, 154 for 517 and 81
        # for 464 after 96-step attacks, and the grid failed to the next ramp two to four lines short.
        # After the attack the line is an ordinary offline line: the attack-aware restoration decides,
        # with its risk and need tests, whether re-arming the area is worth the capacity.
        duration = self._attack_duration()
        for position, line in enumerate(ids):
            line = int(line)
            if position >= since.size or line >= status.size:
                continue
            if not status[line] and 0 <= int(since[position]) < duration:
                blocked.add(line)
        return blocked

    def _attack_duration(self):
        try:
            value = int(getattr(self.env, "_opponent_attack_duration", 0) or 0)
        except Exception:
            value = 0
        return value if value > 0 else 96

    def act(self, observation, reward, done):
        if self.degraded:
            self.step += 1
            return self.action_space({})
        started = time.perf_counter()
        try:
            action = self._act_core(observation)
        except Exception as error:
            self.errors = getattr(self, "errors", 0) + 1
            if VERBOSE or self.errors <= 20:
                import traceback
                print(f"[acopf_agent] step {self.step} error: {error!r} :: {traceback.format_exc().strip().splitlines()[-3:]}", flush=True)
            action = self.action_space({})
        elapsed = time.perf_counter() - started
        from ACOPF.ACOPF_mod import pace_mod

        pace_mod.record(elapsed)
        if elapsed > STEP_BUDGET_S:
            self.slow_steps += 1
            if self.slow_steps >= SLOW_STEPS_BEFORE_DEGRADE:
                self.degraded = True
                if VERBOSE:
                    print("[acopf_agent] step budget exceeded; holding")
        self.step += 1
        return action

    def _act_core(self, obs):
        result = controller.choose_action(
            self.env,
            obs,
            self.step,
            self.managed_records,
            self._blocked_lines(obs),
        )
        trim_index = result.get("trim_record_index")
        if trim_index is not None and trim_index < len(self.managed_records):
            if trim_record(self.managed_records[trim_index], drop_topology=True):
                self.managed_records.pop(trim_index)
        replace_index = result.get("replace_index")
        if replace_index is not None and replace_index < len(self.managed_records):
            self.managed_records.pop(replace_index)
        if result.get("record") is not None:
            self.managed_records.append(result["record"])
        if VERBOSE:
            print(f"[acopf_agent] step {self.step} {result['decision']}")
        return result["action"]


def make_agent(env, this_directory_path=None):
    if not IPOPT_OK and os.environ.get("ACOPF_REQUIRE_IPOPT", "1") == "1":
        raise RuntimeError(
            f"IPOPT is not runnable at {IPOPT_BINARY!r}; refusing to build an "
            "agent that would silently lose its dispatch stage. Set "
            "ACOPF_REQUIRE_IPOPT=0 to downgrade this to a warning."
        )
    return ACOPFAgent(env.action_space, env)
