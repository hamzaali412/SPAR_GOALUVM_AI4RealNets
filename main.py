import os
import sys
import time

import grid2op
from grid2op.Agent import DoNothingAgent
from lightsim2grid import LightSimBackend

AGENT = "trad"
ENV = "l2rpn_idf_2023"
DIFFICULTY = "competition"
SCENARIO = 0
SEED = 42 + SCENARIO
MAX_STEPS = None
VERBOSE = False

SUBMISSION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission")
sys.path.insert(0, SUBMISSION)
os.environ["ACOPF_QUIET"] = "0" if VERBOSE else "1"

env = grid2op.make(ENV, backend=LightSimBackend(), difficulty=DIFFICULTY)

if AGENT == "trad":
    from make_agent import make_agent
    agent = make_agent(env, SUBMISSION)
else:
    agent = DoNothingAgent(env.action_space)

env.seed(SEED)
env.set_id(SCENARIO)
obs = env.reset()
agent.reset(obs)

horizon = int(env.max_episode_duration())
limit = horizon if MAX_STEPS is None else min(MAX_STEPS, horizon)
print(f"agent={AGENT} scenario={SCENARIO} seed={SEED} steps={limit}")

reward, done, total_reward, steps = 0.0, False, 0.0, 0
started = time.perf_counter()
while steps < limit and not done:
    action = agent.act(obs, reward, done)
    obs, reward, done, info = env.step(action)
    total_reward += float(reward)
    steps += 1
    stars = "*" * (40 * steps // limit)
    print(f"\r[{stars:<40}] step {steps}/{limit}  max rho {float(obs.rho.max()):.3f}", end="", flush=True)

elapsed = time.perf_counter() - started
game_over = done and steps < horizon
print(f"\nplayed {steps}/{limit} steps: {'GAME OVER' if game_over else 'survived'}")
print(f"total reward {total_reward:.2f}")
print(f"time {elapsed:.1f} s ({elapsed / max(steps, 1):.3f} s/step)")
if game_over and info.get("exception"):
    print(f"ended by: {info['exception'][0]!r}")
env.close()
