import argparse
import importlib
import random

import numpy as np
import numpy.typing as npt
import torch


def set_seeds(base_seed: int) -> dict[str, int]:
    seed_sequence = np.random.SeedSequence(base_seed)
    python_seed, numpy_seed, torch_seed = [int(seed) for seed in seed_sequence.generate_state(3, dtype=np.uint32)]
    random.seed(python_seed)
    np.random.seed(numpy_seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    return {"python": python_seed, "numpy": numpy_seed, "torch": torch_seed}

def as_tensors(*transition: npt.NDArray[np.float32]|float|bool) -> tuple[torch.Tensor, ...]:
    return tuple(torch.as_tensor(arr, dtype=torch.float32) for arr in transition)

def train(env, agent, total_timesteps: int, seed: int, verbose_freq: int = 10000):
    rng = np.random.default_rng(seed=seed)
    state = env.reset(rng)
    agent_ctx = agent.create_context() if hasattr(agent, "create_context") else None
    if agent_ctx is not None:
        agent.reset(agent_ctx, torch.as_tensor(state.observation, dtype=torch.float32))
    verbose_flag = False
    for t in range(total_timesteps):
        if t % verbose_freq == 0 and not verbose_flag:
            verbose_flag = True
        observation = torch.as_tensor(state.observation, dtype=torch.float32)
        action = agent.sample(agent_ctx, observation) if agent_ctx is not None else agent.predict(observation, evaluate=False)
        env.step(state, action.numpy(), rng)
        transition = as_tensors(state.observation, state.reward, state.termination, state.truncation)
        learnable = agent.collect(agent_ctx, *transition) if agent_ctx is not None else agent.collect(observation, action, *transition)
        losses = agent.learn() if learnable else None
        if verbose_flag and losses:
            print(t, losses)
            verbose_flag = False
        if state.termination or state.truncation:
            state = env.reset(rng)
            if agent_ctx is not None:
                agent.reset(agent_ctx, torch.as_tensor(state.observation, dtype=torch.float32))

def eval(env, agent, eval_episodes: int, eval_timelimit: int, seed: int):
    rng = np.random.default_rng(seed=seed)
    agent_ctx = agent.create_context(evaluate=True) if hasattr(agent, "create_context") else None
    mean_rew = 0.0
    for ep in range(eval_episodes):
        ep_rew = 0.0
        state = env.reset(rng)
        if agent_ctx is not None:
            agent.reset(agent_ctx, torch.as_tensor(state.observation, dtype=torch.float32))
        for t in range(eval_timelimit):
            observation = torch.as_tensor(state.observation, dtype=torch.float32)
            action = agent.sample(agent_ctx, observation) if agent_ctx is not None else agent.predict(observation, evaluate=True)
            env.step(state, action.numpy(), rng)
            ep_rew += state.reward
        mean_rew += ep_rew
    return mean_rew / eval_episodes

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum")
    parser.add_argument("--agent", type=str, default="sac")
    parser.add_argument("--total-timesteps", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-timelimit", type=int, default=200)
    parser.add_argument("--verbose-freq", type=int, default=1000)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=42)
    return parser

def main():
    args = build_parser().parse_args()
    set_seeds(args.global_seed)
    env_collection = importlib.import_module("envs.classic_control")
    agent_module = importlib.import_module("algo."+args.agent)

    env = env_collection.make(args.env)

    observation_spec = {"ndim": env.observation_space.ndim, "high": env.observation_space.high, "low": env.observation_space.low}
    action_spec = {"ndim": env.action_space.ndim, "high": env.action_space.high, "low": env.action_space.low}
    agent = agent_module.Agent(observation_spec, action_spec)

    mean_rew = eval(env, agent, args.eval_episodes, args.eval_timelimit, args.eval_seed)
    print(mean_rew)

    train(env, agent, args.total_timesteps, args.train_seed, args.verbose_freq)

    mean_rew = eval(env, agent, args.eval_episodes, args.eval_timelimit, args.eval_seed)
    print(mean_rew)

if __name__ == "__main__":
    main()
