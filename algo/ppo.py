from dataclasses import dataclass, field

import torch


# PPO-Clip: https://arxiv.org/abs/1707.06347 (Eq. 7, Algorithm 1)
@dataclass
class Config:
    buffer_size: int = 2048  # consecutive transitions per rollout
    batch_size: int = 64
    epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    hidden_dims: list[int] = field(default_factory=lambda:[64, 64])
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4

@dataclass
class Context:
    """Per-stream state; training and evaluation use independent instances."""
    evaluate: bool = False
    observation: torch.Tensor|None = None
    action: torch.Tensor|None = None
    log_prob: torch.Tensor|None = None

def build_mlp(dims: list[int], activation: type[torch.nn.Module] = torch.nn.Tanh) -> torch.nn.Sequential:
    layers = [torch.nn.Linear(dims[0], dims[1])]
    for i in range(1, len(dims) - 1):
        layers += [activation(), torch.nn.Linear(dims[i], dims[i+1])]
    return torch.nn.Sequential(*layers)

class Critic(torch.nn.Module):
    def __init__(self, observation_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.network = build_mlp([observation_dim, *hidden_dims, 1])
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)

class Actor(torch.nn.Module):
    action_scale: torch.Tensor
    action_bias: torch.Tensor
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: list[int], action_scale: torch.Tensor, action_bias: torch.Tensor):
        super().__init__()
        self.network = build_mlp([observation_dim, *hidden_dims, action_dim])
        self.logstd = torch.nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)
    def distribution(self, observation: torch.Tensor) -> torch.distributions.normal.Normal:
        mean = self.action_scale * self.network(observation) + self.action_bias
        std = self.action_scale * self.logstd.exp()
        return torch.distributions.normal.Normal(mean, std.expand_as(mean))
    def predict(self, observation: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """Return raw Gaussian actions; clip only when applying them to the environment."""
        normal = self.distribution(observation)
        return normal.mean if deterministic else normal.sample()
    def log_prob(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.distribution(observation).log_prob(action).sum(dim=-1)
    def entropy(self, observation: torch.Tensor) -> torch.Tensor:
        return self.distribution(observation).entropy().sum(dim=-1)

class Buffer:
    def __init__(self, buffer_size: int, observation_dim: int, action_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.head = 0
        self.observations = torch.zeros((buffer_size, observation_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((buffer_size, observation_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.discounts = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.continuations = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.device = device
    def add(self, observation: torch.Tensor, action: torch.Tensor, next_observation: torch.Tensor, reward: torch.Tensor, discount: torch.Tensor, continuation: torch.Tensor, log_prob: torch.Tensor):
        if self.head == self.buffer_size:
            raise RuntimeError("Rollout buffer is full; call learn() before collecting more transitions.")
        self.observations[self.head] = observation
        self.actions[self.head] = action
        self.next_observations[self.head] = next_observation
        self.rewards[self.head] = reward
        self.discounts[self.head] = discount
        self.continuations[self.head] = continuation
        self.log_probs[self.head] = log_prob
        self.head += 1
    def get(self) -> tuple[torch.Tensor, ...]:
        return (self.observations[:self.head], self.actions[:self.head], self.next_observations[:self.head], self.rewards[:self.head], self.discounts[:self.head], self.continuations[:self.head], self.log_probs[:self.head])
    def clear(self) -> None:
        self.head = 0

class Agent:
    """Continuous-action PPO for one environment with sequential transitions."""
    def __init__(self, observation_spec: dict, action_spec: dict, config: dict|None = None, *, device: str = "cpu"):
        self.config = Config(**config) if config else Config()
        if min(self.config.buffer_size, self.config.batch_size, self.config.epochs) < 1:
            raise ValueError("buffer_size, batch_size, and epochs must be positive.")
        self.device = torch.device(device)
        observation_dim = observation_spec["ndim"]
        action_dim = action_spec["ndim"]
        action_low = torch.as_tensor(action_spec["low"], dtype=torch.float32, device=self.device)
        action_high = torch.as_tensor(action_spec["high"], dtype=torch.float32, device=self.device)
        action_scale = 0.5 * (action_high - action_low)
        action_bias = 0.5 * (action_high + action_low)
        if not bool(torch.all(torch.isfinite(action_scale) & torch.isfinite(action_bias) & (action_scale > 0))):
            raise ValueError("Continuous actions require finite bounds with low < high.")
        # construct models
        self.actor = Actor(observation_dim, action_dim, self.config.hidden_dims, action_scale, action_bias).to(self.device)
        self.critic = Critic(observation_dim, self.config.hidden_dims).to(self.device)
        # construct optims
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)
        # construct buffer
        self.buffer = Buffer(self.config.buffer_size, observation_dim, action_dim, device=self.device)

    def create_context(self, *, evaluate: bool = False) -> Context:
        return Context(evaluate=evaluate)

    @torch.no_grad()
    def reset(self, ctx: Context, observation: torch.Tensor) -> None:
        """Reset stream state in place, preserving models, optimizers, and the rollout."""
        ctx.observation = observation.detach().clone()
        ctx.action = None
        ctx.log_prob = None

    @torch.no_grad()
    def sample(self, ctx: Context, observation: torch.Tensor) -> torch.Tensor:
        """Sample an unclipped action and retain training metadata until collect()."""
        if ctx.evaluate:
            ctx.observation = observation.detach().clone()
            return self.actor.predict(ctx.observation, deterministic=True)
        if ctx.action is not None:
            raise RuntimeError("Collect the previous sample before sampling again with this context.")
        if self.buffer.head == self.buffer.buffer_size:
            raise RuntimeError("Rollout buffer is full; call learn() before sampling more training actions.")
        ctx.observation = observation.detach().clone()
        normal = self.actor.distribution(ctx.observation)
        action = normal.sample()
        ctx.action = action.clone()
        ctx.log_prob = normal.log_prob(action).sum(dim=-1)
        return action

    @torch.no_grad()
    def collect(self, ctx: Context, next_observation: torch.Tensor, reward: torch.Tensor, termination: torch.Tensor, truncation: torch.Tensor) -> bool:
        """Consume the pending sample; next_observation must precede any environment reset."""
        if ctx.evaluate:
            raise RuntimeError("Evaluation contexts do not collect training transitions.")
        if ctx.observation is None or ctx.action is None or ctx.log_prob is None:
            raise RuntimeError("Sample an action with this context before collecting a transition.")
        discount = 1 - termination.to(dtype=torch.float32)
        continuation = 1 - (termination.bool() | truncation.bool()).to(dtype=torch.float32)
        self.buffer.add(ctx.observation, ctx.action, next_observation, reward, discount, continuation, ctx.log_prob)
        ctx.observation = next_observation.detach().clone()
        ctx.action = None
        ctx.log_prob = None
        return bool(self.buffer.head == self.buffer.buffer_size)

    def learn(self) -> dict[str, float]:
        """Update on all collected transitions, including a partial rollout, then clear it."""
        if self.buffer.head == 0:
            raise RuntimeError("Rollout buffer is empty; collect transitions before calling learn().")
        with torch.no_grad():
            observations, actions, next_observations, rewards, discounts, continuations, old_log_probs = self.buffer.get()
            values = self.critic(observations)
            next_values = self.critic(next_observations)
            # truncated GAE and TD residuals (PPO Eq. 11-12)
            deltas = rewards + self.config.gamma * discounts * next_values - values
            advantages = torch.zeros_like(rewards)
            gae = torch.zeros((), dtype=torch.float32, device=self.device)
            for t in reversed(range(self.buffer.head)):
                gae = deltas[t] + self.config.gamma * self.config.gae_lambda * continuations[t] * gae
                advantages[t] = gae
            returns = advantages + values
            if self.config.normalize_advantage and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        losses = {"actor": 0.0, "critic": 0.0, "entropy": 0.0}
        samples = 0
        # multiple epochs of shuffled minibatch updates (Algorithm 1)
        for epoch in range(self.config.epochs):
            indices = torch.randperm(self.buffer.head, device=self.device)
            for start in range(0, self.buffer.head, self.config.batch_size):
                batch = indices[start:start + self.config.batch_size]
                # update actor with the clipped surrogate objective (Eq. 7)
                log_probs = self.actor.log_prob(observations[batch], actions[batch])
                ratios = (log_probs - old_log_probs[batch]).exp()
                surrogate = ratios * advantages[batch]
                clipped_surrogate = ratios.clamp(1 - self.config.clip_range, 1 + self.config.clip_range) * advantages[batch]
                actor_loss = -torch.min(surrogate, clipped_surrogate).mean()
                entropy = self.actor.entropy(observations[batch]).mean()
                self.optim_actor.zero_grad()
                (actor_loss - self.config.entropy_coef * entropy).backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.optim_actor.step()
                # update the state-value critic with fixed rollout returns
                values_pi = self.critic(observations[batch])
                critic_loss = 0.5 * torch.nn.functional.mse_loss(values_pi, returns[batch])
                self.optim_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.optim_critic.step()
                losses["actor"] += actor_loss.item() * batch.numel()
                losses["critic"] += critic_loss.item() * batch.numel()
                losses["entropy"] += entropy.item() * batch.numel()
                samples += batch.numel()
        self.buffer.clear()
        return {name: loss / samples for name, loss in losses.items()}
