from dataclasses import dataclass, field
import math

import torch


# TQC: https://proceedings.mlr.press/v119/kuznetsov20a.html (Algorithm 1)
@dataclass
class Config:
    learning_starts: int = 100
    batch_size: int = 256
    buffer_size: int = 1_000_000
    tau: float = 0.005
    gamma: float = 0.99
    hidden_dims: list[int] = field(default_factory=lambda:[256, 256])
    n_critics: int = 5
    n_quantiles: int = 25
    # Drop n_critics times this count from the pooled target quantiles.
    top_quantiles_to_drop_per_critic: int = 2
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4

@dataclass
class Context:
    """Per-stream state; training and evaluation use independent instances."""
    evaluate: bool = False
    observation: torch.Tensor|None = None
    action: torch.Tensor|None = None

def build_mlp(dims: list[int], activation: type[torch.nn.Module] = torch.nn.ReLU) -> torch.nn.Sequential:
    layers = [torch.nn.Linear(dims[0], dims[1])]
    for i in range(1, len(dims) - 1):
        layers += [activation(), torch.nn.Linear(dims[i], dims[i+1])]
    return torch.nn.Sequential(*layers)

def set_requires_grad(module: torch.nn.Module, flag: bool) -> torch.nn.Module:
    for p in module.parameters():
        p.requires_grad_(flag)
    return module

def quantile_huber_loss(quantiles: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Eq. 8, 14: quantiles [batch, critics, quantiles], targets [batch, target atoms]."""
    deltas = targets[:, None, None, :] - quantiles[:, :, :, None]
    abs_deltas = deltas.abs()
    huber_loss = torch.where(abs_deltas <= 1, 0.5 * deltas.square(), abs_deltas - 0.5)
    n_quantiles = quantiles.shape[-1]
    taus = (torch.arange(n_quantiles, dtype=quantiles.dtype, device=quantiles.device) + 0.5) / n_quantiles
    weights = (taus[None, None, :, None] - (deltas.detach() < 0).to(quantiles.dtype)).abs()
    return (weights * huber_loss).mean()

class QuantileQ(torch.nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: list[int], n_critics: int, n_quantiles: int):
        super().__init__()
        self.networks = torch.nn.ModuleList([
            build_mlp([observation_dim + action_dim, *hidden_dims, n_quantiles])
            for _ in range(n_critics)
        ])
    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = torch.cat((observation, action), dim=-1)
        return torch.stack([network(z) for network in self.networks], dim=-2)

class Critic(torch.nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: list[int], n_critics: int, n_quantiles: int):
        super().__init__()
        self.online = QuantileQ(observation_dim, action_dim, hidden_dims, n_critics, n_quantiles)
        self.target = QuantileQ(observation_dim, action_dim, hidden_dims, n_critics, n_quantiles)
        self.target.load_state_dict(self.online.state_dict())
        set_requires_grad(self.target, False).train(False)

class Actor(torch.nn.Module):
    action_scale: torch.Tensor
    action_bias: torch.Tensor
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: list[int], action_scale: torch.Tensor, action_bias: torch.Tensor):
        super().__init__()
        self.network = build_mlp([observation_dim, *hidden_dims, action_dim * 2])
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)
    def distribution(self, observation: torch.Tensor) -> torch.distributions.normal.Normal:
        mean, logstd = self.network(observation).chunk(2, dim=-1)
        return torch.distributions.normal.Normal(mean, torch.clamp(logstd, -20, 2).exp())
    def predict(self, observation: torch.Tensor, deterministic: bool) -> torch.Tensor:
        normal = self.distribution(observation)
        return self.action_scale * torch.tanh(normal.mean if deterministic else normal.sample()) + self.action_bias
    def rsample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized actions and their log density, as in SAC."""
        normal = self.distribution(observation)
        z = normal.rsample()
        actions = torch.tanh(z)
        log_jacobian = 2 * (math.log(2) - z - torch.nn.functional.softplus(-2 * z))
        log_prob = (normal.log_prob(z) - log_jacobian).sum(dim=-1)
        return actions, log_prob

class Buffer:
    def __init__(self, buffer_size: int, observation_dim: int, action_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.head = 0
        self.full = False
        self.observations = torch.zeros((buffer_size, observation_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((buffer_size, observation_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.discounts = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.device = device
    def add(self, observation: torch.Tensor, action: torch.Tensor, next_observation: torch.Tensor, reward: torch.Tensor, discount: torch.Tensor):
        self.observations[self.head] = observation
        self.actions[self.head] = action
        self.next_observations[self.head] = next_observation
        self.rewards[self.head] = reward
        self.discounts[self.head] = discount
        self.head += 1
        if self.head == self.buffer_size:
            self.head = 0
            self.full = True
    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        last = self.buffer_size if self.full else self.head
        indices = torch.randint(0, last, size=(batch_size,), dtype=torch.long, device=self.device)
        return (self.observations[indices], self.actions[indices], self.next_observations[indices], self.rewards[indices], self.discounts[indices])

class Agent:
    def __init__(self, observation_spec: dict, action_spec: dict, config: dict|None = None, *, device: str = "cpu"):
        self.config = Config(**config) if config else Config()
        if min(self.config.n_critics, self.config.n_quantiles, self.config.batch_size, self.config.buffer_size) < 1:
            raise ValueError("n_critics, n_quantiles, batch_size, and buffer_size must be positive.")
        if not 0 <= self.config.top_quantiles_to_drop_per_critic < self.config.n_quantiles:
            raise ValueError("top_quantiles_to_drop_per_critic must be in [0, n_quantiles).")
        self.device = torch.device(device)
        observation_dim = observation_spec["ndim"]
        action_dim = action_spec["ndim"]
        action_low = torch.as_tensor(action_spec["low"], dtype=torch.float32, device=self.device)
        action_high = torch.as_tensor(action_spec["high"], dtype=torch.float32, device=self.device)
        action_scale = 0.5 * (action_high - action_low)
        action_bias = 0.5 * (action_high + action_low)
        if not bool(torch.all(torch.isfinite(action_scale) & torch.isfinite(action_bias) & (action_scale > 0))):
            raise ValueError("Continuous actions require finite bounds with low < high.")
        self.target_entropy = torch.tensor(-action_dim, dtype=torch.float32, device=self.device)
        # construct models
        self.actor = Actor(observation_dim, action_dim, self.config.hidden_dims, action_scale, action_bias).to(self.device)
        self.critic = Critic(observation_dim, action_dim, self.config.hidden_dims, self.config.n_critics, self.config.n_quantiles).to(self.device)
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device), requires_grad=True)
        # construct optims
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.online.parameters(), lr=self.config.lr_critic)
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=self.config.lr_alpha)
        # construct buffer
        self.buffer = Buffer(self.config.buffer_size, observation_dim, action_dim, device=self.device)

    def create_context(self, *, evaluate: bool = False) -> Context:
        return Context(evaluate=evaluate)

    @torch.no_grad()
    def reset(self, ctx: Context, observation: torch.Tensor) -> None:
        ctx.observation = observation.detach().clone()
        ctx.action = None

    @torch.no_grad()
    def sample(self, ctx: Context, observation: torch.Tensor) -> torch.Tensor:
        if not ctx.evaluate and ctx.action is not None:
            raise RuntimeError("Collect the previous sample before sampling again with this context.")
        ctx.observation = observation.detach().clone()
        action = self.actor.predict(ctx.observation, deterministic=ctx.evaluate)
        if not ctx.evaluate:
            ctx.action = action.clone()
        return action

    @torch.no_grad()
    def collect(self, ctx: Context, next_observation: torch.Tensor, reward: torch.Tensor, termination: torch.Tensor, truncation: torch.Tensor) -> bool:
        """Consume the pending sample; truncation keeps the bootstrap discount."""
        if ctx.evaluate:
            raise RuntimeError("Evaluation contexts do not collect training transitions.")
        if ctx.observation is None or ctx.action is None:
            raise RuntimeError("Sample an action with this context before collecting a transition.")
        self.buffer.add(ctx.observation, ctx.action, next_observation, reward, 1 - termination.to(dtype=torch.float32))
        ctx.observation = next_observation.detach().clone()
        ctx.action = None
        return bool(self.buffer.full or self.buffer.head >= self.config.learning_starts)

    def learn(self) -> dict[str, float]:
        with torch.no_grad():
            observations, actions, next_observations, rewards, discounts = self.buffer.sample(self.config.batch_size)
            actions = (actions - self.actor.action_bias) / self.actor.action_scale
            alpha = self.log_alpha.exp().item()
            next_actions_pi, next_log_probs_pi = self.actor.rsample(next_observations)
            # Pool all critics' atoms, then truncate the upper tail (Eq. 10-12).
            next_quantiles = self.critic.target(next_observations, next_actions_pi)
            next_quantiles = next_quantiles.flatten(start_dim=1).sort(dim=-1).values
            n_keep = self.config.n_critics * (self.config.n_quantiles - self.config.top_quantiles_to_drop_per_critic)
            next_quantiles = next_quantiles[:, :n_keep]
            target_quantiles = rewards[:, None] + self.config.gamma * discounts[:, None] * (next_quantiles - alpha * next_log_probs_pi[:, None])
        # update critic with quantile Huber regression (Eq. 8, 14)
        quantiles = self.critic.online(observations, actions)
        critic_loss = quantile_huber_loss(quantiles, target_quantiles)
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()
        # update actor using all online atoms, without truncation (Eq. 15)
        set_requires_grad(self.critic.online, False)
        actions_pi, log_probs_pi = self.actor.rsample(observations)
        q_values_pi = self.critic.online(observations, actions_pi).mean(dim=(-2, -1))
        actor_loss = (alpha * log_probs_pi - q_values_pi).mean()
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()
        set_requires_grad(self.critic.online, True)
        # update alpha (Eq. 5)
        alpha_loss = -(self.log_alpha * (log_probs_pi.detach() + self.target_entropy)).mean()
        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()
        # update target critics
        with torch.no_grad():
            for param_online, param_target in zip(self.critic.online.parameters(), self.critic.target.parameters()):
                param_target.data.copy_(self.config.tau * param_online.data + (1 - self.config.tau) * param_target.data)
        return {"actor": actor_loss.item(), "critic": critic_loss.item(), "alpha": alpha_loss.item()}
