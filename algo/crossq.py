from dataclasses import dataclass, field
import math

import torch


# CrossQ: https://arxiv.org/abs/1902.05605 (Figure 2, Section 3, Appendix A.2)
@dataclass
class Config:
    learning_starts: int = 100
    batch_size: int = 256
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    hidden_dims: list[int] = field(default_factory=lambda:[256, 256])
    critic_hidden_dims: list[int] = field(default_factory=lambda:[2048, 2048])
    policy_delay: int = 3
    # EMA decay, as in the paper; PyTorch BatchNorm uses the opposite convention.
    bn_momentum: float = 0.99
    bn_eps: float = 1e-3
    brn_warmup_steps: int = 100_000
    adam_beta1: float = 0.5
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    lr_alpha: float = 1e-3

@dataclass
class Context:
    """Per-stream state; training and evaluation use independent instances."""
    evaluate: bool = False
    observation: torch.Tensor|None = None
    action: torch.Tensor|None = None

class BatchRenorm(torch.nn.Module):
    """Feature-wise Batch Renormalization, with BN during the warm-up period."""
    running_mean: torch.Tensor
    running_var: torch.Tensor
    num_batches_tracked: torch.Tensor
    def __init__(self, n_features: int, momentum: float, eps: float, warmup_steps: int):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.weight = torch.nn.Parameter(torch.ones(n_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_features))
        self.register_buffer("running_mean", torch.zeros(n_features))
        self.register_buffer("running_var", torch.ones(n_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize with running statistics, without updating them."""
        normalized = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        return self.weight * normalized + self.bias
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a learning batch and update running statistics."""
        if x.ndim != 2:
            raise ValueError("BatchRenorm learning inputs must have shape [batch, features].")
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        std = (var + self.eps).sqrt()
        # BRN corrections are constants for backpropagation (Ioffe, 2017).
        with torch.no_grad():
            running_std = (self.running_var + self.eps).sqrt()
            r = (std / running_std).clamp(1 / 3, 3)
            d = ((mean - self.running_mean) / running_std).clamp(-5, 5)
            warmed_up = self.num_batches_tracked >= self.warmup_steps
            r = torch.where(warmed_up, r, 1.0)
            d = torch.where(warmed_up, d, 0.0)
        normalized = (x - mean) / std * r + d
        with torch.no_grad():
            self.running_mean.mul_(self.momentum).add_(mean, alpha=1 - self.momentum)
            self.running_var.mul_(self.momentum).add_(var, alpha=1 - self.momentum)
            self.num_batches_tracked.add_(1)
        return self.weight * normalized + self.bias

class MLP(torch.nn.Sequential):
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = layer.forward_train(x) if isinstance(layer, BatchRenorm) else layer(x)
        return x

def build_mlp(dims: list[int], momentum: float, eps: float, warmup_steps: int) -> MLP:
    # Normalize the inputs and each activated hidden layer, as in the reference.
    layers: list[torch.nn.Module] = [BatchRenorm(dims[0], momentum, eps, warmup_steps)]
    for i in range(len(dims) - 2):
        layers += [torch.nn.Linear(dims[i], dims[i+1]), torch.nn.ReLU(), BatchRenorm(dims[i+1], momentum, eps, warmup_steps)]
    layers += [torch.nn.Linear(dims[-2], dims[-1])]
    return MLP(*layers)

def set_requires_grad(module: torch.nn.Module, flag: bool) -> torch.nn.Module:
    for p in module.parameters():
        p.requires_grad_(flag)
    return module

class Critic(torch.nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: list[int], momentum: float, eps: float, warmup_steps: int):
        super().__init__()
        self.q1 = build_mlp([observation_dim + action_dim, *hidden_dims, 1], momentum, eps, warmup_steps)
        self.q2 = build_mlp([observation_dim + action_dim, *hidden_dims, 1], momentum, eps, warmup_steps)
    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat((observation, action), dim=-1)
        return self.q1(z).squeeze(-1), self.q2(z).squeeze(-1)
    def forward_train(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat((observation, action), dim=-1)
        return self.q1.forward_train(z).squeeze(-1), self.q2.forward_train(z).squeeze(-1)
    def clipped(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(observation, action)
        return torch.min(q1, q2)

class Actor(torch.nn.Module):
    action_scale: torch.Tensor
    action_bias: torch.Tensor
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: list[int], action_scale: torch.Tensor, action_bias: torch.Tensor, momentum: float, eps: float, warmup_steps: int):
        super().__init__()
        self.network = build_mlp([observation_dim, *hidden_dims, action_dim * 2], momentum, eps, warmup_steps)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)
    def distribution(self, observation: torch.Tensor) -> torch.distributions.normal.Normal:
        mean, logstd = self.network(observation).chunk(2, dim=-1)
        return torch.distributions.normal.Normal(mean, logstd.clamp(-20, 2).exp())
    def distribution_train(self, observation: torch.Tensor) -> torch.distributions.normal.Normal:
        mean, logstd = self.network.forward_train(observation).chunk(2, dim=-1)
        return torch.distributions.normal.Normal(mean, logstd.clamp(-20, 2).exp())
    @torch.no_grad()
    def predict(self, observation: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """Use running statistics for both exploratory and deterministic sampling."""
        normal = self.distribution(observation)
        return self.action_scale * torch.tanh(normal.mean if deterministic else normal.sample()) + self.action_bias
    def rsample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized actions and their log density, as in SAC."""
        return self._rsample(self.distribution(observation))
    def rsample_train(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample for the actor update, updating normalization statistics."""
        return self._rsample(self.distribution_train(observation))
    @staticmethod
    def _rsample(normal: torch.distributions.normal.Normal) -> tuple[torch.Tensor, torch.Tensor]:
        z = normal.rsample()
        actions = torch.tanh(z)
        log_jacobian = 2 * (math.log(2) - z - torch.nn.functional.softplus(-2 * z))
        return actions, (normal.log_prob(z) - log_jacobian).sum(dim=-1)

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
        if min(self.config.batch_size, self.config.buffer_size, self.config.policy_delay) < 1:
            raise ValueError("batch_size, buffer_size, and policy_delay must be positive.")
        if not 0 <= self.config.bn_momentum < 1 or self.config.bn_eps <= 0 or self.config.brn_warmup_steps < 0:
            raise ValueError("Require bn_momentum in [0, 1), bn_eps > 0, and brn_warmup_steps >= 0.")
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
        # construct models (CrossQ has no target networks)
        self.actor = Actor(observation_dim, action_dim, self.config.hidden_dims, action_scale, action_bias, self.config.bn_momentum, self.config.bn_eps, self.config.brn_warmup_steps).to(self.device)
        self.critic = Critic(observation_dim, action_dim, self.config.critic_hidden_dims, self.config.bn_momentum, self.config.bn_eps, self.config.brn_warmup_steps).to(self.device)
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device), requires_grad=True)
        # construct optims
        betas = (self.config.adam_beta1, 0.999)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr_actor, betas=betas)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr_critic, betas=betas)
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=self.config.lr_alpha, betas=betas)
        # construct buffer
        self.buffer = Buffer(self.config.buffer_size, observation_dim, action_dim, device=self.device)
        self.updates = 0

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
        # Joint forward pass: current and next pairs share normalization moments (Fig. 2).
        all_q1, all_q2 = self.critic.forward_train(torch.cat((observations, next_observations)), torch.cat((actions, next_actions_pi)))
        q1, next_q1 = all_q1.chunk(2, dim=0)
        q2, next_q2 = all_q2.chunk(2, dim=0)
        # Detach only the Bellman targets, after the joint forward pass.
        with torch.no_grad():
            target_q_values = rewards + self.config.gamma * discounts * (torch.min(next_q1, next_q2) - alpha * next_log_probs_pi)
        critic_loss = 0.5 * (torch.nn.functional.mse_loss(q1, target_q_values) + torch.nn.functional.mse_loss(q2, target_q_values))
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()
        losses = {"critic": critic_loss.item()}
        self.updates += 1
        # Actor and temperature updates are delayed, as in the reference implementation.
        if self.updates % self.config.policy_delay == 0:
            set_requires_grad(self.critic, False)
            actions_pi, log_probs_pi = self.actor.rsample_train(observations)
            q_values_pi = self.critic.clipped(observations, actions_pi)
            actor_loss = (alpha * log_probs_pi - q_values_pi).mean()
            self.optim_actor.zero_grad()
            actor_loss.backward()
            self.optim_actor.step()
            set_requires_grad(self.critic, True)
            alpha_loss = -(self.log_alpha * (log_probs_pi.detach() + self.target_entropy)).mean()
            self.optim_alpha.zero_grad()
            alpha_loss.backward()
            self.optim_alpha.step()
            losses.update(actor=actor_loss.item(), alpha=alpha_loss.item())
        return losses
