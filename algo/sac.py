from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    learning_starts: int = 100
    batch_size: int = 256
    buffer_size: int = 1_000_000
    tau: float = 0.005
    gamma: float = 0.99
    hidden_dims: list[int] = field(default_factory=lambda:[256, 256])
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4

def build_mlp(dims: list[int], activation: type[torch.nn.Module] = torch.nn.ReLU) -> torch.nn.Sequential:
    layers = [torch.nn.Linear(dims[0], dims[1])]
    for i in range(1, len(dims) - 1):
        layers += [activation(), torch.nn.Linear(dims[i], dims[i+1])]
    return torch.nn.Sequential(*layers)

def set_requires_grad(module: torch.nn.Module, flag: bool) -> torch.nn.Module:
    for p in module.parameters():
        p.requires_grad_(flag)
    return module

class DoubleQ(torch.nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.q1 = build_mlp([observation_dim + action_dim, *hidden_dims, 1])
        self.q2 = build_mlp([observation_dim + action_dim, *hidden_dims, 1])
    def forward(self, observation, action) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat((observation, action), dim=-1)
        return self.q1(z).squeeze(-1), self.q2(z).squeeze(-1)
    def clipped(self, observation, action) -> torch.Tensor:
        q1, q2 = self(observation, action)
        return torch.min(q1, q2)

class Critic(torch.nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.online = DoubleQ(observation_dim, action_dim, hidden_dims)
        self.target = DoubleQ(observation_dim, action_dim, hidden_dims)
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
    def rsample(self, observation: torch.Tensor):
        normal = self.distribution(observation)
        z = normal.rsample()
        u = torch.tanh(z)
        log_prob = (normal.log_prob(z) - (1 - u**2 + 1e-6).log()).sum(dim=1)
        return u, log_prob

class Buffer:
    def __init__(self, buffer_size: int, observation_dim: int, action_dim: int, device: torch.Device):
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
        self.device = torch.device(device)
        observation_dim = observation_spec["ndim"]
        action_dim = action_spec["ndim"]
        action_low = torch.as_tensor(action_spec["low"], dtype=torch.float32, device=self.device)
        action_high = torch.as_tensor(action_spec["high"], dtype=torch.float32, device=self.device)
        action_scale = 0.5 * (action_high - action_low)
        action_bias = 0.5 * (action_high + action_low)
        self.target_entropy = torch.tensor(-action_dim, dtype=torch.float32, device=self.device)
        # construct models
        self.actor = Actor(observation_dim, action_dim, self.config.hidden_dims, action_scale, action_bias)
        self.critic = Critic(observation_dim, action_dim, self.config.hidden_dims)
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device), requires_grad=True)
        # construct optims
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=self.config.lr_alpha)
        # construct buffer
        self.buffer = Buffer(self.config.buffer_size, observation_dim, action_dim, device=self.device)

    @torch.no_grad()
    def predict(self, observation: torch.Tensor, evaluate: bool) -> torch.Tensor:
        return self.actor.predict(observation, deterministic=evaluate)

    @torch.no_grad()
    def collect(self, observation: torch.Tensor, action: torch.Tensor, next_observation: torch.Tensor, reward: torch.Tensor, termination: torch.Tensor, truncation: torch.Tensor) -> bool:
        self.buffer.add(observation, action, next_observation, reward, 1 - termination)
        return bool(self.buffer.head >= self.config.learning_starts)

    def learn(self) -> dict[str, float]:
        with torch.no_grad():
            observations, actions, next_observations, rewards, discounts = self.buffer.sample(self.config.batch_size)
            actions = (actions - self.actor.action_bias) / self.actor.action_scale
            alpha = self.log_alpha.exp().item()
            next_actions_pi, next_log_probs_pi = self.actor.rsample(next_observations)
            next_q_values = self.critic.target.clipped(next_observations, next_actions_pi)
            target_q_values = rewards + self.config.gamma * discounts * (next_q_values - alpha * next_log_probs_pi)
        # update critic
        set_requires_grad(self.critic.online, True)
        q1, q2 = self.critic.online(observations, actions)
        q1_loss = torch.nn.functional.mse_loss(q1, target_q_values)
        q2_loss = torch.nn.functional.mse_loss(q2, target_q_values)
        critic_loss = 0.5 * (q1_loss + q2_loss)
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()
        with torch.no_grad():
            for param_online, param_target in zip(self.critic.online.parameters(), self.critic.target.parameters()):
                param_target.data.copy_(self.config.tau * param_online.data + (1 - self.config.tau) * param_target.data)
        set_requires_grad(self.critic.online, False)
        # update actor
        actions_pi, log_probs_pi = self.actor.rsample(observations)
        q_values_pi = self.critic.online.clipped(observations, actions_pi)
        actor_loss = (alpha * log_probs_pi - q_values_pi).mean()
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()
        # update alpha
        alpha_loss = -(self.log_alpha * (log_probs_pi.detach() + self.target_entropy)).mean()
        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()
        return {"actor": actor_loss.item(), "critic": critic_loss.item(), "alpha": alpha_loss.item()}
