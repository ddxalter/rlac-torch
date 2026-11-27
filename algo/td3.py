from dataclasses import dataclass, field

import torch


# TD3: https://proceedings.mlr.press/v80/fujimoto18a.html (Algorithm 1)
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
    policy_delay: int = 2
    # Noise standard deviations and clipping are in normalized action coordinates.
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5

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
        self.network = build_mlp([observation_dim, *hidden_dims, action_dim])
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return normalized actions for training."""
        return torch.tanh(self.network(observation))
    def predict(self, observation: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        """Return environment actions, optionally with Gaussian exploration noise."""
        actions = self(observation)
        if noise_std > 0:
            actions = (actions + noise_std * torch.randn_like(actions)).clamp(-1, 1)
        return self.action_scale * actions + self.action_bias

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
        self.device = torch.device(device)
        observation_dim = observation_spec["ndim"]
        action_dim = action_spec["ndim"]
        action_low = torch.as_tensor(action_spec["low"], dtype=torch.float32, device=self.device)
        action_high = torch.as_tensor(action_spec["high"], dtype=torch.float32, device=self.device)
        action_scale = 0.5 * (action_high - action_low)
        action_bias = 0.5 * (action_high + action_low)
        # construct models
        self.actor = Actor(observation_dim, action_dim, self.config.hidden_dims, action_scale, action_bias).to(self.device)
        self.actor_target = Actor(observation_dim, action_dim, self.config.hidden_dims, action_scale, action_bias).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        set_requires_grad(self.actor_target, False).train(False)
        self.critic = Critic(observation_dim, action_dim, self.config.hidden_dims).to(self.device)
        # construct optims
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.online.parameters(), lr=self.config.lr_critic)
        # construct buffer
        self.buffer = Buffer(self.config.buffer_size, observation_dim, action_dim, device=self.device)
        self.updates = 0

    @torch.no_grad()
    def predict(self, observation: torch.Tensor, evaluate: bool) -> torch.Tensor:
        return self.actor.predict(observation, noise_std=0.0 if evaluate else self.config.exploration_noise)

    @torch.no_grad()
    def collect(self, observation: torch.Tensor, action: torch.Tensor, next_observation: torch.Tensor, reward: torch.Tensor, termination: torch.Tensor, truncation: torch.Tensor) -> bool:
        self.buffer.add(observation, action, next_observation, reward, 1 - termination.to(dtype=torch.float32))
        return bool(self.buffer.full or self.buffer.head >= self.config.learning_starts)

    def learn(self) -> dict[str, float]:
        with torch.no_grad():
            observations, actions, next_observations, rewards, discounts = self.buffer.sample(self.config.batch_size)
            actions = (actions - self.actor.action_bias) / self.actor.action_scale
            # target policy smoothing (Algorithm 1)
            noise = (self.config.policy_noise * torch.randn_like(actions)).clamp(-self.config.noise_clip, self.config.noise_clip)
            next_actions_pi = (self.actor_target(next_observations) + noise).clamp(-1, 1)
            next_q_values = self.critic.target.clipped(next_observations, next_actions_pi)
            target_q_values = rewards + self.config.gamma * discounts * next_q_values
        # update critic
        q1, q2 = self.critic.online(observations, actions)
        q1_loss = torch.nn.functional.mse_loss(q1, target_q_values)
        q2_loss = torch.nn.functional.mse_loss(q2, target_q_values)
        critic_loss = 0.5 * (q1_loss + q2_loss)
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()
        losses = {"critic": critic_loss.item()}
        # delayed actor and target updates (Algorithm 1)
        self.updates += 1
        if self.updates % self.config.policy_delay == 0:
            set_requires_grad(self.critic.online, False)
            actions_pi = self.actor(observations)
            q1_pi = self.critic.online.q1(torch.cat((observations, actions_pi), dim=-1)).squeeze(-1)
            actor_loss = -q1_pi.mean()
            self.optim_actor.zero_grad()
            actor_loss.backward()
            self.optim_actor.step()
            set_requires_grad(self.critic.online, True)
            with torch.no_grad():
                for param_online, param_target in zip(self.critic.online.parameters(), self.critic.target.parameters()):
                    param_target.data.copy_(self.config.tau * param_online.data + (1 - self.config.tau) * param_target.data)
                for param_online, param_target in zip(self.actor.parameters(), self.actor_target.parameters()):
                    param_target.data.copy_(self.config.tau * param_online.data + (1 - self.config.tau) * param_target.data)
            losses["actor"] = actor_loss.item()
        return losses
