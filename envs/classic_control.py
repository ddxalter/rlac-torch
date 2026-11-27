from typing import Any
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class DiscreteSpace:
    ndim: int
    size: int
    high: int
    low: int
    @staticmethod
    def from_ndim(ndim: Any):
       return DiscreteSpace(ndim=ndim, size=1, high=ndim, low=0)

@dataclass
class BoxSpace:
    ndim: int
    size: int
    high: npt.NDArray[np.float32]
    low: npt.NDArray[np.float32]
    @staticmethod
    def from_high(high: Any):
        flatten = np.array(high, dtype=np.float32).flatten()
        return BoxSpace(ndim=flatten.size, size=flatten.size, high=flatten, low=-flatten)
    @staticmethod
    def from_bounds(low: Any, high: Any):
        h = np.array(high, dtype=np.float32).flatten()
        l = np.array(low, dtype=np.float32).flatten()
        return BoxSpace(ndim=l.size, size=l.size, high=h, low=l)

@dataclass
class Step:
    observation: Any
    reward: float
    termination: bool
    truncation: bool

@dataclass
class CartPoleContext:
    state: npt.NDArray[np.float32]
    timestep: int = 0

class CartPole:
    def __init__(self, config: dict|None = None):
        config = config if config else {}
        self.dtime = config.get("dtime", 0.02)
        self.gravity = config.get("gravity", 9.8)
        self.cart_mass = config.get("cart_mass", 1.0)
        self.pole_mass = config.get("pole_mass", 0.1)
        self.pole_length_half = config.get("pole_length_half", 0.5)
        self.cart_position_max = config.get("cart_position_max", 2.4)
        self.pole_angle_max = config.get("pole_angle_max", 12 * 2 * np.pi / 360)
        self.timelimit = config.get("timelimit", 200)
        self.force_max = config.get("force_max", 10.0)
        self.observation_space = BoxSpace.from_high([self.cart_position_max, np.inf, self.pole_angle_max, np.inf])
        self.action_space = BoxSpace.from_high([1.0])
        self.timestep = 0
        self.total_mass = self.cart_mass + self.pole_mass

    def reset(self, rng: np.random.Generator):
        state = np.zeros(4, dtype=np.float32)
        observation = self.observe(state, rng)
        return CartPoleContext(state=state, timestep=0), Step(observation, 0.0, False, False)

    def step(self, ctx: CartPoleContext, action: Any, rng: np.random.Generator):
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = ctx.state
        force = self.force_max * np.clip(action, -1, 1)
        sin_angle = np.sin(pole_angle)
        cos_angle = np.cos(pole_angle)
        temp = (force + self.pole_mass * self.pole_length_half * pole_angular_velocity**2 * sin_angle) / self.total_mass
        pole_angular_acceleration = (self.gravity * sin_angle - cos_angle * temp) / (self.pole_length_half * (4.0 / 3.0 - self.pole_mass * cos_angle**2 / self.total_mass))
        cart_acceleration = temp - self.pole_mass * self.pole_length_half * pole_angular_acceleration * cos_angle / self.total_mass
        ctx.state[0] = cart_position + self.dtime * cart_velocity
        ctx.state[1] = cart_velocity + self.dtime * cart_acceleration
        ctx.state[2] = pole_angle + self.dtime * pole_angular_velocity
        ctx.state[3] = pole_angular_velocity + self.dtime * pole_angular_acceleration

        observation = self.observe(ctx.state, rng)
        reward = 1.0
        termination = bool(np.abs(ctx.state[0]) > self.cart_position_max or np.abs(ctx.state[2]) > self.pole_angle_max)
        truncation = self.timestep >= self.timelimit
        return Step(observation, reward, termination, truncation)
    def observe(self, state, rng: np.random.Generator):
        return np.array(state, dtype=np.float32)

@dataclass
class PendulumState:
    state: npt.NDArray[np.float32]
    observation: npt.NDArray[np.float32]
    reward: float
    termination: bool
    truncation: bool
    timestep: int = 0

class Pendulum:
    def __init__(self, config: dict|None = None):
        config = config if config else {}
        self.dtime = config.get("dtime", 0.05)
        self.gravity = config.get("gravity", 10.0)
        self.max_speed = config.get("max_speed", 8.0)
        self.max_torque = config.get("max_torque", 2.0)
        self.mass = config.get("mass", 1.0)
        self.length = config.get("length", 1.0)
        self.timelimit = config.get("timelimit", 200)
        self.observation_space = BoxSpace.from_high([1, 1, self.max_speed])
        self.action_space = BoxSpace.from_high([self.max_torque])
        self.initial_space = BoxSpace.from_high([np.pi, 1.0])
        self.coef1 = 3.0 * self.gravity / (2 * self.length)
        self.coef2 = 3.0 / self.mass * self.length**2
    def reset(self, rng: np.random.Generator):
        state = rng.uniform(self.initial_space.low, self.initial_space.high).astype(np.float32)
        return PendulumState(state, self.observe(state, rng), 0.0, False, False, timestep=0)
    def step(self, ctx: PendulumState, action: npt.NDArray[np.float32], rng: np.random.Generator):
        theta, dtheta = ctx.state
        torque = np.clip(action, -self.max_torque, self.max_torque)[0]
        ctx.reward = -(self.normalize_angle(theta) ** 2 + 0.1 * (dtheta ** 2) + 0.001 * (torque**2))
        ddtheta = self.coef1 * np.sin(theta) + self.coef2 * torque
        dtheta = np.clip(dtheta + self.dtime * ddtheta, -self.max_speed, self.max_speed)
        theta = theta + self.dtime * dtheta
        ctx.state[0] = theta
        ctx.state[1] = dtheta
        ctx.observation = self.observe(ctx.state, rng)
        ctx.timestep += 1
        ctx.termination = False
        ctx.truncation = ctx.timestep >= self.timelimit

    def normalize_angle(self, angle):
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    def observe(self, state, rng: np.random.Generator):
        return np.array([np.cos(state[0]), np.sin(state[0]), state[1]], dtype=np.float32)

@dataclass
class MountainCarContext:
    state: npt.NDArray[np.float32]
    timestep: int = 0

class MountainCar:
    def __init__(self, config: dict|None = None):
        config = config if config else {}
        self.min_position =  -1.2
        self.max_position = 0.6
        self.min_velocity = -0.07
        self.max_velocity = -0.07
        self.max_speed = 0.07
        self.goal_position = 0.45
        self.goal_velocity = 0.0
        self.power = 0.0015
        self.min_action = -1.0
        self.max_action = 1.0
        self.timelimit = config.get("timelimit", 999)
        self.observation_space = BoxSpace.from_bounds(low=[self.min_position, self.min_velocity], high=[self.max_position, self.max_velocity])
        self.action_space = BoxSpace.from_high([1.0])
    def reset(self, rng: np.random.Generator):
        initial_position = rng.uniform(-0.6, -0.4)
        initial_state = np.array([initial_position, 0], dtype=np.float32)
        observation = self.observe(initial_state, rng)
        return MountainCarContext(initial_state, 0), Step(observation, 0.0, False, False)
    def step(self, ctx: MountainCarContext, action: npt.NDArray[np.float32], rng: np.random.Generator):
        position, velocity = ctx.state
        force = np.clip(action, self.min_action, self.max_action)[0]
        velocity += force * self.power - 0.0025 * np.cos(3.0 * position)
        velocity = np.clip(velocity, self.min_velocity, self.max_velocity)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position <= self.min_position and velocity < 0.0:
            velocity = 0.0
        ctx.state[:] = position, velocity
        ctx.timestep += 1
        termination = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = (0 if termination else 100.0) - 0.1 * (action[0] ** 2)
        truncation = ctx.timestep >= self.timelimit
        return Step(self.observe(ctx.state, rng), reward, termination, truncation), {}
    def observe(self, state, rng: np.random.Generator):
        return np.array(state, dtype=np.float32)

def make(name: str, *args, **kwargs):
    match name:
        case "CartPole": return CartPole(kwargs)
        case "Pendulum": return Pendulum(kwargs)
        case "MountainCar": return MountainCar(kwargs)
    raise ValueError("unsupported env name")
