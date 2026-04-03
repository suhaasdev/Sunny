from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging
import time


@dataclass
class StepResult:
    next_state: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

    def __iter__(self):
        return iter((self.next_state, self.reward, self.done, self.info))


@dataclass
class SpaceSpec:
    dtype: str
    shape: Tuple
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    discrete_n: Optional[int] = None
    description: str = ""

    def sample(self) -> np.ndarray:
        if self.discrete_n is not None:
            return np.random.randint(0, self.discrete_n)
        return np.random.uniform(self.low, self.high).astype(self.dtype)


class EnvLogger:
    def __init__(self, name: str, level: int = logging.WARNING):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                datefmt="%H:%M:%S"
            ))
            self.logger.addHandler(handler)
        self.logger.setLevel(level)
        self._step_log: List[Dict] = []

    def log_step(self, step: int, action: Any, reward: float, done: bool, info: Dict):
        entry = {
            "step": step,
            "action": action,
            "reward": reward,
            "done": done,
            "info": info,
            "timestamp": time.time()
        }
        self._step_log.append(entry)
        self.logger.debug(f"Step {step}: action={action}, reward={reward:.4f}, done={done}")

    def get_episode_log(self) -> List[Dict]:
        return list(self._step_log)

    def clear(self):
        self._step_log.clear()

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)


class BaseEnvironment(ABC):
    metadata: Dict[str, Any] = {"render_modes": ["ansi"]}

    def __init__(self, config: Dict[str, Any], log_level: int = logging.WARNING):
        self.config = config
        self.np_random = np.random.default_rng(config.get("seed", None))
        self._step_count = 0
        self._episode_count = 0
        self._total_reward = 0.0
        self.logger = EnvLogger(self.__class__.__name__, log_level)
        self._initialized = False

    @property
    @abstractmethod
    def observation_space(self) -> SpaceSpec:
        ...

    @property
    @abstractmethod
    def action_space(self) -> SpaceSpec:
        ...

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def step(self, action: Any) -> StepResult:
        ...

    @abstractmethod
    def state(self) -> Dict[str, Any]:
        ...

    def render(self, mode: str = "ansi") -> Optional[str]:
        return None

    def close(self):
        pass

    def seed(self, seed: int):
        self.np_random = np.random.default_rng(seed)
        self.config["seed"] = seed

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def episode_reward(self) -> float:
        return self._total_reward

    def _pre_reset(self):
        if self._initialized:
            self._episode_count += 1
        self._step_count = 0
        self._total_reward = 0.0
        self.logger.clear()
        self._initialized = True

    def _post_step(self, action: Any, result: StepResult):
        self._step_count += 1
        self._total_reward += result.reward
        self.logger.log_step(self._step_count, action, result.reward, result.done, result.info)


class EnvRegistry:
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(env_cls):
            cls._registry[name] = env_cls
            return env_cls
        return decorator

    @classmethod
    def make(cls, name: str, config: Optional[Dict] = None) -> BaseEnvironment:
        if name not in cls._registry:
            raise ValueError(f"Environment '{name}' not registered. Available: {list(cls._registry)}")
        return cls._registry[name](config or {})

    @classmethod
    def list_envs(cls) -> List[str]:
        return list(cls._registry.keys())
