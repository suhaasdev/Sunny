from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np

from openenv.core.base import BaseEnvironment


class BaseAgent(ABC):

    def __init__(self, env: BaseEnvironment, seed: int = 0):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self._episode_rewards: List[float] = []
        self._current_episode_reward = 0.0

    @abstractmethod
    def act(self, state: Dict[str, Any]) -> Any:
        ...

    def on_reset(self, state: Dict[str, Any]):
        self._episode_rewards.append(self._current_episode_reward)
        self._current_episode_reward = 0.0

    def on_step(self, state, action, reward, next_state, done, info):
        self._current_episode_reward += reward

    @property
    def episode_rewards(self) -> List[float]:
        return self._episode_rewards


class RandomAgent(BaseAgent):

    def act(self, state: Dict[str, Any]) -> Any:
        space = self.env.action_space
        if space.discrete_n is not None:
            return [int(self.rng.integers(0, space.discrete_n)) for _ in range(space.shape[0])]
        return space.sample()


class RuleBasedWarehouseAgent(BaseAgent):
    """Heuristic agent for SmartWarehouse that:
    - prioritizes charging when battery < 20%
    - tries to move toward nearest unassigned order pickup
    - drops items when at dropoff zone
    """

    def act(self, state: Dict[str, Any]) -> List[int]:
        from openenv.envs.warehouse import ACTIONS
        actions = []
        robots = state["robots"]
        orders = state["orders"]
        shelves = state["shelves"]
        charging = state["charging_stations"]

        action_inv = {v: k for k, v in ACTIONS.items()}

        for r in robots:
            pos = r["position"]
            bat = r["battery"]
            carrying = r["carrying"]

            if bat < 0.2:
                nearest_charge = min(charging, key=lambda c: abs(c[0]-pos[0]) + abs(c[1]-pos[1]))
                if list(nearest_charge) == pos:
                    actions.append(action_inv["CHARGE"])
                else:
                    actions.append(self._move_toward(pos, nearest_charge))
                continue

            if carrying is not None:
                order = next((o for o in orders if o["id"] == carrying), None)
                if order:
                    target = order["dropoff"]
                    if pos == target:
                        actions.append(action_inv["DROP_ITEM"])
                    else:
                        actions.append(self._move_toward(pos, target))
                    continue

            open_orders = [o for o in orders if o["assigned_robot"] is None]
            if open_orders:
                order = open_orders[0]
                shelf = next((s for s in shelves if s["id"] == order["pickup_shelf"]), None)
                if shelf:
                    target = shelf["position"]
                    if pos == target:
                        actions.append(action_inv["PICK_ITEM"])
                    else:
                        actions.append(self._move_toward(pos, target))
                    continue

            actions.append(action_inv["WAIT"])

        return actions

    def _move_toward(self, pos: List[int], target: List[int]) -> int:
        from openenv.envs.warehouse import ACTIONS
        action_inv = {v: k for k, v in ACTIONS.items()}
        dr = target[0] - pos[0]
        dc = target[1] - pos[1]
        if abs(dr) >= abs(dc):
            return action_inv["MOVE_DOWN"] if dr > 0 else action_inv["MOVE_UP"]
        return action_inv["MOVE_RIGHT"] if dc > 0 else action_inv["MOVE_LEFT"]
