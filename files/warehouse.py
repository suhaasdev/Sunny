from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging

from openenv.core.base import BaseEnvironment, EnvRegistry, SpaceSpec, StepResult

DEFAULT_CONFIG = {
    "grid_size": 8,
    "num_robots": 3,
    "num_shelves": 12,
    "num_charging_stations": 2,
    "max_steps": 300,
    "order_arrival_rate": 0.25,
    "max_pending_orders": 8,
    "battery_drain_rate": 0.04,
    "battery_drain_carrying": 0.07,
    "noise_std": 0.02,
    "partial_observability": False,
    "obs_radius": 3,
    "seed": 42,
}

ACTIONS = {
    0: "MOVE_UP",
    1: "MOVE_DOWN",
    2: "MOVE_LEFT",
    3: "MOVE_RIGHT",
    4: "PICK_ITEM",
    5: "DROP_ITEM",
    6: "CHARGE",
    7: "WAIT",
}

NUM_ACTIONS = len(ACTIONS)


@dataclass
class Robot:
    robot_id: int
    position: List[int]
    battery: float = 1.0
    carrying: Optional[int] = None
    total_deliveries: int = 0
    total_distance: float = 0.0
    is_charging: bool = False


@dataclass
class Shelf:
    shelf_id: int
    position: Tuple[int, int]
    item_type: int
    has_item: bool = True


@dataclass
class Order:
    order_id: int
    item_type: int
    pickup_shelf: int
    dropoff_position: Tuple[int, int]
    age: int = 0
    assigned_robot: Optional[int] = None


@EnvRegistry.register("SmartWarehouse-v1")
class SmartWarehouseEnv(BaseEnvironment):

    def __init__(self, config: Dict[str, Any], log_level: int = logging.WARNING):
        cfg = {**DEFAULT_CONFIG, **config}
        super().__init__(cfg, log_level)
        self.G = cfg["grid_size"]
        self.num_robots = cfg["num_robots"]
        self.num_shelves = cfg["num_shelves"]
        self.num_charging = cfg["num_charging_stations"]
        self.max_steps = cfg["max_steps"]
        self.order_rate = cfg["order_arrival_rate"]
        self.max_orders = cfg["max_pending_orders"]
        self.drain = cfg["battery_drain_rate"]
        self.drain_carry = cfg["battery_drain_carrying"]
        self.noise_std = cfg["noise_std"]
        self.partial_obs = cfg["partial_observability"]
        self.obs_radius = cfg["obs_radius"]

        self._build_layout()

    def _build_layout(self):
        g = self.G
        rng = np.random.default_rng(self.config["seed"])

        shelf_cols = list(range(1, g - 1, 2))
        shelf_rows = list(range(1, g - 1, 2))
        candidates = [(r, c) for r in shelf_rows for c in shelf_cols]
        rng.shuffle(candidates)
        num_shelves = min(self.num_shelves, len(candidates))
        shelf_positions = candidates[:num_shelves]
        self.num_shelves = num_shelves

        self._fixed_shelves = [
            Shelf(
                shelf_id=i,
                position=tuple(shelf_positions[i]),
                item_type=int(rng.integers(0, 4)),
            )
            for i in range(self.num_shelves)
        ]

        self._fixed_charging = [
            (0, 0),
            (0, g - 1),
        ][:self.num_charging]

        dropoff_row = g - 1
        self._dropoff_zones = [(dropoff_row, c) for c in range(0, g, 2)]
        self._robot_start_positions = [(g - 1, c) for c in range(1, self.num_robots + 1)]

    @property
    def observation_space(self) -> SpaceSpec:
        obs_dim = (
            self.num_robots * 5 +
            self.num_shelves * 3 +
            self.max_orders * 4 +
            1
        )
        return SpaceSpec(
            dtype="float32",
            shape=(obs_dim,),
            low=np.zeros(obs_dim, dtype="float32"),
            high=np.ones(obs_dim, dtype="float32") * self.G,
            description="Warehouse flat observation vector"
        )

    @property
    def action_space(self) -> SpaceSpec:
        return SpaceSpec(
            dtype="int32",
            shape=(self.num_robots,),
            discrete_n=NUM_ACTIONS,
            description=f"Per-robot action, {NUM_ACTIONS} discrete choices each"
        )

    def reset(self) -> Dict[str, Any]:
        self._pre_reset()
        rng = self.np_random

        self.shelves: List[Shelf] = [
            Shelf(
                shelf_id=s.shelf_id,
                position=s.position,
                item_type=s.item_type,
                has_item=True,
            )
            for s in self._fixed_shelves
        ]

        self.robots: List[Robot] = [
            Robot(
                robot_id=i,
                position=list(self._robot_start_positions[i]),
                battery=float(rng.uniform(0.7, 1.0)),
            )
            for i in range(self.num_robots)
        ]

        self.orders: List[Order] = []
        self._order_counter = 0
        self._completed_orders = 0
        self._failed_orders = 0
        self._penalties = 0.0

        self._maybe_spawn_orders()
        self.logger.info("Environment reset.")
        return self.state()

    def step(self, actions: List[int]) -> StepResult:
        if len(actions) != self.num_robots:
            raise ValueError(f"Expected {self.num_robots} actions, got {len(actions)}")

        rewards = []
        info: Dict[str, Any] = {
            "completed_orders": 0,
            "failed_orders": 0,
            "low_battery_robots": [],
            "collisions": 0,
            "actions": [ACTIONS[a] for a in actions],
        }

        intended_positions = {}
        for robot_id, action in enumerate(actions):
            robot = self.robots[robot_id]
            if robot.battery <= 0:
                continue
            new_pos = self._compute_move(robot.position, action)
            if new_pos in intended_positions.values():
                info["collisions"] += 1
                new_pos = robot.position
            intended_positions[robot_id] = new_pos

        step_reward = 0.0

        for robot_id, action in enumerate(actions):
            robot = self.robots[robot_id]
            r = self._apply_action(robot, action, intended_positions.get(robot_id, robot.position), info)
            step_reward += r

        for order in self.orders:
            order.age += 1

        expired = [o for o in self.orders if o.age > 60 and o.assigned_robot is None]
        for o in expired:
            self.orders.remove(o)
            self._failed_orders += 1
            info["failed_orders"] += 1
            step_reward -= 2.0

        self._maybe_spawn_orders()

        step_reward += self._add_noise(0.0)

        step_reward = max(step_reward, -10.0)

        done = (
            self._step_count >= self.max_steps or
            all(r.battery <= 0 for r in self.robots)
        )

        info.update({
            "step": self._step_count,
            "total_completed": self._completed_orders,
            "total_failed": self._failed_orders,
            "pending_orders": len(self.orders),
            "robot_batteries": [round(r.battery, 3) for r in self.robots],
            "episode_reward": self._total_reward + step_reward,
        })

        result = StepResult(
            next_state=self.state(),
            reward=step_reward,
            done=done,
            info=info,
        )
        self._post_step(actions, result)
        return result

    def _apply_action(
        self, robot: Robot, action: int, new_pos: List[int], info: Dict
    ) -> float:
        reward = -0.01

        if robot.battery <= 0:
            robot.is_charging = False
            return -0.05

        action_name = ACTIONS[action]

        if action_name == "CHARGE":
            if tuple(robot.position) in self._fixed_charging:
                robot.battery = min(1.0, robot.battery + 0.15)
                robot.is_charging = True
                reward += 0.02
            else:
                reward -= 0.1
            return reward

        robot.is_charging = False

        if action_name in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"):
            old_pos = list(robot.position)
            robot.position = new_pos
            moved = robot.position != old_pos
            drain = self.drain_carry if robot.carrying is not None else self.drain
            robot.battery = max(0.0, robot.battery - drain)
            if moved:
                robot.total_distance += 1
            if robot.battery < 0.15:
                info["low_battery_robots"].append(robot.robot_id)
                reward -= 0.05

        elif action_name == "PICK_ITEM":
            picked = False
            for shelf in self.shelves:
                if tuple(robot.position) == shelf.position and shelf.has_item:
                    for order in self.orders:
                        if (order.item_type == shelf.item_type
                                and order.assigned_robot is None
                                and robot.carrying is None):
                            robot.carrying = order.order_id
                            order.assigned_robot = robot.robot_id
                            shelf.has_item = False
                            picked = True
                            reward += 1.0
                            break
                if picked:
                    break
            if not picked:
                reward -= 0.2
            robot.battery = max(0.0, robot.battery - self.drain)

        elif action_name == "DROP_ITEM":
            dropped = False
            if robot.carrying is not None:
                order = next((o for o in self.orders if o.order_id == robot.carrying), None)
                if order and tuple(robot.position) == order.dropoff_position:
                    self.orders.remove(order)
                    robot.carrying = None
                    robot.total_deliveries += 1
                    self._completed_orders += 1
                    info["completed_orders"] += 1
                    reward += 5.0
                    dropped = True
                elif order:
                    reward -= 0.3
            if not dropped and robot.carrying is None:
                reward -= 0.1
            robot.battery = max(0.0, robot.battery - self.drain)

        elif action_name == "WAIT":
            robot.battery = max(0.0, robot.battery - self.drain * 0.3)
            reward -= 0.02

        return reward

    def _compute_move(self, position: List[int], action: int) -> List[int]:
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        delta = moves.get(action, (0, 0))
        nr = max(0, min(self.G - 1, position[0] + delta[0]))
        nc = max(0, min(self.G - 1, position[1] + delta[1]))
        shelf_positions = {s.position for s in self.shelves}
        if (nr, nc) in shelf_positions:
            return list(position)
        return [nr, nc]

    def _maybe_spawn_orders(self):
        if len(self.orders) >= self.max_orders:
            return
        rng = self.np_random
        if rng.random() < self.order_rate:
            available_shelves = [s for s in self.shelves if s.has_item]
            if not available_shelves:
                return
            shelf = available_shelves[int(rng.integers(0, len(available_shelves)))]
            dropoff = self._dropoff_zones[int(rng.integers(0, len(self._dropoff_zones)))]
            order = Order(
                order_id=self._order_counter,
                item_type=shelf.item_type,
                pickup_shelf=shelf.shelf_id,
                dropoff_position=dropoff,
            )
            self.orders.append(order)
            self._order_counter += 1

    def _add_noise(self, value: float) -> float:
        return value + float(self.np_random.normal(0, self.noise_std))

    def state(self) -> Dict[str, Any]:
        if self.partial_obs:
            return self._partial_state()
        return self._full_state()

    def _full_state(self) -> Dict[str, Any]:
        return {
            "robots": [
                {
                    "id": r.robot_id,
                    "position": list(r.position),
                    "battery": round(r.battery, 4),
                    "carrying": r.carrying,
                    "is_charging": r.is_charging,
                }
                for r in self.robots
            ],
            "shelves": [
                {
                    "id": s.shelf_id,
                    "position": list(s.position),
                    "item_type": s.item_type,
                    "has_item": s.has_item,
                }
                for s in self.shelves
            ],
            "orders": [
                {
                    "id": o.order_id,
                    "item_type": o.item_type,
                    "pickup_shelf": o.pickup_shelf,
                    "dropoff": list(o.dropoff_position),
                    "age": o.age,
                    "assigned_robot": o.assigned_robot,
                }
                for o in self.orders
            ],
            "step": self._step_count,
            "completed": self._completed_orders,
            "failed": self._failed_orders,
            "charging_stations": [list(c) for c in self._fixed_charging],
        }

    def _partial_state(self) -> Dict[str, Any]:
        full = self._full_state()
        filtered_robots = []
        for r_data in full["robots"]:
            visible_robots = []
            rx, ry = r_data["position"]
            for other in full["robots"]:
                ox, oy = other["position"]
                if abs(ox - rx) <= self.obs_radius and abs(oy - ry) <= self.obs_radius:
                    visible_robots.append(other)
            filtered_robots.append({**r_data, "visible_robots": visible_robots})
        return {**full, "robots": filtered_robots}

    def to_vector(self) -> np.ndarray:
        s = self._full_state()
        vec = []
        for r in s["robots"]:
            vec += [r["position"][0] / self.G, r["position"][1] / self.G,
                    r["battery"], float(r["carrying"] is not None), float(r["is_charging"])]
        for sh in s["shelves"]:
            vec += [sh["position"][0] / self.G, sh["position"][1] / self.G, float(sh["has_item"])]
        for i in range(self.max_orders):
            if i < len(s["orders"]):
                o = s["orders"][i]
                vec += [o["item_type"] / 4.0, o["dropoff"][0] / self.G,
                        o["dropoff"][1] / self.G, min(o["age"] / 60.0, 1.0)]
            else:
                vec += [0.0, 0.0, 0.0, 0.0]
        vec.append(self._step_count / self.max_steps)
        return np.array(vec, dtype=np.float32)

    def render(self, mode: str = "ansi") -> str:
        g = self.G
        grid = [["." for _ in range(g)] for _ in range(g)]

        for pos in self._fixed_charging:
            grid[pos[0]][pos[1]] = "C"

        for s in self.shelves:
            r, c = s.position
            grid[r][c] = str(s.item_type) if s.has_item else "_"

        for pos in self._dropoff_zones:
            if grid[pos[0]][pos[1]] == ".":
                grid[pos[0]][pos[1]] = "D"

        for robot in self.robots:
            rr, rc = robot.position
            sym = f"R{robot.robot_id}" if not robot.carrying else f"r{robot.robot_id}"
            grid[rr][rc] = sym[0]

        header = f"\n{'='*32}\n  SmartWarehouse Step {self._step_count}\n{'='*32}"
        top_border = "  +" + "-" * (g * 2 - 1) + "+"
        rows = [top_border]
        for i, row in enumerate(grid):
            rows.append(f"{i} |" + " ".join(row) + "|")
        rows.append(top_border)
        cols = "    " + " ".join(str(c) for c in range(g))

        robot_info = "\n  Robots:"
        for r in self.robots:
            bat_bar = "█" * int(r.battery * 10) + "░" * (10 - int(r.battery * 10))
        robot_info = "\n  Robots:\n" + "\n".join(
            f"    R{r.robot_id} pos={r.position} bat=[{('█'*int(r.battery*10))+('░'*(10-int(r.battery*10)))}]"
            f" carry={'order_'+str(r.carrying) if r.carrying is not None else 'None'}"
            for r in self.robots
        )
        order_info = f"\n  Orders pending: {len(self.orders)} | Completed: {self._completed_orders} | Failed: {self._failed_orders}"
        legend = "\n  Legend: C=Charging, D=Dropoff, 0-3=Shelf(item_type), R=Robot"

        return header + "\n" + "\n".join(rows) + "\n" + cols + robot_info + order_info + legend + "\n"
