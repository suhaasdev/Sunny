from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging

from openenv.core.base import BaseEnvironment, EnvRegistry, SpaceSpec, StepResult

DEFAULT_CONFIG = {
    "num_intersections": 4,
    "max_steps": 200,
    "max_queue_length": 20,
    "arrival_rate_mean": 3.0,
    "arrival_rate_std": 1.0,
    "green_min_duration": 2,
    "green_max_duration": 10,
    "yellow_duration": 1,
    "noise_std": 0.1,
    "seed": 0,
}

PHASE_NS = 0
PHASE_EW = 1
PHASE_YELLOW = 2


@dataclass
class Intersection:
    iid: int
    phase: int = PHASE_NS
    phase_timer: int = 0
    queue_ns: int = 0
    queue_ew: int = 0
    waiting_time_ns: float = 0.0
    waiting_time_ew: float = 0.0
    throughput: int = 0


@EnvRegistry.register("TrafficControl-v1")
class TrafficControlEnv(BaseEnvironment):

    def __init__(self, config: Dict[str, Any], log_level: int = logging.WARNING):
        cfg = {**DEFAULT_CONFIG, **config}
        super().__init__(cfg, log_level)
        self.n = cfg["num_intersections"]
        self.max_steps = cfg["max_steps"]
        self.max_queue = cfg["max_queue_length"]
        self.arr_mean = cfg["arrival_rate_mean"]
        self.arr_std = cfg["arrival_rate_std"]
        self.green_min = cfg["green_min_duration"]
        self.green_max = cfg["green_max_duration"]
        self.yellow_dur = cfg["yellow_duration"]
        self.noise_std = cfg["noise_std"]

    @property
    def observation_space(self) -> SpaceSpec:
        dim = self.n * 5
        return SpaceSpec(
            dtype="float32",
            shape=(dim,),
            low=np.zeros(dim, dtype="float32"),
            high=np.ones(dim, dtype="float32") * self.max_queue,
            description="Queue lengths + phases per intersection"
        )

    @property
    def action_space(self) -> SpaceSpec:
        return SpaceSpec(
            dtype="int32",
            shape=(self.n,),
            discrete_n=2,
            description="Per-intersection: 0=keep current phase, 1=switch phase"
        )

    def reset(self) -> Dict[str, Any]:
        self._pre_reset()
        rng = self.np_random
        self.intersections = [
            Intersection(
                iid=i,
                phase=int(rng.integers(0, 2)),
                phase_timer=int(rng.integers(self.green_min, self.green_max // 2)),
                queue_ns=int(rng.integers(0, 5)),
                queue_ew=int(rng.integers(0, 5)),
            )
            for i in range(self.n)
        ]
        return self.state()

    def step(self, actions: List[int]) -> StepResult:
        if len(actions) != self.n:
            raise ValueError(f"Expected {self.n} actions, got {len(actions)}")

        total_wait = 0.0
        total_throughput = 0
        penalties = 0.0

        for i, inter in enumerate(self.intersections):
            action = actions[i]
            rng = self.np_random

            arrivals_ns = max(0, int(rng.normal(self.arr_mean, self.arr_std)))
            arrivals_ew = max(0, int(rng.normal(self.arr_mean, self.arr_std)))
            inter.queue_ns = min(self.max_queue, inter.queue_ns + arrivals_ns)
            inter.queue_ew = min(self.max_queue, inter.queue_ew + arrivals_ew)

            if inter.phase == PHASE_YELLOW:
                inter.phase_timer += 1
                if inter.phase_timer >= self.yellow_dur:
                    inter.phase = PHASE_EW if action == 1 else PHASE_NS
                    inter.phase_timer = 0
            else:
                if action == 1 and inter.phase_timer >= self.green_min:
                    inter.phase = PHASE_YELLOW
                    inter.phase_timer = 0
                else:
                    inter.phase_timer += 1

                if inter.phase == PHASE_NS and inter.queue_ns > 0:
                    discharged = min(inter.queue_ns, 3)
                    inter.queue_ns -= discharged
                    inter.throughput += discharged
                    total_throughput += discharged
                elif inter.phase == PHASE_EW and inter.queue_ew > 0:
                    discharged = min(inter.queue_ew, 3)
                    inter.queue_ew -= discharged
                    inter.throughput += discharged
                    total_throughput += discharged

            inter.waiting_time_ns += inter.queue_ns
            inter.waiting_time_ew += inter.queue_ew
            total_wait += inter.queue_ns + inter.queue_ew

            if inter.queue_ns >= self.max_queue or inter.queue_ew >= self.max_queue:
                penalties += 2.0

        reward = (
            0.1 * total_throughput
            - 0.05 * total_wait
            - penalties
            + float(self.np_random.normal(0, self.noise_std))
        )

        done = self._step_count >= self.max_steps

        info = {
            "total_wait": total_wait,
            "total_throughput": total_throughput,
            "penalties": penalties,
            "queues": [(x.queue_ns, x.queue_ew) for x in self.intersections],
        }

        result = StepResult(next_state=self.state(), reward=reward, done=done, info=info)
        self._post_step(actions, result)
        return result

    def state(self) -> Dict[str, Any]:
        return {
            "intersections": [
                {
                    "id": x.iid,
                    "phase": ["NS_GREEN", "EW_GREEN", "YELLOW"][x.phase],
                    "phase_timer": x.phase_timer,
                    "queue_ns": x.queue_ns,
                    "queue_ew": x.queue_ew,
                    "throughput": x.throughput,
                }
                for x in self.intersections
            ],
            "step": self._step_count,
        }

    def to_vector(self) -> np.ndarray:
        vec = []
        for x in self.intersections:
            vec += [
                x.queue_ns / self.max_queue,
                x.queue_ew / self.max_queue,
                float(x.phase == PHASE_NS),
                float(x.phase == PHASE_EW),
                x.phase_timer / self.green_max,
            ]
        return np.array(vec, dtype=np.float32)

    def render(self, mode: str = "ansi") -> str:
        lines = [f"\n{'='*40}", f"  TrafficControl Step {self._step_count}", f"{'='*40}"]
        for x in self.intersections:
            phase_str = ["NS🟢", "EW🟢", "🟡"][x.phase]
            ns_bar = "█" * x.queue_ns + "░" * (self.max_queue - x.queue_ns)
            ew_bar = "█" * x.queue_ew + "░" * (self.max_queue - x.queue_ew)
            lines.append(
                f"  INT {x.iid} [{phase_str} t={x.phase_timer}]"
                f"  NS:[{ns_bar[:15]}] {x.queue_ns:2d}"
                f"  EW:[{ew_bar[:15]}] {x.queue_ew:2d}"
                f"  thru={x.throughput}"
            )
        lines.append(f"{'='*40}\n")
        return "\n".join(lines)
