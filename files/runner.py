from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type
import time

from openenv.core.base import BaseEnvironment
from openenv.agents.base_agent import BaseAgent


@dataclass
class EpisodeStats:
    episode: int
    total_reward: float
    steps: int
    duration_sec: float
    info: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return (
            f"Episode {self.episode:3d} | "
            f"reward={self.total_reward:+8.3f} | "
            f"steps={self.steps:4d} | "
            f"time={self.duration_sec:.2f}s | "
            f"{' | '.join(f'{k}={v}' for k, v in self.info.items())}"
        )


class Runner:

    def __init__(
        self,
        env: BaseEnvironment,
        agent: BaseAgent,
        render: bool = False,
        render_every: int = 1,
        step_callback: Optional[Callable] = None,
        episode_callback: Optional[Callable] = None,
    ):
        self.env = env
        self.agent = agent
        self.render = render
        self.render_every = render_every
        self.step_callback = step_callback
        self.episode_callback = episode_callback
        self.history: List[EpisodeStats] = []

    def run_episode(self, episode_num: int = 0, verbose: bool = True) -> EpisodeStats:
        state = self.env.reset()
        self.agent.on_reset(state)
        total_reward = 0.0
        step = 0
        t0 = time.time()
        final_info: Dict[str, Any] = {}

        while True:
            if self.render and episode_num % self.render_every == 0:
                rendered = self.env.render()
                if rendered:
                    print(rendered)

            action = self.agent.act(state)
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            self.agent.on_step(state, action, reward, next_state, done, info)

            if self.step_callback:
                self.step_callback(step, state, action, reward, next_state, done, info)

            state = next_state
            step += 1
            final_info = info

            if done:
                break

        duration = time.time() - t0
        stats = EpisodeStats(
            episode=episode_num,
            total_reward=total_reward,
            steps=step,
            duration_sec=duration,
            info={
                "completed": final_info.get("total_completed", "n/a"),
                "failed": final_info.get("total_failed", "n/a"),
            },
        )
        self.history.append(stats)

        if verbose:
            print(stats)

        if self.episode_callback:
            self.episode_callback(stats)

        return stats

    def run(self, num_episodes: int, verbose: bool = True) -> List[EpisodeStats]:
        for ep in range(num_episodes):
            self.run_episode(episode_num=ep, verbose=verbose)
        return self.history

    def summary(self) -> Dict[str, float]:
        if not self.history:
            return {}
        rewards = [e.total_reward for e in self.history]
        steps = [e.steps for e in self.history]
        return {
            "episodes": len(self.history),
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "mean_steps": sum(steps) / len(steps),
        }
