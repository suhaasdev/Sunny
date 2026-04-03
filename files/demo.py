import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from openenv import (
    EnvRegistry,
    RandomAgent,
    RuleBasedWarehouseAgent,
    Runner,
)


SEPARATOR = "=" * 60


def demo_random_agent_warehouse():
    print(f"\n{SEPARATOR}")
    print("  DEMO 1: Random Agent — SmartWarehouse-v1")
    print(f"{SEPARATOR}")

    env = EnvRegistry.make("SmartWarehouse-v1", {
        "grid_size": 8,
        "num_robots": 3,
        "num_shelves": 10,
        "max_steps": 30,
        "seed": 7,
    })

    agent = RandomAgent(env, seed=42)
    state = env.reset()

    print(env.render())
    print(f"  Observation vector shape: {env.to_vector().shape}")
    print(f"  Action space: {env.action_space.description}")
    print(f"  Obs space:    {env.observation_space.description}\n")

    total_reward = 0.0
    for step in range(6):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        print(f"  Step {step+1:02d} | actions={info['actions']}")
        print(f"         reward={reward:+.4f} | cumulative={total_reward:+.4f}")
        print(f"         completed={info['completed_orders']} | failed={info['failed_orders']}")
        print(f"         batteries={info['robot_batteries']}")
        print(f"         pending_orders={info['pending_orders']}")
        print()
        state = next_state
        if done:
            print("  [Episode complete]")
            break

    print(env.render())


def demo_rule_based_agent():
    print(f"\n{SEPARATOR}")
    print("  DEMO 2: Rule-Based Agent — SmartWarehouse-v1 (5 episodes)")
    print(f"{SEPARATOR}")

    env = EnvRegistry.make("SmartWarehouse-v1", {
        "grid_size": 8,
        "num_robots": 3,
        "num_shelves": 10,
        "max_steps": 100,
        "order_arrival_rate": 0.3,
        "seed": 42,
    })

    agent = RuleBasedWarehouseAgent(env, seed=0)
    runner = Runner(env, agent, render=False)

    print("\n  Running 5 episodes with heuristic agent...\n")
    stats = runner.run(num_episodes=5)
    print(f"\n  Summary: {runner.summary()}")


def demo_traffic_control():
    print(f"\n{SEPARATOR}")
    print("  DEMO 3: Random Agent — TrafficControl-v1")
    print(f"{SEPARATOR}")

    env = EnvRegistry.make("TrafficControl-v1", {
        "num_intersections": 4,
        "max_steps": 20,
        "seed": 1,
    })

    agent = RandomAgent(env, seed=99)
    state = env.reset()

    print(env.render())
    total_reward = 0.0

    for step in range(6):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state

        print(f"  Step {step+1:02d} | actions={action}")
        print(f"         reward={reward:+.4f} | cumulative={total_reward:+.4f}")
        print(f"         wait={info['total_wait']:.0f} | throughput={info['total_throughput']}")
        print(f"         queues={info['queues']}")
        print()
        if done:
            break

    print(env.render())


def demo_partial_observability():
    print(f"\n{SEPARATOR}")
    print("  DEMO 4: Partial Observability Mode")
    print(f"{SEPARATOR}")

    env = EnvRegistry.make("SmartWarehouse-v1", {
        "partial_observability": True,
        "obs_radius": 2,
        "max_steps": 10,
        "seed": 5,
    })

    state = env.reset()
    print(f"  Robot 0 visible neighbors: {[r for r in state['robots'][0].get('visible_robots', [])]}")
    print(f"  State keys: {list(state.keys())}")
    print("  [Partial obs enabled — each robot only sees within radius 2]\n")


def demo_registry():
    print(f"\n{SEPARATOR}")
    print("  DEMO 5: Environment Registry")
    print(f"{SEPARATOR}")
    print(f"  Registered environments: {EnvRegistry.list_envs()}")

    for name in EnvRegistry.list_envs():
        env = EnvRegistry.make(name)
        env.reset()
        print(f"  {name}")
        print(f"    obs_space : {env.observation_space.shape}")
        print(f"    act_space : discrete={env.action_space.discrete_n}, shape={env.action_space.shape}")
        print(f"    max_steps : {env.config.get('max_steps', 'n/a')}")


if __name__ == "__main__":
    demo_random_agent_warehouse()
    demo_rule_based_agent()
    demo_traffic_control()
    demo_partial_observability()
    demo_registry()

    print(f"\n{SEPARATOR}")
    print("  All demos complete.")
    print(f"{SEPARATOR}\n")
