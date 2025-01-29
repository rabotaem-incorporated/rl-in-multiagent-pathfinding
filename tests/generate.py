import sys
import os

sys.path.append(os.getcwd())
import DCC_implementation.config as config
from DCC_implementation.environment import Environment
import numpy as np
import pickle
import random

np.random.seed(config.test_seed)
random.seed(config.test_seed)

def create_single_test(map_x, map_y, num_agents, density):
    name = f"./tests/test_set/{map_x}x{map_y}size_{num_agents}agents_{density}density_1.pth"
    env = Environment(
        obstacle_density=density, num_agents=num_agents, map_size=(map_x, map_y)
    )
    mp, agents_xy, targets_xy = (
        np.copy(env.mp),
        np.copy(env.agents_pos),
        np.copy(env.goals_pos),
    )
    with open(name, "wb") as f:
        pickle.dump([(mp, agents_xy, targets_xy)], f)

    return name


def create_single_test_from_map(mp, agents_xy, targets_xy):
    map_x, map_y = mp.shape
    num_agents = agents_xy.shape[0]
    name = f"./tests/test_set/{map_x}x{map_y}size_{num_agents}agents_from_map_1.pth"
    with open(name, "wb") as f:
        pickle.dump([(mp, agents_xy, targets_xy)], f)

    return name


def create_multiple_tests(map_x, map_y, num_agents, density, n_cases):
    name = f"./tests/test_set/{map_x}x{map_y}size_{num_agents}agents_{density}density_{n_cases}.pth"
    tests = []
    for i in range(n_cases):
        print(name, i)
        env = Environment(
            obstacle_density=density, num_agents=num_agents, map_size=(map_x, map_y)
        )
        tests.append((np.copy(env.mp), np.copy(env.agents_pos), np.copy(env.goals_pos)))
    with open(name, "wb") as f:
        pickle.dump(tests, f)

    return name


if __name__ == '__main__':
    for map_size in (32, 64, 96):
        for agents in (16, 32, 64, 96):
            for density in (0.2, 0.3, 0.4):
                create_multiple_tests(map_size, map_size, agents, density, 30)
    maps = [
        np.array(
            [
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
            ]
        ),
        np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ]),
        np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ]),
        np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ]),
    ]

    agents_xy = [
        np.array([[0, 0]]),
        np.array([[3, 0], [3, 6]]),
        np.array([[1, 0], [1, 6]]),
        np.array([[1, 0], [1, 6], [5, 0], [5, 6]]),
    ]

    targets_xy = [
        np.array([[7, 7]]),
        np.array([[3, 6], [3, 0]]),
        np.array([[1, 6], [1, 0]]),
        np.array([[5, 6], [5, 0], [1, 6], [1, 0]]),
    ]

    for test in zip(maps, agents_xy, targets_xy):
        create_single_test_from_map(*test)