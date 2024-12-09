import dataclasses
import numpy as np
import functools
import typing
import random
import pycosat

from od_mstar3 import od_mstar

from params import *


@dataclasses.dataclass
class EnvironmentParams:
    size: int
    obstacle_prob: float
    num_agents: int = 8

A_LOT = 10 ** 9

def astar(obstacles, start, goal, other_obstacles=None):
    obstacles = obstacles.copy()
    if other_obstacles:
        for x, y in other_obstacles:
            obstacles[x, y] = True

    try:
        path = od_mstar.find_path(obstacles, [start], [goal])
    except od_mstar.NoSolutionError:
        return None

    return path


class Environment:
    @classmethod
    def generate(cls, params: EnvironmentParams):
        obstacles = np.random.random(size=(params.size, params.size)) < params.obstacle_prob
        self = cls(obstacles, [], [])
        self.num_agents = params.num_agents
        self.size = params.size

        visited = np.zeros((params.size, params.size), dtype=bool)

        def connected_area(x, y) -> list:
            if self.obstacles[x, y] or visited[x, y]:
                return []

            visited[x, y] = True
            result = [(x, y)]

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if 0 <= x + dx < params.size and 0 <= y + dy < params.size:
                    result += connected_area(x + dx, y + dy)
            return result

        largest_connected_area = []
        for x in range(params.size):
            for y in range(params.size):
                area = connected_area(x, y)
                if len(area) > len(largest_connected_area):
                    largest_connected_area = area

        self.agent_positions = random.sample(list(largest_connected_area), self.num_agents)
        self.goal_positions = random.sample(list(largest_connected_area), self.num_agents)
        return self

    def __init__(self, obstacles: np.ndarray, agent_positions, goal_positions):
        self.num_agents = len(agent_positions)
        assert len(agent_positions) == len(goal_positions)
        self.size = obstacles.shape[0]
        assert obstacles.shape == (self.size, self.size)
        self.obstacles = obstacles
        self.agent_positions = agent_positions
        self.goal_positions = goal_positions

    def is_solved(self):
        return all(self.agent_positions[i] == self.goal_positions[i] for i in range(self.num_agents))

    def distances_from(self, x: int, y: int) -> np.ndarray:
        q = [(x, y, 0)]
        visited = np.zeros((self.size, self.size), dtype=bool)
        visited[x, y] = True
        distances = np.full((self.size, self.size), A_LOT, dtype=int)
        ptr = 0

        while ptr < len(q):
            x, y, d = q[ptr]
            ptr += 1

            distances[x, y] = d
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if (
                    0 <= x + dx < self.size and
                    0 <= y + dy < self.size and
                    not visited[x + dx, y + dy] and
                    not self.obstacles[x + dx, y + dy]
                ):
                    visited[x + dx, y + dy] = True
                    q.append((x + dx, y + dy, d + 1))

        return distances


    def observe(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        obs = np.zeros((self.num_agents, AP.num_observation_in_dim, AP.fov, AP.fov), dtype=float)
        goal_info = np.zeros((self.num_agents, 3), dtype=float)

        agent_map = np.zeros((self.size, self.size), dtype=int)
        for i, (x, y) in enumerate(self.agent_positions):
            agent_map[x, y] = i + 1

        goal_map = np.zeros((self.size, self.size), dtype=int)
        for i, (x, y) in enumerate(self.goal_positions):
            goal_map[x, y] = i + 1

        for agent in range(self.num_agents):
            x, y = self.agent_positions[agent]
            goal_x, goal_y = self.goal_positions[agent]

            goal_dist = self.distances_from(goal_x, goal_y)
            goal_dist = np.pad(goal_dist, (0, 1), constant_values=A_LOT)

            for dx in range(-AP.fov // 2, AP.fov // 2):
                for dy in range(-AP.fov // 2, AP.fov // 2):
                    if 0 <= x + dx < self.size and 0 <= y + dy < self.size:
                        i, j = dx + AP.fov // 2, dy + AP.fov // 2
                        obs[agent, 0, i, j] = self.obstacles[x + dx, y + dy]
                        obs[agent, 1, i, j] = goal_map[x + dx, y + dy] == agent + 1
                        obs[agent, 2, i, j] = goal_map[x + dx, y + dy] > 0
                        obs[agent, 3, i, j] = agent_map[x + dx, y + dy] > 0
                        obs[agent, 4, i, j] = goal_dist[x + dx + 1, y + dy] < goal_dist[x + dx, y + dy]
                        obs[agent, 5, i, j] = goal_dist[x + dx - 1, y + dy] < goal_dist[x + dx, y + dy]
                        obs[agent, 6, i, j] = goal_dist[x + dx, y + dy + 1] < goal_dist[x + dx, y + dy]
                        obs[agent, 7, i, j] = goal_dist[x + dx, y + dy - 1] < goal_dist[x + dx, y + dy]
                    else:
                        obs[agent, 0, dx + AP.fov // 2, dy + AP.fov // 2] = 1

            goal_dist = ((goal_x - x) ** 2 + (goal_y - y) ** 2) ** 0.5
            goal_info[agent, 0] = (goal_x - x) / goal_dist if goal_dist > 0 else 0
            goal_info[agent, 1] = (goal_y - y) / goal_dist if goal_dist > 0 else 0
            goal_info[agent, 2] = goal_dist

        return obs, goal_info

    def compute_blocking_reward(self) -> np.ndarray:
        # TODO: properly test
        times_blocking = np.zeros(self.num_agents, dtype=int)
        for i in range(self.num_agents):
            agents_in_fov = []
            for j in range(self.num_agents):
                if i == j:
                    continue
                x, y = self.agent_positions[j]
                if abs(x - self.agent_positions[i][0]) <= AP.fov // 2 and abs(y - self.agent_positions[i][1]) <= AP.fov // 2:
                    agents_in_fov.append((x, y))
            path = astar(self.obstacles, self.agent_positions[i], self.goal_positions[i], agents_in_fov)
            for j in range(self.num_agents):
                if self.agent_positions[j] in agents_in_fov:
                    other_agents_in_fov = [agent for agent in agents_in_fov if agent != self.agent_positions[j]]
                    excluded_path = astar(self.obstacles, self.agent_positions[j], self.goal_positions[j], other_agents_in_fov)
                    if (
                            excluded_path is None and path is not None or
                            path is not None and
                                len(excluded_path) >= len(path) + HP.blocking_path_inflation_threshold
                    ):
                        times_blocking[i] += 1
        return times_blocking * HP.blocking_reward

    def go(self, x, y, action):
        if action == 0:
            nx, ny = x, y
        elif action == 1:
            nx, ny = x + 1, y
        elif action == 2:
            nx, ny = x - 1, y
        elif action == 3:
            nx, ny = x, y + 1
        elif action == 4:
            nx, ny = x, y - 1
        else:
            raise ValueError('Invalid action')

        if 0 <= nx < self.size and 0 <= ny < self.size and not self.obstacles[nx, ny]:
            return nx, ny, True
        else:
            return x, y, False


    def fix_actions(self, actions: np.ndarray, locked: np.ndarray) -> np.ndarray | None:
        v = (np.arange(1, self.num_agents * self.size * self.size * 2 + 1, dtype=object)
             .reshape((self.num_agents, self.size, self.size, 2)))
        cnf = []
        for i in range(self.num_agents):
            for x in range(self.size):
                for y in range(self.size):
                    for t in range(2):
                        if self.obstacles[x, y]:
                            cnf.append([-v[i, x, y, 0]])
                            cnf.append([-v[i, x, y, 1]])
                        else:
                            for j in range(i):
                                cnf.append([-v[i, x, y, t], -v[j, x, y, t]])
        for i in range(self.num_agents):
            cnf.append([v[i, self.agent_positions[i][0], self.agent_positions[i][1], 0]])
            if locked[i]:
                nx, ny, ok = self.go(self.agent_positions[i][0], self.agent_positions[i][1], actions[i])
                if not ok:
                    return None
                cnf.append([v[i, nx, ny, 1]])
            else:
                clause = []
                for action in range(5):
                    nx, ny, ok = self.go(self.agent_positions[i][0], self.agent_positions[i][1], action)
                    if ok:
                        clause.append(v[i, nx, ny, 1])
                cnf.append(clause)

        for i in range(self.num_agents):
            for x in range(self.size):
                for y in range(self.size):
                    for action in range(1, 5):
                        nx, ny, ok = self.go(x, y, action)
                        if ok:
                            for j in range(i):
                                cnf.append([-v[i, x, y, 0], -v[i, nx, ny, 1], -v[j, nx, ny, 0], -v[j, x, y, 1]])

        # print(cnf)
        solution = pycosat.solve(cnf)
        if solution == 'UNSAT':
            return None

        result = np.full(self.num_agents, -1, dtype=int)
        for i in range(self.num_agents):
            for action in range(5):
                nx, ny, ok = self.go(self.agent_positions[i][0], self.agent_positions[i][1], action)
                if ok and v[i, nx, ny, 1] in solution:
                    result[i] = action
                    break

        return result

    def commit(self, actions):
        new_positions = [None] * self.num_agents
        for i in range(self.num_agents):
            x, y, ok = self.go(*self.agent_positions[i], actions[i])
            assert ok
            new_positions[i] = (x, y)

        for i in range(self.num_agents):
            for j in range(i):
                assert tuple(new_positions[i]) != tuple(new_positions[j])
                assert not (
                    tuple(self.agent_positions[i]) == tuple(new_positions[j]) and
                    tuple(self.agent_positions[j]) == tuple(new_positions[i])
                ), f"{self.agent_positions} {new_positions} {i} {j}"

        self.agent_positions = new_positions

        return self.compute_blocking_reward()

    def __str__(self):
        result = ""
        agent_map = np.zeros((self.size, self.size), dtype=int)
        for i, (x, y) in enumerate(self.agent_positions):
            agent_map[x, y] = i + 1
        goal_map = np.zeros((self.size, self.size), dtype=int)
        for i, (x, y) in enumerate(self.goal_positions):
            goal_map[x, y] = i + 1

        for x in range(self.size):
            for y in range(self.size):
                if self.obstacles[x, y]:
                    result += ' # '
                elif agent_map[x, y]:
                    result += f"<{agent_map[x, y]}>"
                elif goal_map[x, y]:
                    result += f" {goal_map[x, y]} "
                else:
                    result += ' . '
            result += '\n'
        return result


if __name__ == '__main__':
    env = Environment(EnvironmentParams(size=10, obstacle_prob=0.5))
    print(env.obstacles)
    print(env.agent_positions)
    print(env.goal_positions)
    print(env.observe())
    print(env.compute_blocking_reward())

    while True:
        actions = np.random.randint(0, 5, env.num_agents)
        print(actions)
        locked = np.zeros(env.num_agents, dtype=bool)
        locked[0] = True
        actions = env.fix_actions(actions, locked)
        if actions is not None:
            env.commit(actions)
        else:
            print('No solution')
            print(str(env))
