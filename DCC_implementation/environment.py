import random
from typing import List, Union, Tuple
import numpy as np
import scipy.stats as ss
import DCC_implementation.config as config
import sys

sys.setrecursionlimit(1000000)


class Environment:
    def __init__(
        self,
        num_agents: int = config.init_num_agents,
        map_size: Tuple[int, int] = config.init_map_size,
        obstacle_density: float = 0.3,
        obs_radius: int = config.obs_radius,
        reward_fn: dict = config.reward_fn,
        random=False,
        from_map=False,
        mp: np.ndarray = None,
        agents_pos: np.ndarray = None,
        goals_pos: np.ndarray = None,
    ):
        if from_map:
            self.mp = np.copy(mp)
            self.agents_pos = np.copy(agents_pos)
            self.goals_pos = np.copy(goals_pos)
            self.num_agents = agents_pos.shape[0]
            self.map_size = (mp.shape[0], mp.shape[1])
        else:
            if random:
                self.num_agents, self.map_size = np.random.choice(config.maps)
                self.obstacle_density = ss.triang(0.66, 0, 0.5).rvs()
            else:
                self.map_size = map_size
                self.num_agents = num_agents
                self.obstacle_density = obstacle_density
            partitions = []
            while (
                len(partitions) == 0
                or sum([len(partition) // 2 for partition in partitions])
                < self.num_agents
            ):
                self.mp = (
                    ss.bernoulli(self.obstacle_density)
                    .rvs(size=self.map_size)
                    .astype(int)
                )
                partitions = self.map_partition()
            self.agents_pos = np.empty((self.num_agents, 2), dtype=int)
            self.goals_pos = np.empty((self.num_agents, 2), dtype=int)

            partitions_to_sample_2d = [
                [i] * (len(partitions[i]) // 2) for i in range(len(partitions))
            ]
            partitions_to_sample = [j for i in partitions_to_sample_2d for j in i]
            agent_partition_idx = np.random.choice(
                partitions_to_sample, self.num_agents, replace=False
            )
            agent_idx_per_partition = [
                [i for i in range(num_agents) if agent_partition_idx[i] == j]
                for j in range(len(partitions))
            ]
            for i, agents in enumerate(agent_idx_per_partition):
                sz = 2 * len(agents)
                coords = [
                    partitions[i][j]
                    for j in np.random.choice(
                        range(len(partitions[i])), sz, replace=False
                    )
                ]
                for i, agent in enumerate(agents):
                    self.agents_pos[agent] = np.asarray(coords[2 * i], dtype=int)
                    self.goals_pos[agent] = np.asarray(coords[2 * i + 1], dtype=int)

        self.obs_radius = obs_radius
        self.reward_fn = reward_fn
        self.get_heuri_map()
        self.steps = 0
        self.last_actions = np.zeros((self.num_agents, 5), dtype=bool)

    def dfs(self, i, j):
        partition = []
        q = [(i, j)]
        while q:
            i, j = q.pop()
            if self.visited[i, j]:
                continue
            partition.append((i, j))
            self.visited[i, j] = True
            for action in config.ACTIONS[:-1]:
                new_i, new_j = i + action[0], j + action[1]
                if not (
                    new_i < 0
                    or new_j < 0
                    or new_i >= self.map_size[0]
                    or new_j >= self.map_size[1]
                    or self.visited[new_i, new_j]
                ):
                    q.append((new_i, new_j))
        return partition

    def map_partition(self):
        partitions = []
        self.visited = self.mp == 1
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if not self.visited[i, j]:
                    partitions.append(self.dfs(i, j))

        return partitions

    def get_heuri_map(self):
        dist_map = np.full((self.num_agents, *self.map_size), 1e9, dtype=np.int32)

        empty_pos = np.argwhere(self.mp == 0).tolist()
        empty_pos = set([tuple(pos) for pos in empty_pos])

        for i in range(self.num_agents):
            open = set()
            x, y = tuple(self.goals_pos[i])
            open.add((x, y))
            dist_map[i, x, y] = 0

            while open:
                x, y = open.pop()
                dist = dist_map[i, x, y]

                for dx, dy in config.ACTIONS[:-1]:
                    new_x, new_y = x + dx, y + dy
                    if (new_x, new_y) in empty_pos and dist_map[
                        i, new_x, new_y
                    ] > dist + 1:
                        dist_map[i, new_x, new_y] = dist + 1
                        open.add((new_x, new_y))

        self.heuri_map = np.zeros((self.num_agents, 4, *self.map_size), dtype=bool)

        for x, y in empty_pos:
            for i in range(self.num_agents):
                for dx, dy in config.ACTIONS[:-1]:
                    new_x, new_y = x + dx, y + dy
                    if (
                        new_x < 0
                        or new_x >= self.map_size[0]
                        or new_y < 0
                        or new_y >= self.map_size[1]
                    ):
                        continue
                    if dist_map[i, new_x, new_y] < dist_map[i, x, y]:
                        if dx == -1:
                            self.heuri_map[i, 0, x, y] = 1
                        elif dx == 1:
                            self.heuri_map[i, 1, x, y] = 1
                        elif dy == -1:
                            self.heuri_map[i, 2, x, y] = 1
                        else:
                            self.heuri_map[i, 3, x, y] = 1

        self.heuri_map = np.pad(
            self.heuri_map,
            (
                (0, 0),
                (0, 0),
                (self.obs_radius, self.obs_radius),
                (self.obs_radius, self.obs_radius),
            ),
        )

    def observe(self):
        # returns obs, last_actions, agents_pos
        # 1 - agents, 2 - obstacles, 3-6 - heuri
        r = 2 * self.obs_radius + 1
        obs = np.zeros(
            (self.num_agents, 6, r, r),
            dtype=bool,
        )

        obstacle_map = np.pad(self.mp, self.obs_radius)

        agent_map = np.zeros((self.map_size), dtype=bool)
        agent_map[self.agents_pos[:, 0], self.agents_pos[:, 1]] = 1
        agent_map = np.pad(agent_map, self.obs_radius)

        for i, agent_pos in enumerate(self.agents_pos):
            x, y = agent_pos
            obs[i, 0] = agent_map[x : x + r, y : y + r]
            obs[i, 0, self.obs_radius, self.obs_radius] = 0
            obs[i, 1] = obstacle_map[x : x + r, y : y + r]
            obs[i, 2:] = self.heuri_map[i, :, x : x + r, y : y + r]

        return obs, np.copy(self.last_actions), np.copy(self.agents_pos)

    def step(self, actions: List[int]):
        reward = np.empty(self.num_agents)
        next_pos = np.copy(self.agents_pos)
        undecided = set(range(self.num_agents))
        for i in list(undecided):
            if actions[i] == 4:
                if np.array_equal(self.agents_pos[i], self.goals_pos[i]):
                    reward[i] = self.reward_fn["stay_on_goal"]
                else:
                    reward[i] = self.reward_fn["stay_off_goal"]
                undecided.remove(i)
            else:
                next_pos[i] += config.ACTIONS[actions[i]]
                reward[i] = self.reward_fn["move"]

        for i in range(self.num_agents):
            if (
                np.any(next_pos[i] < 0)
                or next_pos[i][0] >= self.map_size[0]
                or next_pos[i][1] >= self.map_size[1]
            ):
                reward[i] = self.reward_fn["collision"]
                next_pos[i] = self.agents_pos[i]
                undecided.remove(i)
            elif self.mp[tuple(next_pos[i])] == 1:
                reward[i] = self.reward_fn["collision"]
                next_pos[i] = self.agents_pos[i]
                undecided.remove(i)

        for i in undecided.copy():
            ids = np.where(np.all(self.agents_pos == next_pos[i], axis=1))[0]
            if ids.size > 1:
                for j in ids:
                    reward[j] = self.reward_fn["collision"]
                    next_pos[j] = self.agents_pos[j]
                    if j in undecided:
                        undecided.remove(j)

        distinct_next_pos = set(map(tuple, next_pos))
        for pos in distinct_next_pos:
            conflict = np.where(np.all(next_pos == pos, axis=1))[0]
            if len(conflict) > 1:
                conflict = conflict.tolist()
                conflict.sort(
                    key=lambda x: self.agents_pos[x][0] * self.map_size[1]
                    + self.agents_pos[x][1]
                )
                conflict = np.array(conflict)
                conflict = conflict[1:]
                reward[conflict] = self.reward_fn["collision"]
                next_pos[conflict] = self.agents_pos[conflict]
                undecided = undecided - set(conflict)

        self.agents_pos = np.copy(next_pos)
        self.steps += 1

        if np.array_equal(self.agents_pos, self.goals_pos):
            done = True
            reward = [self.reward_fn["finish"] for _ in range(self.num_agents)]
        else:
            done = False

        info = {"step": self.steps - 1}

        self.last_actions = np.zeros((self.num_agents, 5), dtype=bool)
        self.last_actions[np.arange(self.num_agents), np.array(actions)] = 1

        return self.observe(), reward, done, info

    def step_orig(self, actions: List[int]):
        """
        actions:
            list of indices
                0 up
                1 down
                2 left
                3 right
                4 stay
        """

        assert (
            len(actions) == self.num_agents
        ), "only {} actions as input while {} agents in environment".format(
            len(actions), self.num_agents
        )
        assert all(
            [action_idx < 5 and action_idx >= 0 for action_idx in actions]
        ), "action index out of range"

        checking_list = list(range(self.num_agents))

        rewards = []
        next_pos = np.copy(self.agents_pos)

        # remove unmoving agent id
        for agent_id in checking_list.copy():
            if actions[agent_id] == 4:
                # unmoving
                if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                    rewards.append(self.reward_fn["stay_on_goal"])
                else:
                    rewards.append(self.reward_fn["stay_off_goal"])
                checking_list.remove(agent_id)
            else:
                # move
                next_pos[agent_id] += config.ACTIONS[actions[agent_id]]
                rewards.append(self.reward_fn["move"])

        # first round check, these two conflicts have the highest priority
        for agent_id in checking_list.copy():

            if np.any(next_pos[agent_id] < 0) or np.any(
                next_pos[agent_id] >= self.map_size[0]
            ):
                # agent out of mp range
                rewards[agent_id] = self.reward_fn["collision"]
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

            elif self.mp[tuple(next_pos[agent_id])] == 1:
                # collide obstacle
                rewards[agent_id] = self.reward_fn["collision"]
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

        # second round check, agent swapping conflict
        no_conflict = False
        while not no_conflict:

            no_conflict = True
            for agent_id in checking_list:

                target_agent_id = np.where(
                    np.all(next_pos[agent_id] == self.agents_pos, axis=1)
                )[0]

                if len(target_agent_id) > 0:

                    target_agent_id = target_agent_id.item()

                    if np.array_equal(
                        next_pos[target_agent_id], self.agents_pos[agent_id]
                    ):
                        assert (
                            target_agent_id in checking_list
                        ), "target_agent_id should be in checking list"

                        next_pos[agent_id] = self.agents_pos[agent_id]
                        rewards[agent_id] = self.reward_fn["collision"]

                        next_pos[target_agent_id] = self.agents_pos[target_agent_id]
                        rewards[target_agent_id] = self.reward_fn["collision"]

                        checking_list.remove(agent_id)
                        checking_list.remove(target_agent_id)

                        no_conflict = False
                        break

        # third round check, agent collision conflict
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:

                collide_agent_id = np.where(
                    np.all(next_pos == next_pos[agent_id], axis=1)
                )[0].tolist()
                if len(collide_agent_id) > 1:
                    # collide agent

                    # if all agents in collide agent are in checking list
                    all_in_checking = True
                    for id in collide_agent_id.copy():
                        if id not in checking_list:
                            all_in_checking = False
                            collide_agent_id.remove(id)

                    if all_in_checking:

                        collide_agent_pos = next_pos[collide_agent_id].tolist()
                        for pos, id in zip(collide_agent_pos, collide_agent_id):
                            pos.append(id)
                        collide_agent_pos.sort(
                            key=lambda x: x[0] * self.map_size[0] + x[1]
                        )

                        collide_agent_id.remove(collide_agent_pos[0][2])

                    next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                    for id in collide_agent_id:
                        rewards[id] = self.reward_fn["collision"]

                    for id in collide_agent_id:
                        checking_list.remove(id)

                    no_conflict = False
                    break

        self.agents_pos = np.copy(next_pos)

        self.steps += 1

        # check done
        if np.array_equal(self.agents_pos, self.goals_pos):
            done = True
            rewards = [self.reward_fn["finish"] for _ in range(self.num_agents)]
        else:
            done = False

        info = {"step": self.steps - 1}

        # make sure no overlapping agents
        assert np.unique(self.agents_pos, axis=0).shape[0] == self.num_agents

        # update last actions
        self.last_actions = np.zeros((self.num_agents, 5), dtype=bool)
        self.last_actions[np.arange(self.num_agents), np.array(actions)] = 1

        return self.observe(), rewards, done, info
