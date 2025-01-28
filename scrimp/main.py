from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats as ss
from od_mstar3 import od_mstar
import tqdm

import time
import copy
import random
import dataclasses

from model import ScrimpNet
from params import *
from environment import Environment, EnvironmentParams
from model import ScrimpNetOutputs
from od_mstar3.col_set_addition import NoSolutionError


@dataclasses.dataclass
class Experiences:
    env: Environment
    action: torch.Tensor
    out: ScrimpNetOutputs
    extrinsic_rewards: torch.Tensor
    intrinsic_rewards: torch.Tensor
    blocking_rewards: torch.Tensor


class Agents:
    def __init__(self, save_attention: bool = False):
        self.model = ScrimpNet(save_attention).to(DEV)
        self.num_agents = None
        self.last_action = None
        self.state = None
        self.message = None
        self.last_intrinsic_reward = None
        self.last_extrinsic_reward = None
        self.episodic_buffer = None
        self.episodic_buffer_ptrs = None
        self.optim = torch.optim.Adam(self.model.parameters(), lr=HP.lr)
        self.experience_buffer = []
        self.action_cache = None

    def reset(self, num_agents):
        self.num_agents = num_agents
        self.last_action = torch.zeros(num_agents, dtype=torch.long, device=DEV)
        self.state = (
            torch.zeros(num_agents, AP.memory_dim, device=DEV),
            torch.zeros(num_agents, AP.memory_dim, device=DEV),
        )
        self.message = torch.zeros((num_agents, AP.message_feature_dim), device=DEV)
        self.last_intrinsic_reward = torch.zeros(num_agents, device=DEV)
        self.last_extrinsic_reward = torch.zeros(num_agents, device=DEV)
        self.episodic_buffer = torch.zeros((num_agents, AP.episodic_buffer_size, 2), device=DEV)
        self.episodic_buffer_ptrs = torch.zeros(num_agents, 1, device=DEV, dtype=torch.long)

    def write_to_episodic_buffer(self, agent_idx, x, y):
        ptr = self.episodic_buffer_ptrs[agent_idx].item()
        if ptr < AP.episodic_buffer_size:
            self.episodic_buffer[agent_idx, ptr] = torch.tensor([x, y], device=DEV)
            self.episodic_buffer_ptrs[agent_idx] += 1
        else:
            ptr = random.randrange(AP.episodic_buffer_size)
            self.episodic_buffer[agent_idx, ptr] = torch.tensor([x, y], device=DEV)

    def query_episodic_buffer(self, agent_id, x, y):
        buf = self.episodic_buffer[agent_id]
        ptr = self.episodic_buffer_ptrs[agent_id].item()
        dmin = float("inf")
        for i in range(ptr):
            bx, by = buf[i]
            d = ((x - bx) ** 2 + (y - by) ** 2) ** 0.5
            if d < dmin:
                dmin = d
        return dmin

    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.optim.load_state_dict(checkpoint["optim"])

    def load_raw(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint)

    @torch.no_grad()
    def act(self, env: Environment):
        env = copy.deepcopy(env)
        obs, observed_goal_info = env.observe()

        num_agents = self.num_agents

        obs = torch.tensor(obs, device=DEV, dtype=torch.float32)
        goal_info = torch.zeros((num_agents, AP.goal_in_dim), device=DEV)

        for i in range(num_agents):
            episodic_buffer_data = self.query_episodic_buffer(i, *env.agent_positions[i])
            if episodic_buffer_data > HP.episodic_buffer_dist:
                self.write_to_episodic_buffer(i, *env.agent_positions[i])

        goal_info[:, :3] = torch.tensor(observed_goal_info)
        goal_info[:, 3] = self.last_extrinsic_reward
        goal_info[:, 4] = self.last_intrinsic_reward
        goal_info[:, 5] = torch.tensor([self.query_episodic_buffer(i, *env.agent_positions[i]) for i in range(num_agents)])
        goal_info[:, 6] = self.last_action

        out: ScrimpNetOutputs = self.model(obs, goal_info, self.state, self.message)
        self.message = out.messages
        self.state = out.state
        rewards = np.zeros(num_agents)

        for i in range(512):
            mask = np.zeros((num_agents, 5), dtype=bool)

            def resample():
                logits = out.policy_logits
                logits[mask] = -torch.inf
                action_dist = torch.distributions.Categorical(logits=logits / AP.temperature)
                return action_dist.sample().cpu().numpy()

            action = resample()
            locked = np.zeros(num_agents, dtype=bool)
            new_positions = copy.copy(env.agent_positions)

            while True:
                all_ok = True
                for i in range(num_agents):
                    x, y, ok = env.go(*env.agent_positions[i], action[i])
                    new_positions[i] = x, y
                    if not ok:
                        rewards[i] += HP.obstacle_reward
                        mask[i, action[i]] = True
                        all_ok = False
                if all_ok:
                    break
                action = resample()

            agents_map = np.full((env.size, env.size), object, dtype=object)
            for x in range(env.size):
                for y in range(env.size):
                    agents_map[x, y] = []

            for i, (x, y) in enumerate(new_positions):
                agents_map[x, y].append(i)

            conflicting = []

            for x in range(env.size):
                for y in range(env.size):
                    if len(agents_map[x, y]) > 1:
                        conflicting.append(agents_map[x, y])

            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    if new_positions[i] == env.agent_positions[j] and new_positions[j] == env.agent_positions[i]:
                        conflicting.append([i, j])

            is_conflicting = [False] * num_agents

            for conflict in conflicting:
                for agent in conflict:
                    is_conflicting[agent] = True

            for i in range(num_agents):
                if not is_conflicting[i]:
                    locked[i] = True
                else:
                    rewards[i] += HP.collision_reward

            fixed = env.fix_actions(action, locked)
            if fixed is None:
                continue

            for conflict in conflicting:
                current_value = out.intrinsic_value.sum() + out.extrinsic_value.sum() + out.blocking.sum()
                values = []
                num_unlocked = 0
                for agent in conflict:
                    if not locked[agent]:
                        locked[agent] = True
                        fixed = env.fix_actions(action, locked)
                        locked[agent] = False
                        if fixed is not None:
                            num_unlocked += 1
                            temp_env = copy.deepcopy(env)
                            temp_env.commit(fixed)
                            obs, observed_goal_info = temp_env.observe()
                            obs = torch.tensor(obs, device=DEV, dtype=torch.float32)
                            goal_info = torch.zeros((num_agents, AP.goal_in_dim), device=DEV)
                            goal_info[:, :3] = torch.tensor(observed_goal_info)
                            goal_info[:, 3] = torch.tensor(rewards)
                            goal_info[:, 4] = torch.zeros(num_agents)
                            goal_info[:, 5] = torch.tensor(
                                [self.query_episodic_buffer(i, *temp_env.agent_positions[i]) for i in range(num_agents)]
                            )
                            goal_info[:, 6] = torch.tensor(action)
                            new_out = self.model(obs, goal_info, self.state, self.message)
                            value = new_out.intrinsic_value.sum() + new_out.extrinsic_value.sum() + new_out.blocking.sum()

                            values.append(value)

                if num_unlocked == 0:
                    continue

                goal_distances = []
                for i in range(num_agents):
                    x, y = new_positions[i]
                    goal_distances.append(((x - env.goal_positions[i][0]) ** 2 + (y - env.goal_positions[i][1]) ** 2) ** 0.5)
                total_goal_distance = sum(goal_distances)

                diffs = [value - current_value for value in values]
                for i in range(len(diffs)):
                    diffs[i] += HP.priority_mu * goal_distances[conflict[i]] / max(total_goal_distance, 1)
                priority_logits = torch.tensor(diffs)
                winner = torch.distributions.Categorical(logits=priority_logits).sample().item()

                locked[conflict[winner]] = True
                fixed = env.fix_actions(action, locked)
                if fixed is None:
                    locked[conflict[winner]] = False
                    continue

            break
        else:
            return None

        action = env.fix_actions(action, locked)
        assert action is not None
        self.last_action = torch.tensor(action, device=DEV)

        rewards = np.zeros(num_agents)
        for i in range(num_agents):
            if env.goal_positions[i] == env.agent_positions[i]:
                if action[i]:
                    rewards[i] += HP.goal_idle_reward
                else:
                    rewards[i] += HP.action_reward
            else:
                rewards[i] += HP.action_reward

        resulting_env = copy.deepcopy(env)
        self.action_cache = action
        blocking_rewards = resulting_env.commit(action)

        intrinsic_rewards = torch.zeros(num_agents, device=DEV)
        for i in range(num_agents):
            x, y = env.agent_positions[i]
            episodic = self.query_episodic_buffer(i, x, y)
            in_recent_area = float(episodic < HP.intrinsic_reward_dist)
            intrinsic_rewards[i] = HP.intrinsic_reward * (HP.intrinsic_reward_beta - in_recent_area)

        self.experience_buffer.append(Experiences(
            env=env,
            action=self.last_action,
            out=out,
            extrinsic_rewards=torch.tensor(rewards, device=DEV),
            intrinsic_rewards=intrinsic_rewards,
            blocking_rewards=torch.tensor(blocking_rewards, device=DEV),
        ))

        self.last_intrinsic_reward = intrinsic_rewards
        self.last_extrinsic_reward = torch.tensor(rewards, device=DEV)
        self.last_action = torch.tensor(action, device=DEV)

        return resulting_env

    def train_imitation_epoch(self):
        env = Environment.generate(EnvironmentParams(
            # size=random.choice([10, 25, 40]),
            size=10,
            num_agents=4,
            obstacle_prob=np.clip(ss.triang(c=0.66, loc=0.25, scale=0.25).rvs(), 0.0, 0.5),
        ))
        self.reset(env.num_agents)

        try:
            paths = od_mstar.find_path(
                env.obstacles,
                env.agent_positions,
                env.goal_positions,
                time_limit=10,
            )
        except:
            return

        actions = np.zeros((env.num_agents, len(paths) - 1), dtype=int)
        for i in range(env.num_agents):
            for t in range(len(paths) - 1):
                for action in range(5):
                    x, y, ok = env.go(*paths[t][i], action)
                    if (x, y) == paths[t + 1][i]:
                        actions[i, t] = action
                        break

        loss = torch.tensor(0.0)

        self.optim.zero_grad()
        for t in range(len(paths) - 1):
            env = copy.deepcopy(env)
            obs, observed_goal_info = env.observe()

            num_agents = self.num_agents

            obs = torch.tensor(obs, device=DEV, dtype=torch.float32)
            goal_info = torch.zeros((num_agents, AP.goal_in_dim), device=DEV)

            for i in range(num_agents):
                episodic_buffer_data = self.query_episodic_buffer(i, *env.agent_positions[i])
                if episodic_buffer_data > HP.episodic_buffer_dist:
                    self.write_to_episodic_buffer(i, *env.agent_positions[i])

            goal_info[:, :3] = torch.tensor(observed_goal_info)
            goal_info[:, 3] = self.last_extrinsic_reward
            goal_info[:, 4] = self.last_intrinsic_reward
            goal_info[:, 5] = torch.tensor([self.query_episodic_buffer(i, *env.agent_positions[i]) for i in range(num_agents)])
            goal_info[:, 6] = self.last_action

            out: ScrimpNetOutputs = self.model(obs, goal_info, self.state, self.message)
            one_hot = torch.zeros((num_agents, 5), device=DEV)
            for i in range(num_agents):
                one_hot[i, actions[i, t]] = 1.0

            loss = loss + F.cross_entropy(out.policy_logits, one_hot)
            self.state = out.state
            self.message = out.messages
            self.last_intrinsic_reward = torch.zeros(num_agents)
            self.last_extrinsic_reward = torch.zeros(num_agents)
            self.last_action = torch.tensor(actions[:, t], device=DEV)

        loss.backward()
        self.optim.step()

    def pathfind(self, env: Environment, step_limit: int = 10 ** 9):
        self.reset(env.num_agents)
        actions = []
        for i in range(1, step_limit):
            env = self.act(env)
            actions.append(self.action_cache)
            if env.is_solved():
                return actions, True
        return actions, False


if __name__ == "__main__":
    agents = Agents()
    agents.load_raw("scrimp/final/checkpoint.pkl")

    # env = Environment.generate(EnvironmentParams(
    #     size=10,
    #     num_agents=8,
    #     obstacle_prob=0.2,
    # ))
    # env = Environment(np.array([
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    # ]), [(0, 4), (0, 0), (2, 2)], [(4, 0), (4, 4), (2, 2)])
    env = Environment(np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]), [(2, 0), (3, 0)], [(2, 3), (3, 3)])

    print(env)
    agents.reset(env.num_agents)
    for i in range(5):
        env = agents.act(env)
        print(env)
        time.sleep(0.1)

    # actions, success = agents.pathfind(env, step_limit=10)
