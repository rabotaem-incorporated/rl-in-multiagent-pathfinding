import sys
import os

sys.path.append(os.getcwd())
from test_visualizer.animate import animate
import DCC_implementation.config as config
from DCC_implementation.model import Network
from DCC_implementation.environment import Environment
import torch.multiprocessing as mp
import torch
import numpy as np
import pickle
import random

torch.manual_seed(config.test_seed)
np.random.seed(config.test_seed)
random.seed(config.test_seed)
DEVICE = torch.device("cpu")
torch.set_num_threads(1)


def create_single_test(map_x, map_y, num_agents, density):
    name = f"./DCC_test/test_set/{map_x}x{map_y}size_{num_agents}agents_{density}density_1.pth"
    env = Environment(obstacle_density=density, num_agents=num_agents, map_size=(map_x, map_y))
    map, agents_xy, targets_xy = (
        np.copy(env.mp),
        np.copy(env.agents_pos),
        np.copy(env.goals_pos),
    )
    with open(name, "wb") as f:
        pickle.dump([(map, agents_xy, targets_xy)], f)

    return name

def create_single_test_from_map(map, agents_xy, targets_xy):
    map_x, map_y = map.shape
    num_agents = agents_xy.shape[0]
    name = f"./DCC_test/test_set/{map_x}x{map_y}size_{num_agents}agents_from_map_1.pth"
    with open(name, "wb") as f:
        pickle.dump([(map, agents_xy, targets_xy)], f)

    return name

def create_multiple_tests(map_x, map_y, num_agents, density, n_cases):
    name = f"./DCC_test/test_set/{map_x}x{map_y}size_{num_agents}agents_{density}density_{n_cases}.pth"
    tests = []
    for i in range(n_cases):
        print(name, i)
        env = Environment(obstacle_density=density, num_agents=num_agents, map_size=(map_x, map_y))
        tests.append(
            (np.copy(env.mp), np.copy(env.agents_pos), np.copy(env.goals_pos))
        )
    with open(name, "wb") as f:
        pickle.dump(tests, f)

    return name


def run_single_test(test):
    mp, agents_xy, targets_xy, network = test
    env = Environment(from_map=True, mp=mp, agents_pos=agents_xy, goals_pos=targets_xy)
    network.reset()
    obs, last_act, pos = env.observe()

    done = False
    step = 0
    num_comm = 0
    while not done and env.steps < config.max_episode_length:
        actions, _, _, _, comm_mask = network.step(
            torch.as_tensor(obs.astype(np.float32)).to(DEVICE),
            torch.as_tensor(last_act.astype(np.float32)).to(DEVICE),
            torch.as_tensor(pos.astype(int)),
        )
        (obs, last_act, pos), _, done, _ = env.step(actions)
        step += 1
        num_comm += np.sum(comm_mask)

    return np.all(env.agents_pos == env.goals_pos), step, num_comm


def run_multiple_tests(tests):
    pool = mp.Pool(mp.cpu_count() // 2)
    ret = pool.map(run_single_test, tests)

    success, steps, num_comm = zip(*ret)

    print("Success rate: {:.2f}%".format(sum(success) / len(success) * 100))
    print("Average step: {}".format(sum(steps) / len(steps)))
    print("Communication times: {}".format(sum(num_comm) / len(num_comm)))
    
    return success, steps, num_comm


def run_tests_from_file(path, network):
    print(f"Running tests from {path}")
    with open(path, "rb") as f:
        tests = pickle.load(f)
    tests = [(*test, network) for test in tests]
    if path.split("_")[-1] == "1.pth":
        run_single_test_and_animate(tests[0], "./DCC_test/animations/animation.svg")
    else:
        run_multiple_tests(tests)


def run_single_test_and_animate(test, save_path):
    map, agents_xy, targets_xy, network = test
    env = Environment(from_map=True, map=map, agents_pos=agents_xy, goals_pos=targets_xy)
    network.reset()
    obs, last_act, pos = env.observe()

    done = False
    step = 0
    num_comm = 0
    map = env.mp
    agents_xy = env.agents_pos
    targets_xy = env.goals_pos
    steps = []
    prev_pos = env.agents_pos
    while not done and env.steps < config.max_episode_length:
        actions, _, _, _, comm_mask = network.step(
            torch.as_tensor(obs.astype(np.float32)).to(DEVICE),
            torch.as_tensor(last_act.astype(np.float32)).to(DEVICE),
            torch.as_tensor(pos.astype(int)),
        )
        (obs, last_act, pos), _, done, _ = env.step(actions)
        step += 1
        num_comm += np.sum(comm_mask)
        cur_pos = np.copy(env.agents_pos)
        real_actions = np.zeros_like(actions)
        delta = cur_pos - prev_pos
        for i in range(len(actions)):
            real_actions[i] = config.INV_ACTIONS[tuple(delta[i])]
        prev_pos = cur_pos
        steps.append(real_actions.tolist())

    animate(
        map,
        agents_xy,
        targets_xy,
        np.array(steps),
        save_path=save_path,
    )
    print(f"success: {np.all(env.agents_pos == env.goals_pos)}")
    print(f"Step: {step}")
    print(f"Communication times: {num_comm}")


def prepare():
    network = Network()
    network.eval()
    network.to(DEVICE)
    state_dict = torch.load(
        os.path.join(config.save_path, "128000.pth"),
        map_location=DEVICE,
        weights_only=False,
    )
    network.load_state_dict(state_dict)
    network.eval()

    return network


if __name__ == "__main__":
    network = prepare()
    for filename in os.listdir("./DCC_test/test_set"):
        if filename.endswith("30.pth"):
            results = run_tests_from_file(f"./DCC_test/test_set/{filename}", network)
            pickle.dump(results, open(f"./DCC_test/results_v2/{filename}", "wb"))
    # for map_size in (32, 64, 96):
    #     for agents in (16, 32, 64, 96):
    #         for density in (0.2, 0.3, 0.4):
    #             create_multiple_tests(map_size, map_size, agents, density, 30)
                