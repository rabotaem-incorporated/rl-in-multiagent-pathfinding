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
import functools
import re

torch.manual_seed(config.test_seed)
np.random.seed(config.test_seed)
random.seed(config.test_seed)
DEVICE = torch.device("cpu")
torch.set_num_threads(1)


def run_single_test(test, original=False):
    mp, agents_xy, targets_xy, network = test
    env = Environment(from_map=True, mp=mp, agents_pos=agents_xy, goals_pos=targets_xy)
    network.reset()
    obs, last_act, pos = env.observe()

    done = False
    step = 0
    num_comm = 0
    hist = []
    comm_hist = []
    while not done and env.steps < config.max_episode_length:
        actions, _, _, _, comm_mask = network.step(
            torch.as_tensor(obs.astype(np.float32)).to(DEVICE),
            torch.as_tensor(last_act.astype(np.float32)).to(DEVICE),
            torch.as_tensor(pos.astype(int)),
        )
        (obs, last_act, pos), _, done, _ = (
            env.step_orig(actions) if original else env.step(actions)
        )
        step += 1
        hist.append(env.agents_pos)
        comm_hist.append(comm_mask)
        num_comm += np.sum(comm_mask)

    return np.all(env.agents_pos == env.goals_pos), step, num_comm, hist, comm_hist


def run_multiple_tests(tests, original=False):
    pool = mp.Pool(mp.cpu_count() // 2)
    ret = pool.starmap(run_single_test, zip(tests, [original] * len(tests)))

    success, steps, num_comm, hist, comm_hist = zip(*ret)

    print("Success rate: {:.2f}%".format(sum(success) / len(success) * 100))
    print("Average step: {}".format(sum(steps) / len(steps)))
    print("Communication times: {}".format(sum(num_comm) / len(num_comm)))

    return success, steps, num_comm, hist, comm_hist


def run_tests_from_file(path, network, original=False):
    print(f"Running tests from {path}")
    with open(path, "rb") as f:
        tests = pickle.load(f)
    tests = [(*test, network) for test in tests]
    if path.split("_")[-1] == "1.pth":
        return run_single_test_and_animate(
            tests[0], f"./tests/animations/{path.split('/')[-1].split('.')[0]}/animation.svg"
        )
    return run_multiple_tests(tests, original)


def run_single_test_and_animate(test, save_path):
    mp, agents_xy, targets_xy, network = test
    env = Environment(from_map=True, mp=mp, agents_pos=agents_xy, goals_pos=targets_xy)
    network.reset()
    obs, last_act, pos = env.observe()

    done = False
    step = 0
    num_comm = 0
    hist = []
    comm_hist = []
    mp = env.mp
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
        hist.append(env.agents_pos)
        comm_hist.append(comm_mask)
        num_comm += np.sum(comm_mask)
        cur_pos = np.copy(env.agents_pos)
        real_actions = np.zeros_like(actions)
        delta = cur_pos - prev_pos
        for i in range(len(actions)):
            real_actions[i] = config.INV_ACTIONS[tuple(delta[i])]
        prev_pos = cur_pos
        steps.append(real_actions.tolist())
    if not os.path.exists(save_path):
        os.mkdir('/'.join(save_path.split('/')[:-1]))
    animate(
        mp,
        agents_xy,
        targets_xy,
        np.array(steps),
        save_path=save_path,
    )
    print(f"success: {np.all(env.agents_pos == env.goals_pos)}")
    print(f"Step: {step}")
    print(f"Communication times: {num_comm}")
    return np.all(env.agents_pos == env.goals_pos),step,num_comm,hist, comm_hist
    


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
    # res = run_tests_from_file('/home/justermak/study/rl-in-multiagent-pathfinding/tests/test_set/32x32size_32agents_0.3density_30.pth', network)
    # for filename in os.listdir("./tests/test_set"):
    #     if filename.endswith("30.pth"):
    #         results = run_tests_from_file(f"./tests/test_set/{filename}", network, original=True)
    #         pickle.dump(results, open(f"./tests/results/{filename}", "wb"))
    for filename in os.listdir("./tests/test_set"):
        if re.search(r'from_map', filename) is not None:
            results = run_tests_from_file(f"./tests/test_set/{filename}", network, original=False)
            pickle.dump(results, open(f"./tests/results_v2/{filename}", "wb"))
            results = run_tests_from_file(f"./tests/test_set/{filename}", network, original=True)
            pickle.dump(results, open(f"./tests/results/{filename}", "wb"))
