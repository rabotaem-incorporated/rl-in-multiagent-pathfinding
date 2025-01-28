import sys
import os

from main import *
import torch.multiprocessing as mp
import torch
import numpy as np
import pickle
import random
import tqdm
import functools
import dataclasses

SEED = 42
MAX_EPISODE_LENGTH = 512

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_num_threads(1)
PARALLEL = True

if PARALLEL:
    mp.set_start_method("spawn", force=True)


@dataclasses.dataclass
class Test:
    mp: np.ndarray
    agents_xy: np.ndarray
    targets_xy: np.ndarray
    agents: Agents


@dataclasses.dataclass
class TestResult:
    success: bool
    steps: int
    num_comm: int
    history: np.ndarray
    attentions: np.ndarray


def run_single_test(test: Test) -> TestResult:
    mp = test.mp
    agents_xy = list(map(tuple, test.agents_xy))
    targets_xy = list(map(tuple, test.targets_xy))
    agents: Agents = test.agents
    env = Environment(mp, agents_xy, targets_xy)
    agents.reset(env.num_agents)
    num_comms = 0
    solved = False
    agent_positions = []
    attentions = []
    for i in range(MAX_EPISODE_LENGTH):
        print(i)
        env = agents.act(env)
        num_comms += env.num_agents ** 2
        attentions.append(agents.model.communication_block.encoder.saved_attention[0][0])
        agent_positions.append(np.array(env.agent_positions))
        if env.is_solved():
            solved = True
            break

    return TestResult(
        solved,
        i + 1,
        num_comms,
        agent_positions,
        attentions,
    )


def run_multiple_tests(tests):
    if PARALLEL:
        pool = mp.Pool(16)
        return pool.map(run_single_test, tests)
    else:
        return [run_single_test(test) for test in tests]


def run_tests_from_file(path, agents):
    print(f"Running tests from {path}")
    with open(path, "rb") as f:
        tests = pickle.load(f)
    tests = [Test(*test, agents) for test in tests]
    return run_multiple_tests(tests)


def prepare():
    agents = Agents(save_attention=True)
    agents.load_raw("scrimp/final/checkpoint.pkl")
    agents.model.eval()
    return agents


if __name__ == "__main__":
    agents = prepare()
    for filename in tqdm.tqdm(sorted(os.listdir("./DCC_test/test_set"))[:2:]):
        if filename.endswith("30.pth"):
            results = run_tests_from_file(f"./DCC_test/test_set/{filename}", agents)
            results: list[TestResult]
            with open(f"./scrimp_test/results/{filename}", "wb") as f:
                pickle.dump((
                    [result.success for result in results],
                    [result.steps for result in results],
                    [result.num_comm for result in results],
                    [result.history for result in results],
                    [result.attentions for result in results],
                ), f)
