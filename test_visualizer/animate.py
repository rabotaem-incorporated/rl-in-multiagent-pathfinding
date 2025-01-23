from pogema import pogema_v0
from pogema import GridConfig
from pogema.svg_animation.animation_wrapper import AnimationMonitor
import numpy as np


def animate(
    mp: np.ndarray,
    agents_xy: np.ndarray,
    targets_xy: np.ndarray,
    steps: np.ndarray,
    save_path: str = "./test-visualizer/animation.svg",
    actions: np.ndarray = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]]),
) -> None:
    num_agents = agents_xy.shape[0]
    config = GridConfig(
        on_target='nothing',
        size=max(mp.shape[0], mp.shape[1]),
        map=mp.tolist(),
        num_agents=num_agents,
        agents_xy=agents_xy.tolist(),
        targets_xy=targets_xy.tolist(),
    )
    config.MOVES = actions.tolist()
    env = pogema_v0(grid_config=config)
    monitor = AnimationMonitor(env)
    monitor.reset()
    for step in steps:
        monitor.step(step)
    monitor.save_animation(save_path)

