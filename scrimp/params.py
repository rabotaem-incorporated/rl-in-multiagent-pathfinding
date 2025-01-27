import dataclasses
import torch


@dataclasses.dataclass
class ArchitectureParameters:
    fov: int = 3
    num_observation_in_dim: int = 8
    convolution_size: int = 2
    num_observation_hidden_channels: int = 512
    goal_in_dim: int = 7
    goal_embedding_size: int = 12
    memory_dim: int = 256

    message_feature_dim: int = 512
    transformer_ff_dim: int = 1024
    transformer_num_layers: int = 1
    transformer_num_heads: int = 8
    transformer_vdim: int = 32
    transformer_kdim: int = 32

    output_hidden_dim: int = 512

    num_actions: int = 5

    episodic_buffer_size: int = 80


@dataclasses.dataclass
class Hyperparameters:
    lr: float = 1e-5
    weight_decay: float = 0.0
    intrinsic_surrogate1: float = 0.2
    intrinsic_reward_threshold: int = 10 ** 6
    episodic_buffer_dist: float = 3.0
    intrinsic_reward_dist: float = 3.0
    intrinsic_reward_beta: float = 1.0
    intrinsic_reward: float = 1.0

    blocking_reward: float = -1.0
    obstacle_reward: float = -1.0
    collision_reward: float = -2.0
    action_reward: float = -0.3
    idle_reward: float = -0.3
    goal_idle_reward: float = 0.0

    blocking_path_inflation_threshold: int = 10
    priority_mu: float = 0.1


AP = ArchitectureParameters()
HP = Hyperparameters()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
