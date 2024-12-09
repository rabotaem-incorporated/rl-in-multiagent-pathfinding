import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import typing

from params import *


ScrimpNetState = typing.Tuple[torch.Tensor, torch.Tensor]


class ObservationEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        N = AP.num_observation_hidden_channels

        self.convolutional = nn.Sequential(
            nn.Conv2d(AP.num_observation_in_dim, N // 4, kernel_size=AP.convolution_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(N // 4, N // 4, kernel_size=AP.convolution_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(N // 4, N // 4, kernel_size=AP.convolution_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(AP.convolution_size),
            nn.Conv2d(N // 4, N // 2, kernel_size=AP.convolution_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(N // 2, N // 2, kernel_size=AP.convolution_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(N // 2, N // 2, kernel_size=AP.convolution_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(AP.convolution_size),
            nn.Conv2d(N // 2, N - AP.goal_embedding_size, kernel_size=AP.fov, padding=0),
            nn.ReLU(),
        )

        self.embed_goal = nn.Linear(AP.goal_in_dim, AP.goal_embedding_size)

        self.mlp = nn.Sequential(
            nn.Linear(N, N),
            nn.ReLU(),
            nn.Linear(N, N),
        )

        self.lstm = nn.LSTMCell(input_size=N, hidden_size=AP.memory_dim)

    def forward(self, x: torch.Tensor, goal: torch.Tensor, memory: ScrimpNetState):
        B = x.shape[0]
        assert x.shape == (B, AP.num_observation_in_dim, AP.fov, AP.fov)
        assert goal.shape == (B, AP.goal_in_dim)
        assert memory[0].shape == (B, AP.memory_dim)
        assert memory[1].shape == (B, AP.memory_dim)

        x = self.convolutional(x).view(B, AP.num_observation_hidden_channels - AP.goal_embedding_size)
        goal = self.embed_goal(goal)

        x = torch.cat([x, goal], dim=-1)
        assert x.shape == (B, AP.num_observation_hidden_channels)

        x = F.relu(self.mlp(x) + x)
        hidden, memory = self.lstm(x, memory)

        return x, hidden, (hidden, memory)


# https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        # return self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(AP.message_feature_dim)
        self.self_attention = nn.MultiheadAttention(AP.message_feature_dim, AP.transformer_num_heads)
        self.gru1 = nn.GRUCell(AP.message_feature_dim, AP.message_feature_dim)

        self.layer_norm2 = nn.LayerNorm(AP.message_feature_dim)
        self.ff_linear1 = nn.Linear(AP.message_feature_dim, AP.transformer_ff_dim)
        self.ff_linear2 = nn.Linear(AP.transformer_ff_dim, AP.message_feature_dim)
        self.gru2 = nn.GRUCell(AP.message_feature_dim, AP.message_feature_dim)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attention(x, x, x)
        x = self.gru1(residual, x)

        residual = x
        x = self.layer_norm2(x)
        x = self.ff_linear2(F.relu(self.ff_linear1(x)))
        x = self.gru2(residual, x)

        return x


class CommunicationBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.positional_encoding = PositionalEncoding(AP.message_feature_dim)
        self.encoder_layers = nn.ModuleList([EncoderLayer() for _ in range(AP.transformer_num_layers)])

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        assert x.shape == (B, AP.message_feature_dim)

        x = self.positional_encoding(x)
        assert x.shape == (B, AP.message_feature_dim)

        for layer in self.encoder_layers:
            x = layer(x)
            assert x.shape == (B, AP.message_feature_dim)

        return x


class OutputHeads(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(sum((
            AP.num_observation_hidden_channels,
            AP.memory_dim,
            AP.message_feature_dim,
        )), AP.output_hidden_dim)

        self.policy_fc = nn.Linear(AP.output_hidden_dim, AP.num_actions)
        self.extrinsic_value_fc = nn.Linear(AP.output_hidden_dim, 1)
        self.intrinsic_value_fc = nn.Linear(AP.output_hidden_dim, 1)
        self.blocking_fc = nn.Linear(AP.output_hidden_dim, 1)
        self.message_fc = nn.Linear(AP.output_hidden_dim, AP.message_feature_dim)
    
    def forward(self, observation: torch.Tensor, memory: torch.Tensor, message: torch.Tensor):
        B = observation.shape[0]
        assert observation.shape == (B, AP.num_observation_hidden_channels)
        assert memory.shape == (B, AP.memory_dim)
        assert message.shape == (B, AP.message_feature_dim)

        observation = torch.cat([observation, memory, message], dim=-1)
        observation = F.relu(self.fc(observation))

        policy_logits = self.policy_fc(observation)
        extrinsic_value = self.extrinsic_value_fc(observation)
        intrinsic_value = self.intrinsic_value_fc(observation)
        blocking = self.blocking_fc(observation)
        message = self.message_fc(observation)

        return policy_logits, extrinsic_value, intrinsic_value, blocking, message


@dataclasses.dataclass
class ScrimpNetOutputs:
    policy_logits: torch.Tensor
    extrinsic_value: torch.Tensor
    intrinsic_value: torch.Tensor
    blocking: torch.Tensor
    messages: torch.Tensor
    state: ScrimpNetState


class ScrimpNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.observation_encoder = ObservationEncoder()
        self.communication_block = CommunicationBlock()
        self.output_heads = OutputHeads()

    def forward(self, obs: torch.Tensor, goal_info: torch.Tensor, state: ScrimpNetState, message: torch.Tensor) -> ScrimpNetOutputs:
        obs, memory, state = self.observation_encoder(obs, goal_info, state)
        message = self.communication_block(message)
        policy, extrinsic_value, intrinsic_value, blocking, message = self.output_heads(obs, memory, message)

        return ScrimpNetOutputs(
            policy_logits=policy,
            extrinsic_value=extrinsic_value,
            intrinsic_value=intrinsic_value,
            blocking=blocking,
            messages=message,
            state=state,
        )


if __name__ == "__main__":
    net = ScrimpNet()
    obs = torch.randn(16, 8, 3, 3)
    goal = torch.randn(16, 7)
    state = (
        torch.randn(16, AP.memory_dim),
        torch.randn(16, AP.memory_dim),
    )
    message = torch.randn(16, AP.message_feature_dim)

    outputs = net(obs, goal, state, message)
    print(outputs)
