import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
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
        goal = F.relu(self.embed_goal(goal))

        x = torch.cat([x, goal], dim=-1)
        assert x.shape == (B, AP.num_observation_hidden_channels)

        x = F.relu(self.mlp(x) + x)
        hidden, memory = self.lstm(x, memory)

        return x, hidden, (hidden, memory)


class ScaledDotProductAttention(nn.Module):
    """scaled dot-product attention"""

    def __init__(self, temperature):
        """initialization"""
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v):
        """ run multiple independent attention heads in parallel"""
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # attn = attn.masked_fill(mask == 0, -1e6) # if mask==0,the input value will =-1e6
        # then the attention score will around 0
        attn = F.softmax(attn, dim=-1)  # attention score
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """multi-head self attention module"""

    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()
        """initialization"""
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    def forward(self, q, k, v):
        """calculate multi-head attention"""
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # pass through the pre-attention projection
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # transpose for attention dot product
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # calculate attention
        q, attn = self.attention(q, k, v)
        # combine the last two dimensions to concatenate all the heads together
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid):
        """Initialization"""
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

    def forward(self, x):
        """run a ff layer"""
        x = self.w_2(F.relu(self.w_1(x)))
        return x


class GatingMechanism(nn.Module):
    """a GRU cell"""

    def __init__(self, d_model, bg=2):
        """Initialization"""
        super(GatingMechanism, self).__init__()
        self.Wr = nn.Linear(d_model, d_model)
        self.Ur = nn.Linear(d_model, d_model)
        self.Wz = nn.Linear(d_model, d_model)
        self.Uz = nn.Linear(d_model, d_model)
        self.Wg = nn.Linear(d_model, d_model)
        self.Ug = nn.Linear(d_model, d_model)
        self.bg = torch.nn.Parameter(torch.full([d_model], bg, dtype=torch.float32))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, y):  # x is residual, y is input
        """run a GRU in the place of residual connection"""
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g



class EncoderLayer(nn.Module):
    """compose with two different sub-layers"""

    def __init__(self, d_model, d_hidden, n_head, d_k, d_v):
        """define one computation block"""
        super(EncoderLayer, self).__init__()
        self.gate1 = GatingMechanism(d_model)
        self.gate2 = GatingMechanism(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_hidden)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, enc_input):
        """run a computation block"""
        enc_output = self.norm1(enc_input)
        enc_output, enc_slf_attn = self.slf_attn(
            enc_output, enc_output, enc_output)
        enc_output_1 = self.gate1(enc_input, enc_output)
        enc_output = self.pos_ffn(self.norm2(enc_output_1))
        enc_output = self.gate2(enc_output_1, enc_output)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    """a encoder model with self attention mechanism"""

    def __init__(self, d_model, d_hidden, n_layers, n_head, d_k, d_v):
        """create multiple computation blocks"""
        super().__init__()
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_hidden, n_head, d_k, d_v) for _ in range(n_layers)])

    def forward(self, enc_output, return_attns=False):
        """use self attention to merge messages"""
        enc_slf_attn_list = []
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class PositionalEncoding(nn.Module):
    """sinusoidal position embedding"""

    def __init__(self, d_hid, n_position=200):
        """create table"""
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """encode unique agent id """
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class TransformerEncoder(nn.Module):
    """a sequence to sequence model with attention mechanism"""

    def __init__(self, d_model, d_hidden, n_layers, n_head, d_k, d_v, n_position, save_attention=False):
        """initialization"""
        super().__init__()
        self.encoder = Encoder(d_model=d_model, d_hidden=d_hidden,
                               n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v)

        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.save_attention = save_attention
        self.saved_attention = []

    def forward(self, encoder_input):
        """run encoder"""
        encoder_input = self.position_enc(encoder_input)

        enc_output, attentions = self.encoder(encoder_input, return_attns=True)
        self.saved_attention = np.array([a.detach().cpu().numpy() for a in attentions])

        return enc_output


class CommunicationBlock(nn.Module):
    def __init__(self, save_attention=False):
        super().__init__()

        self.encoder = TransformerEncoder(
            d_model=AP.message_feature_dim,
            d_hidden=AP.transformer_ff_dim,
            n_layers=AP.transformer_num_layers,
            n_head=AP.transformer_num_heads,
            d_k=AP.transformer_kdim,
            d_v=AP.transformer_vdim,
            n_position=1024,
            save_attention=save_attention,
        )

    def forward(self, message: torch.Tensor):
        B = message.shape[0]
        assert message.shape == (B, AP.message_feature_dim)

        message = self.encoder(message.unsqueeze(0)).squeeze(0)
        assert message.shape == (B, AP.message_feature_dim)

        return message


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

        observation = torch.cat([message, memory, observation], dim=-1)
        observation = F.relu(self.fc(observation))

        policy_logits = self.policy_fc(observation)
        extrinsic_value = self.extrinsic_value_fc(observation)
        intrinsic_value = self.intrinsic_value_fc(observation)
        blocking = F.sigmoid(self.blocking_fc(observation))
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
    def __init__(self, save_attention=False):
        super().__init__()

        self.observation_encoder = ObservationEncoder()
        self.communication_block = CommunicationBlock(save_attention)
        self.output_heads = OutputHeads()

    def forward(self, obs: torch.Tensor, goal_info: torch.Tensor, state: ScrimpNetState, message: torch.Tensor) -> ScrimpNetOutputs:
        # print(obs, goal_info, message[:, :4], sep="\n\n", end="\n\n")
        obs, memory, state = self.observation_encoder(obs, goal_info, state)
        message = self.communication_block(message)
        policy, extrinsic_value, intrinsic_value, blocking, message = self.output_heads(obs, memory, message)

        # print(intrinsic_value, extrinsic_value, blocking, policy, message[:, :4], sep="\n\n", end="\n\n")
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
