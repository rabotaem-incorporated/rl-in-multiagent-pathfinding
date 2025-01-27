import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import sys

from model import ScrimpNet
from params import *


class NetParameters:
    NET_SIZE = 512
    NUM_CHANNEL = 8  # number of channels of observations -[FOV_SIZE x FOV_SIZEx NUM_CHANNEL]
    GOAL_REPR_SIZE = 12
    VECTOR_LEN = 7  # [dx, dy, d total,extrinsic rewards,intrinsic reward, min dist respect to buffer, action t-1]
    N_POSITION = 1024  # maximum number of unique ID
    D_MODEL = NET_SIZE  # for input and inner feature of attention
    D_HIDDEN = 1024  # for feed-forward network
    N_LAYERS = 1  # number of computation block
    N_HEAD = 8
    D_K = 32
    D_V = 32


class EnvParameters:
    N_AGENTS = 8  # number of agents used in training
    N_ACTIONS = 5
    EPISODE_LEN = 256  # maximum episode length in training
    FOV_SIZE = 3
    WORLD_SIZE = (10, 40)
    OBSTACLE_PROB = (0.0, 0.5)
    ACTION_COST = -0.3
    IDLE_COST = -0.3
    GOAL_REWARD = 0.0
    COLLISION_COST = -2
    BLOCKING_COST = -1


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

    def __init__(self, d_model, d_hidden, n_layers, n_head, d_k, d_v, n_position):
        """initialization"""
        super().__init__()
        self.encoder = Encoder(d_model=d_model, d_hidden=d_hidden,
                               n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v)

        self.position_enc = PositionalEncoding(d_model, n_position=n_position)

    def forward(self, encoder_input):
        """run encoder"""
        encoder_input = self.position_enc(encoder_input)

        enc_output, *_ = self.encoder(encoder_input)

        return enc_output



class SCRIMPNet(nn.Module):
    """network with transformer-based communication mechanism"""

    def __init__(self):
        """initialization"""
        super(SCRIMPNet, self).__init__()
        # observation encoder
        self.conv1 = nn.Conv2d(NetParameters.NUM_CHANNEL, NetParameters.NET_SIZE // 4, 2, 1, 1)
        self.conv1a = nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 4, 2, 1, 1)
        self.conv1b = nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 4, 2, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 2, 2, 1, 1)
        self.conv2a = nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE // 2, 2, 1, 1)
        self.conv2b = nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE // 2, 2, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE - NetParameters.GOAL_REPR_SIZE, 3,
                               1, 0)
        self.fully_connected_1 = nn.Linear(NetParameters.VECTOR_LEN, NetParameters.GOAL_REPR_SIZE)
        self.fully_connected_2 = nn.Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE)
        self.fully_connected_3 = nn.Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE)
        self.lstm_memory = nn.LSTMCell(input_size=NetParameters.NET_SIZE, hidden_size=NetParameters.NET_SIZE // 2)

        # output heads
        self.fully_connected_4 = nn.Linear(NetParameters.NET_SIZE * 2 + NetParameters.NET_SIZE // 2,
                                           NetParameters.NET_SIZE)
        self.policy_layer = nn.Linear(NetParameters.NET_SIZE, EnvParameters.N_ACTIONS)
        self.softmax_layer = nn.Softmax(dim=-1)
        self.value_layer_in = nn.Linear(NetParameters.NET_SIZE, 1)
        self.value_layer_ex = nn.Linear(NetParameters.NET_SIZE, 1)
        self.blocking_layer = nn.Linear(NetParameters.NET_SIZE, 1)
        self.message_layer = nn.Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE)

        # transformer based communication block
        self.communication_layer = TransformerEncoder(d_model=NetParameters.D_MODEL,
                                                      d_hidden=NetParameters.D_HIDDEN,
                                                      n_layers=NetParameters.N_LAYERS, n_head=NetParameters.N_HEAD,
                                                      d_k=NetParameters.D_K,
                                                      d_v=NetParameters.D_V, n_position=NetParameters.N_POSITION)

    def forward(self, obs, vector, input_state, message):
        """run neural network"""

        result = {}

        num_agent = obs.shape[1]
        obs = torch.reshape(obs, (-1,  NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE))
        vector = torch.reshape(vector, (-1, NetParameters.VECTOR_LEN))
        # matrix input
        x_1 = F.relu(self.conv1(obs))
        x_1 = F.relu(self.conv1a(x_1))
        x_1 = F.relu(self.conv1b(x_1))
        x_1 = self.pool1(x_1)
        x_1 = F.relu(self.conv2(x_1))
        x_1 = F.relu(self.conv2a(x_1))
        x_1 = F.relu(self.conv2b(x_1))
        x_1 = self.pool2(x_1)
        x_1 = self.conv3(x_1)
        x_1 = F.relu(x_1.view(x_1.size(0), -1))

        result["matrix_output"] = x_1

        # vector input
        x_2 = F.relu(self.fully_connected_1(vector))
        # Concatenation
        x_3 = torch.cat((x_1, x_2), -1)
        h1 = F.relu(self.fully_connected_2(x_3))
        h1 = self.fully_connected_3(h1)
        h2 = F.relu(h1 + x_3)

        result["lstm_input"] = h2

        # LSTM cell
        memories, memory_c = self.lstm_memory(h2, input_state)
        result["hidden"] = memories
        result["memory"] = memory_c
        output_state = (memories, memory_c)
        memories = torch.reshape(memories, (-1, num_agent, NetParameters.NET_SIZE // 2))
        h2 = torch.reshape(h2, (-1, num_agent, NetParameters.NET_SIZE))

        c1 = self.communication_layer(message)

        c1 = torch.cat([c1, memories, h2], -1)
        c1 = F.relu(self.fully_connected_4(c1))
        policy_layer = self.policy_layer(c1)
        policy = self.softmax_layer(policy_layer)
        policy_sig = torch.sigmoid(policy_layer)
        value_in = self.value_layer_in(c1)
        value_ex = self.value_layer_ex(c1)
        blocking = torch.sigmoid(self.blocking_layer(c1))
        message = self.message_layer(c1)

        result["policy"] = policy
        result["value_in"] = value_in
        result["value_ex"] = value_ex
        result["blocking"] = blocking
        result["policy_sig"] = policy_sig
        result["output_state"] = output_state
        result["policy_layer"] = policy_layer
        result["message"] = message
        return result


def main():
    model1 = SCRIMPNet()
    model2 = ScrimpNet()
    model1.load_state_dict(torch.load("scrimp-reference/final/net_checkpoint.pkl")['model'])

    NUM_AGENT = 8
    test_input_obs = torch.randn(NUM_AGENT, AP.num_observation_in_dim, AP.fov, AP.fov)
    test_input_vector = torch.randn(NUM_AGENT, AP.goal_in_dim)
    test_input_state = (torch.randn(NUM_AGENT, AP.memory_dim), torch.randn(NUM_AGENT, AP.memory_dim))
    test_input_message = torch.randn(1, NUM_AGENT, AP.message_feature_dim)

    model2.observation_encoder.convolutional[0].load_state_dict(model1.conv1.state_dict())
    model2.observation_encoder.convolutional[2].load_state_dict(model1.conv1a.state_dict())
    model2.observation_encoder.convolutional[4].load_state_dict(model1.conv1b.state_dict())
    model2.observation_encoder.convolutional[7].load_state_dict(model1.conv2.state_dict())
    model2.observation_encoder.convolutional[9].load_state_dict(model1.conv2a.state_dict())
    model2.observation_encoder.convolutional[11].load_state_dict(model1.conv2b.state_dict())
    model2.observation_encoder.convolutional[14].load_state_dict(model1.conv3.state_dict())

    # Goal embedding layer
    model2.observation_encoder.embed_goal.load_state_dict({
        'weight': model1.fully_connected_1.weight,
        'bias': model1.fully_connected_1.bias,
    })

    # MLP layers
    model2.observation_encoder.mlp[0].load_state_dict({
        'weight': model1.fully_connected_2.weight,
        'bias': model1.fully_connected_2.bias,
    })
    model2.observation_encoder.mlp[2].load_state_dict({
        'weight': model1.fully_connected_3.weight,
        'bias': model1.fully_connected_3.bias,
    })

    # LSTM parameters
    model2.observation_encoder.lstm.load_state_dict(model1.lstm_memory.state_dict())

    with torch.no_grad():
        out1 = model1(test_input_obs, test_input_vector, test_input_state, test_input_message)
        out2 = model2.observation_encoder.convolutional(test_input_obs)
        assert torch.allclose(out1["matrix_output"].reshape(out2.shape), out2, atol=1e-3)

        out2 = model2.observation_encoder(test_input_obs, test_input_vector, test_input_state)
        assert torch.allclose(out1["lstm_input"].reshape(out2[0].shape), out2[0], atol=1e-3)
        assert torch.allclose(out1["hidden"].reshape(out2[1].shape), out2[1], atol=1e-3)
        assert torch.allclose(out1["memory"].reshape(out2[2][1].shape), out2[2][1], atol=1e-3)

    # Communication Block parameters
    # Positional encoding
    # model2.communication_block.positional_encoding.pe.data.copy_(
    #     model1.communication_layer.position_enc.pos_table.data[:, :model2.communication_block.positional_encoding.pe.size(1)]
    # )

    # Encoder layers
    model2.communication_block.encoder.load_state_dict(model1.communication_layer.state_dict())

    with torch.no_grad():
        out1 = model1.communication_layer(test_input_message)
        out2 = model2.communication_block(test_input_message[0])
        assert torch.allclose(out1.reshape(out2.shape), out2, atol=1e-3)

    # Output Heads parameters
    model2.output_heads.fc.load_state_dict({
        'weight': model1.fully_connected_4.weight,
        'bias': model1.fully_connected_4.bias,
    })
    model2.output_heads.policy_fc.load_state_dict(model1.policy_layer.state_dict())
    model2.output_heads.extrinsic_value_fc.load_state_dict(model1.value_layer_ex.state_dict())
    model2.output_heads.intrinsic_value_fc.load_state_dict(model1.value_layer_in.state_dict())
    model2.output_heads.blocking_fc.load_state_dict(model1.blocking_layer.state_dict())
    model2.output_heads.message_fc.load_state_dict(model1.message_layer.state_dict())

    with torch.no_grad():
        out1 = model1(test_input_obs, test_input_vector, test_input_state, test_input_message)
        out2 = model2(test_input_obs, test_input_vector, test_input_state, test_input_message[0])
        assert torch.allclose(out1["policy_layer"].reshape(out2.policy_logits.shape), out2.policy_logits, atol=1e-3)

    torch.save(model2.state_dict(), sys.argv[1])


if __name__ == "__main__":
    main()
