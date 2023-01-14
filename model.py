import torch
import numpy as np
import math
from typing import List, cast
from torch.autograd import Variable

PADDING = 2

def print_summary(model):
    tot = 0
    for name, para in model.named_parameters():
        print('{}: {}'.format(name, para.shape))
        tot += torch.numel(para)
    print('Total: ' + str(tot))

"""TD3 MLP"""
class ActorMLP(torch.nn.Module):
    def __init__(
            self,
            size: int,
            max_waiting: int,
            max_duration: int,
            max_traveling: int,
            shapes: List[int],
            n=32,  # for positional encoding, 0 for using nn
            max_len=10080
    ) -> None:
        super(ActorMLP, self).__init__()
        buf = []
        shapes = [size * size * max_waiting + size * (max_traveling + max_duration) + size + 1 + n] + shapes
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            buf.append(torch.nn.Linear(num_ins, num_outs))
        self.linears = torch.nn.ModuleList(buf)
        # self.tanh = torch.nn.Tanh()
        self.max_len = max_len

        # Positional encoding
        self.n = n
        if n <= 0:
            self.positional_embedding = torch.nn.Embedding(max_len, shapes[1])
        else:
            pe = torch.zeros(max_len, n)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.arange(2, n + 1, 2) * np.pi / max_len
            pe[:, 0::2] = torch.sin(position * div_term) / (size * size)
            pe[:, 1::2] = torch.cos(position * div_term) / (size * size)
            self.register_buffer('pe', pe)

    def initialize(self, rng: torch.Generator) -> None:
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

        print_summary(self)

    def forward(self, x: torch.Tensor, x2d: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        t = t % self.max_len
        if self.n <= 0:
            x = torch.relu(self.linears[0](x) + self.positional_embedding(t.type(torch.long)).squeeze(1))
        else:
            x = torch.concat([x, Variable(self.pe[t[:, 0]], requires_grad=False)], 1)
            x = torch.relu(self.linears[0](x))
        for (l, linear) in enumerate(self.linears[1:-1]):
            #
            x = linear.forward(x)
            x = torch.relu(x)
        # print("----------------- A before tanh----------------")
        # print(x)
        # x = self.tanh(x)
        # scale = 2*np.log(2)
        # x = scale * torch.sigmoid(x) - np.log(2)
        # print("-----------------A----------------")
        # print(x)
        x = self.linears[-1](x)
        # print(x.size())
        return x

class TwinCriticMLP(torch.nn.Module):
    def __init__(
            self,
            size: int,
            max_waiting: int,
            max_duration: int,
            max_traveling: int,
            shapes: List[int],
            n=32,
            max_len=10080
    ) -> None:
        super(TwinCriticMLP, self).__init__()
        buf = []
        buf2 = []
        shapes = [size * size * max_waiting + size * (max_traveling + max_duration) + size + size + 1 + n] + shapes
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            buf.append(torch.nn.Linear(num_ins, num_outs))
            buf2.append(torch.nn.Linear(num_ins, num_outs))
        self.linears = torch.nn.ModuleList(buf)
        self.linears2 = torch.nn.ModuleList(buf2)
        self.max_len = max_len

        # Positional encoding
        self.n = n
        if n <= 0:
            self.positional_embedding = torch.nn.Embedding(max_len, shapes[1])
            self.positional_embedding2 = torch.nn.Embedding(max_len, shapes[1])
        else:
            pe = torch.zeros(max_len, n)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.arange(2, n + 1, 2) * np.pi / max_len
            pe[:, 0::2] = torch.sin(position * div_term) / (size * size)
            pe[:, 1::2] = torch.cos(position * div_term) / (size * size)
            self.register_buffer('pe', pe)

    def initialize(self, rng: torch.Generator) -> None:
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()
        for linear in self.linears2:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()
        print_summary(self)

    def forward(self, x: torch.Tensor, x2d: torch.Tensor, a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = torch.concat([x, a], 1)
        t = t % self.max_len
        if self.n <= 0:
            q1 = torch.relu(self.linears[0](x) + self.positional_embedding(t.type(torch.long)).squeeze(1))
            q2 = torch.relu(self.linears2[0](x) + self.positional_embedding2(t.type(torch.long)).squeeze(1))
        else:
            x = torch.concat([x, Variable(self.pe[t[:, 0]], requires_grad=False)], 1)
            q1 = torch.relu(self.linears[0](x))
            q2 = torch.relu(self.linears2[0](x))
        for (l, linear) in enumerate(self.linears[1:-1]):
            #
            q1 = linear.forward(q1)
            q1 = torch.relu(q1)
        q1 = self.linears[-1](q1)
        for (l, linear) in enumerate(self.linears2[1:-1]):
            #
            q2 = linear.forward(q2)
            q2 = torch.relu(q2)
        q2 = self.linears2[-1](q2)
        return q1, q2

    def forget(self, num_layer=1):
        for linear in self.linears[-num_layer:]:
            # reinitialize the last [num_layer] of the network
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a)
            linear.bias.data.zero_()
        for linear in self.linears2[-num_layer:]:
            # reinitialize the last [num_layer] of the network
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a)
            linear.bias.data.zero_()

"""TD3 CNN"""
class ActorCNN(torch.nn.Module):
    def __init__(
            self,
            size: int, size2: int, size3: int, size4: int, channels: List[int], shapes: List[int],
            kernel_size_conv: int, stride_size_conv: int, kernel_size_pool: int,
            stride_size_pool: int,
            n=32,
            max_len=10080
    ) -> None:
        super(ActorCNN, self).__init__()
        buf_conv = []
        for (in_channels, out_channels) in zip(channels[:-1], channels[1:]):
            buf_conv.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                                            kernel_size=kernel_size_conv, stride=stride_size_conv, padding=PADDING))
            size = int((size - kernel_size_conv + 2 * PADDING) // stride_size_conv + 1)
            size2 = int((size2 - kernel_size_conv + 2 * PADDING) // stride_size_conv + 1)
            size = int((size - kernel_size_pool) // stride_size_pool + 1)
            size2 = int((size2 - kernel_size_pool) // stride_size_pool + 1)
        self.convs = torch.nn.ModuleList(buf_conv)
        self.pool = torch.nn.AvgPool2d(kernel_size=kernel_size_pool, stride=stride_size_pool)
        buf = []
        shapes = [size * size2 * channels[-1] + size3 + size4 + 1 + n] + shapes  # add action in the fully connected layer
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            buf.append(torch.nn.Linear(num_ins, num_outs))
        self.linears = torch.nn.ModuleList(buf)
        # self.tanh = torch.nn.Tanh()
        self.max_len = max_len

        # Positional encoding
        self.n = n
        if n <= 0:
            self.positional_embedding = torch.nn.Embedding(max_len, shapes[1])
        else:
            pe = torch.zeros(max_len, n)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.arange(2, n + 1, 2) * np.pi / max_len
            pe[:, 0::2] = torch.sin(position * div_term) / (size * size2)
            pe[:, 1::2] = torch.cos(position * div_term) / (size * size2)
            self.register_buffer('pe', pe)

    def initialize(self, rng: torch.Generator) -> None:
        for conv in self.convs:
            #
            (ch_outs, ch_ins, h, w) = conv.weight.data.size()
            num_ins = ch_ins * h * w
            num_outs = ch_outs * h * w
            a = np.sqrt(6 / (num_ins + num_outs))
            conv.weight.data.uniform_(-a, a, generator=rng)
            conv.bias.data.zero_()
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

        print_summary(self)

    def forward(self,  x1d: torch.tensor, x: torch.Tensor, t: torch.Tensor, /) -> torch.Tensor:
        for conv in self.convs:
            x = torch.relu(conv(x))
            x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.concat([x, x1d],1)
        t = t % self.max_len
        if self.n <= 0:
            x = torch.relu(self.linears[0](x) + self.positional_embedding(t.type(torch.long)).squeeze(1))
        else:
            x = torch.concat([x, Variable(self.pe[t[:, 0]], requires_grad=False)], 1)
            x = torch.relu(self.linears[0](x))
        for linear in self.linears[1:-1]:
            x = torch.relu(linear(x))
        x = self.linears[-1](x)

        # print(x)
        # scale = 2*np.log(2)
        # x = scale * torch.sigmoid(x) - np.log(2)
        return x

class TwinCriticCNN(torch.nn.Module):
    def __init__(
            self,
            size: int, size2: int, size3: int, size4: int, channels: List[int], shapes: List[int],
            kernel_size_conv: int, stride_size_conv: int, kernel_size_pool: int,
            stride_size_pool: int,
            n=32,
            max_len=10080
    ) -> None:
        super(TwinCriticCNN, self).__init__()

        buf_conv = []
        for (in_channels, out_channels) in zip(channels[:-1], channels[1:]):
            buf_conv.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                                            kernel_size=kernel_size_conv, stride=stride_size_conv, padding=PADDING))
            size = int((size - kernel_size_conv + 2 * PADDING) // stride_size_conv + 1)
            size2 = int((size2 - kernel_size_conv + 2 * PADDING) // stride_size_conv + 1)
            size = int((size - kernel_size_pool) // stride_size_pool + 1)
            size2 = int((size2 - kernel_size_pool) // stride_size_pool + 1)

        self.convs = torch.nn.ModuleList(buf_conv)
        self.pool = torch.nn.AvgPool2d(kernel_size=kernel_size_pool, stride=stride_size_pool)
        buf = []
        shapes = [size * size2 * channels[-1] + size3 + size4 + 1 + n] + shapes  # add action in the fully connected layer
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            #
            buf.append(torch.nn.Linear(num_ins, num_outs))
        self.linears = torch.nn.ModuleList(buf)

        buf_conv2 = []
        for (in_channels, out_channels) in zip(channels[:-1], channels[1:]):
            buf_conv2.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                                            kernel_size=kernel_size_conv, stride=stride_size_conv, padding=PADDING))
            size = int((size - kernel_size_conv + 2 * PADDING) // stride_size_conv + 1)
            size2 = int((size2 - kernel_size_conv + 2 * PADDING) // stride_size_conv + 1)
            size = int((size - kernel_size_pool) // stride_size_pool + 1)
            size2 = int((size2 - kernel_size_pool) // stride_size_pool + 1)

        self.convs2 = torch.nn.ModuleList(buf_conv2)
        self.pool2 = torch.nn.AvgPool2d(kernel_size=kernel_size_pool, stride=stride_size_pool)
        buf2 = []
        # shapes = [size * size2 * channels[-1] + size3 + n] + shapes  # add action in the fully connected layer
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            buf2.append(torch.nn.Linear(num_ins, num_outs))
        self.linears2 = torch.nn.ModuleList(buf2)
        self.max_len = max_len

        # Positional encoding
        self.n = n
        if n <= 0:
            self.positional_embedding = torch.nn.Embedding(max_len, shapes[1])
        else:
            pe = torch.zeros(max_len, n)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.arange(2, n + 1, 2) * np.pi / max_len
            pe[:, 0::2] = torch.sin(position * div_term) / size3
            pe[:, 1::2] = torch.cos(position * div_term) / size3
            self.register_buffer('pe', pe)

    def initialize(self, rng: torch.Generator) -> None:
        for conv in self.convs:
            #
            (ch_outs, ch_ins, h, w) = conv.weight.data.size()
            num_ins = ch_ins * h * w
            num_outs = ch_outs * h * w
            a = np.sqrt(6 / (num_ins + num_outs))
            conv.weight.data.uniform_(-a, a, generator=rng)
            conv.bias.data.zero_()
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

        for conv in self.convs2:
            #
            (ch_outs, ch_ins, h, w) = conv.weight.data.size()
            num_ins = ch_ins * h * w
            num_outs = ch_outs * h * w
            a = np.sqrt(6 / (num_ins + num_outs))
            conv.weight.data.uniform_(-a, a, generator=rng)
            conv.bias.data.zero_()
        for linear in self.linears2:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

        print_summary(self)

    def forward(self, x1d: torch.Tensor, x: torch.Tensor, a: torch.Tensor, t: torch.Tensor, /) -> torch.Tensor:
        t = t % self.max_len
        for l, conv in enumerate(self.convs):
            if l == 0:
                q1 = torch.relu(conv(x))
            else:
                q1 = torch.relu(conv(q1))
            q1 = self.pool(q1)
        q1 = q1.view(q1.size(0), -1)

        q1 = torch.concat([q1, x1d, a], 1)
        if self.n <= 0:
            q1 = torch.relu(self.linears[0](q1) + self.positional_embedding(t.type(torch.long)).squeeze(1))
        else:
            q1 = torch.concat([q1, Variable(self.pe[t[:, 0]], requires_grad=False)], 1)
            q1 = torch.relu(self.linears[0](q1))

        for linear in self.linears[1:-1]:
            q1 = torch.relu(linear(q1))
        q1 = self.linears[-1](q1)

        for l, conv in enumerate(self.convs2):
            if l == 0:
                q2 = torch.relu(conv(x))
            else:
                q2 = torch.relu(conv(q2))
            q2 = self.pool(q2)
        q2 = q2.view(q2.size(0), -1)

        q2 = torch.concat([q2, x1d, a], 1)
        if self.n <= 0:
            q2 = torch.relu(self.linears[0](q2) + self.positional_embedding(t.type(torch.long)).squeeze(1))
        else:
            q2 = torch.concat([q2, Variable(self.pe[t[:, 0]], requires_grad=False)], 1)
            q2 = torch.relu(self.linears[0](q2))

        for linear in self.linears2[1:-1]:
            q2 = torch.relu(linear(q2))
        q2 = self.linears2[-1](q2)

        return q1, q2

    def forget(self, num_layer=1):
        for linear in self.linears[-num_layer:]:
            # reinitialize the last [num_layer] of the network
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a)
            linear.bias.data.zero_()
        for linear in self.linears2[-num_layer:]:
            # reinitialize the last [num_layer] of the network
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a)
            linear.bias.data.zero_()


"""PPO MLP"""
class ProbActorMLP(torch.nn.Module):
    def __init__(
            self,
            size: int,
            max_waiting: int,
            max_duration: int,
            max_traveling: int,
            shapes: List[int],
            n=32,  # for positional encoding
            max_len=10080
    ) -> None:
        super(ProbActorMLP, self).__init__()
        buf = []
        shapes = [size * size * max_waiting + size * (max_traveling + max_duration) + size + 1 + n] + shapes
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            buf.append(torch.nn.Linear(num_ins, num_outs))
        self.linears = torch.nn.ModuleList(buf)
        self.max_len = max_len

        # Positional encoding
        # self.positional_embedding = torch.nn.Embedding(max_len, shapes[1])
        self.n = n
        if n <= 0:
            self.positional_embedding = torch.nn.Embedding(max_len, shapes[1])
        else:
            pe = torch.zeros(max_len, n)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.arange(2, n + 1, 2) * np.pi / max_len
            pe[:, 0::2] = torch.sin(position * div_term) / (size * size)
            pe[:, 1::2] = torch.cos(position * div_term) / (size * size)
            self.register_buffer('pe', pe)

    def initialize(self, rng: torch.Generator) -> None:
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

        print_summary(self)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        t = t % self.max_len
        if self.n <= 0:
            x = torch.relu(self.linears[0](x) + self.positional_embedding(t.type(torch.long)).squeeze(1))
        else:
            x = torch.concat([x, Variable(self.pe[t[:, 0]], requires_grad=False)], 1)
            x = torch.relu(self.linears[0](x))
        for (l, linear) in enumerate(self.linears[1:-1]):
            #
            x = linear.forward(x)
            x = torch.relu(x)
        x = self.linears[-1](x)
        return x


# value function for PPO
class CriticMLP(torch.nn.Module):
    def __init__(
            self,
            size: int,
            max_waiting: int,
            max_duration: int,
            max_traveling: int,
            shapes: List[int],
            n=32,
            max_len=10080
    ) -> None:
        super(CriticMLP, self).__init__()
        buf = []
        shapes = [size * size * max_waiting + size * (max_traveling + max_duration) + size + 1 + n] + shapes
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            buf.append(torch.nn.Linear(num_ins, num_outs))
        self.linears = torch.nn.ModuleList(buf)
        self.max_len = max_len

        # Positional encoding
        self.n = n
        if n <= 0:
            self.positional_embedding = torch.nn.Embedding(max_len, shapes[1])
        else:
            pe = torch.zeros(max_len, n)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.arange(2, n + 1, 2) * np.pi / max_len
            pe[:, 0::2] = torch.sin(position * div_term) / (size * size)
            pe[:, 1::2] = torch.cos(position * div_term) / (size * size)
            self.register_buffer('pe', pe)

    def initialize(self, rng: torch.Generator) -> None:
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

        print_summary(self)

    def forward(self, x: torch.Tensor, x2d: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        t = t % self.max_len
        if self.n <= 0:
            x = torch.relu(self.linears[0](x) + self.positional_embedding(t.type(torch.long)).squeeze(1))
        else:
            x = torch.concat([x, Variable(self.pe[t[:, 0]], requires_grad=False)], 1)
            x = torch.relu(self.linears[0](x))
        for (l, linear) in enumerate(self.linears[1:-1]):
            #
            x = linear.forward(x)
            x = torch.relu(x)
        x = self.linears[-1](x)
        return x

"""PPO CNN"""
class ProbActorCNN(torch.nn.Module):
    def __init__(
            self,
            size: int, size2: int, size3: int, size4: int, channels: List[int], shapes: List[int],
            kernel_size_conv: int, stride_size_conv: int, kernel_size_pool: int,
            stride_size_pool: int,
            n=32,
            max_len=10080
    ) -> None:
        super(ProbActorCNN, self).__init__()
        buf_conv = []
        for (in_channels, out_channels) in zip(channels[:-1], channels[1:]):
            buf_conv.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                                            kernel_size=kernel_size_conv, stride=stride_size_conv, padding=PADDING))
            size = int((size - kernel_size_conv + 2 * PADDING) // stride_size_conv + 1)
            size2 = int((size2 - kernel_size_conv + 2 * PADDING) // stride_size_conv + 1)
            size = int((size - kernel_size_pool) // stride_size_pool + 1)
            size2 = int((size2 - kernel_size_pool) // stride_size_pool + 1)
        self.convs = torch.nn.ModuleList(buf_conv)
        self.pool = torch.nn.AvgPool2d(kernel_size=kernel_size_pool, stride=stride_size_pool)
        buf = []
        shapes = [size * size2 * channels[-1] + size3 + size4 + 1 + n] + shapes  # add action in the fully connected layer
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            buf.append(torch.nn.Linear(num_ins, num_outs))
        self.linears = torch.nn.ModuleList(buf)
        self.max_len = max_len

        # Positional encoding
        self.n = n
        if n <= 0:
            self.positional_embedding = torch.nn.Embedding(max_len, shapes[1])
        else:
            pe = torch.zeros(max_len, n)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.arange(2, n + 1, 2) * np.pi / max_len
            pe[:, 0::2] = torch.sin(position * div_term) / size3
            pe[:, 1::2] = torch.cos(position * div_term) / size3
            self.register_buffer('pe', pe)

    def initialize(self, rng: torch.Generator) -> None:
        for conv in self.convs:
            #
            (ch_outs, ch_ins, h, w) = conv.weight.data.size()
            num_ins = ch_ins * h * w
            num_outs = ch_outs * h * w
            a = np.sqrt(6 / (num_ins + num_outs))
            conv.weight.data.uniform_(-a, a, generator=rng)
            conv.bias.data.zero_()
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

        print_summary(self)

    def forward(self,  x1d: torch.tensor, x: torch.Tensor, t: torch.Tensor, /) -> torch.Tensor:
        for conv in self.convs:
            x = torch.relu(conv(x))
            x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.concat([x, x1d], 1)
        t = t % self.max_len
        if self.n <= 0:
            x = torch.relu(self.linears[0](x) + self.positional_embedding(t.type(torch.long)).squeeze(1))
        else:
            x = torch.concat([x, Variable(self.pe[t[:, 0]], requires_grad=False)], 1)
            x = torch.relu(self.linears[0](x))
        for linear in self.linears[1:-1]:
            x = torch.relu(linear(x))
        x = self.linears[-1](x)
        # x = torch.relu(x)
        # scale = 2*np.log(2)
        # x = scale * torch.sigmoid(x) - np.log(2)
        return x

class CriticCNN(torch.nn.Module):
    def __init__(
            self,
            size: int, size2: int, size3: int, size4: int, channels: List[int], shapes: List[int],
            kernel_size_conv: int, stride_size_conv: int, kernel_size_pool: int,
            stride_size_pool: int,
            n=32,
            max_len=10080
    ) -> None:
        super(CriticCNN, self).__init__()
        buf_conv = []
        for (in_channels, out_channels) in zip(channels[:-1], channels[1:]):
            buf_conv.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                                            kernel_size=kernel_size_conv, stride=stride_size_conv, padding=PADDING))
            size = int((size - kernel_size_conv + 2 * PADDING) // stride_size_conv + 1)
            size2 = int((size2 - kernel_size_conv + 2 * PADDING) // stride_size_conv + 1)
            size = int((size - kernel_size_pool) // stride_size_pool + 1)
            size2 = int((size2 - kernel_size_pool) // stride_size_pool + 1)
        self.convs = torch.nn.ModuleList(buf_conv)
        self.pool = torch.nn.AvgPool2d(kernel_size=kernel_size_pool, stride=stride_size_pool)
        buf = []
        shapes = [size * size2 * channels[-1] + size3 + size4 + 1 + n] + shapes  # add action in the fully connected layer
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            buf.append(torch.nn.Linear(num_ins, num_outs))
        self.linears = torch.nn.ModuleList(buf)
        self.max_len = max_len

        # Positional encoding
        self.n = n
        if n <= 0:
            self.positional_embedding = torch.nn.Embedding(max_len, shapes[1])
        else:
            pe = torch.zeros(max_len, n)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.arange(2, n + 1, 2) * np.pi / max_len
            pe[:, 0::2] = torch.sin(position * div_term) / (size * size2)
            pe[:, 1::2] = torch.cos(position * div_term) / (size * size2)
            self.register_buffer('pe', pe)

    def initialize(self, rng: torch.Generator) -> None:
        for conv in self.convs:
            #
            (ch_outs, ch_ins, h, w) = conv.weight.data.size()
            num_ins = ch_ins * h * w
            num_outs = ch_outs * h * w
            a = np.sqrt(6 / (num_ins + num_outs))
            conv.weight.data.uniform_(-a, a, generator=rng)
            conv.bias.data.zero_()
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

        print_summary(self)

    def forward(self, x1d: torch.Tensor, x: torch.Tensor, t: torch.Tensor, /) -> torch.Tensor:
        for conv in self.convs:
            x = torch.relu(conv(x))
            x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.concat([x, x1d], 1)
        t = t % self.max_len
        if self.n <= 0:
            x = torch.relu(self.linears[0](x) + self.positional_embedding(t.type(torch.long)).squeeze(1))
        else:
            x = torch.concat([x, Variable(self.pe[t[:, 0]], requires_grad=False)], 1)
            x = torch.relu(self.linears[0](x))
        for linear in self.linears[1:-1]:
            x = torch.relu(linear(x))
        x = self.linears[-1](x)
        return x