import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class V_Net(nn.Module):
    def __init__(self, args):
        super(V_Net, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        self.q_embed_dim = getattr(self.args, "q_embed_dim", 1)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear( self.embed_dim, self.embed_dim),
                                nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))


    def forward(self, states):
        v = self.V(states)
        return v



class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        self.q_embed_dim = getattr(self.args, "q_embed_dim", 1)

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents * self.q_embed_dim)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        if getattr(self.args, "hypernet_layers", 1) > 1:
            assert self.args.hypernet_layers == 2, "Only 1 or 2 hypernet_layers is supported atm!"
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents * self.q_embed_dim))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))

        # Initialise the hyper networks with a fixed variance, if specified
        if self.args.hyper_initialization_nonzeros > 0:
            std = self.args.hyper_initialization_nonzeros ** -0.5
            self.hyper_w_1.weight.data.normal_(std=std)
            self.hyper_w_1.bias.data.normal_(std=std)
            self.hyper_w_final.weight.data.normal_(std=std)
            self.hyper_w_final.bias.data.normal_(std=std)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

        if self.args.gated:
            self.gate = nn.Parameter(th.ones(size=(1,)) * 0.5)

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents*self.q_embed_dim)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents*self.q_embed_dim, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Skip connections
        s = 0
        if self.args.skip_connections:
            s = agent_qs.sum(dim=2, keepdim=True)

        if self.args.gated:
            y = th.bmm(hidden, w_final) * self.gate + v + s
        else:
            # Compute final output
            y = th.bmm(hidden, w_final) + v + s

        # Reshape and return
        q_tot = y.view(bs, -1, 1)

        return q_tot

from collections import OrderedDict
class PeVFA_QMixer(nn.Module):
    def __init__(self, args):
        super(PeVFA_QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.pr = args.pr_dim
        self.ls = args.rnn_hidden_dim


        self.add_module('layer_p1', nn.Linear(self.ls + 1, self.pr))
        self.add_module('layer_p2', nn.Linear(self.pr, self.pr))
        self.add_module('layer_p3', nn.Linear(self.pr, self.pr))

        self.embed_dim = args.mixing_embed_dim
        self.q_embed_dim = getattr(self.args, "q_embed_dim", 1)

        self.hyper_w_1 = nn.Linear(self.state_dim + self.pr , self.embed_dim * self.n_agents * self.q_embed_dim)
        self.hyper_w_final = nn.Linear(self.state_dim + self.pr, self.embed_dim)

        if getattr(self.args, "hypernet_layers", 1) > 1:
            assert self.args.hypernet_layers == 2, "Only 1 or 2 hypernet_layers is supported atm!"
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim + self.pr , hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents * self.q_embed_dim))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim + self.pr, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))

        # Initialise the hyper networks with a fixed variance, if specified
        if self.args.hyper_initialization_nonzeros > 0:
            std = self.args.hyper_initialization_nonzeros ** -0.5
            self.hyper_w_1.weight.data.normal_(std=std)
            self.hyper_w_1.bias.data.normal_(std=std)
            self.hyper_w_final.weight.data.normal_(std=std)
            self.hyper_w_final.bias.data.normal_(std=std)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim + self.pr, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim + self.pr, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))
        self.params = OrderedDict(self.named_parameters())
        self.nonlinearity = F.leaky_relu
        if self.args.gated:
            self.gate = nn.Parameter(th.ones(size=(1,)) * 0.5)

    def forward(self, agent_qs, states,param):



        param = param.reshape([-1, self.ls + 1])

        output = F.linear(param, weight=self.params['layer_p1.weight'],
                          bias=self.params['layer_p1.bias'])
        output = self.nonlinearity(output)
        output = F.linear(output, weight=self.params['layer_p2.weight'],
                          bias=self.params['layer_p2.bias'])
        output = self.nonlinearity(output)
        output = F.linear(output, weight=self.params['layer_p3.weight'],
                          bias=self.params['layer_p3.bias'])
        out_p = output.reshape([-1, self.n_agents ,self.args.n_actions, self.pr])

        out_p = th.sum(th.mean(out_p, dim=2),dim=1)
        out_p = out_p.repeat(int(states.shape[0] * states.shape[1]), 1)

        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        state_pr = th.cat([states, out_p], -1)

        agent_qs = agent_qs.view(-1, 1, self.n_agents*self.q_embed_dim)
        # First layer
        w1 = th.abs(self.hyper_w_1(state_pr))
        b1 = self.hyper_b_1(state_pr)
        w1 = w1.view(-1, self.n_agents*self.q_embed_dim, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(state_pr))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(state_pr).view(-1, 1, 1)
        # Skip connections
        s = 0
        if self.args.skip_connections:
            s = agent_qs.sum(dim=2, keepdim=True)

        if self.args.gated:
            y = th.bmm(hidden, w_final) * self.gate + v + s
        else:
            # Compute final output
            y = th.bmm(hidden, w_final) + v + s

        # Reshape and return
        q_tot = y.view(bs, -1, 1)

        return q_tot
