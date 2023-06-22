import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FACMACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FACMACCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.view(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape

class FACMACDiscreteCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FACMACDiscreteCritic, self).__init__()
        self.args = args
        self.n_actions = scheme["actions_onehot"]["vshape"][0]
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape

from collections import OrderedDict
class PeVFA_FACMACDiscreteCritic(nn.Module):
    def __init__(self, scheme, args):
        super(PeVFA_FACMACDiscreteCritic, self).__init__()
        self.args = args
        self.n_actions = scheme["actions_onehot"]["vshape"][0]
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        self.pr = args.pr_dim
        self.ls = args.rnn_hidden_dim
        self.add_module('layer_p1', nn.Linear(self.ls + 1, self.pr))
        self.add_module('layer_p2', nn.Linear(self.pr, self.pr))
        self.add_module('layer_p3', nn.Linear(self.pr, self.pr))

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape + self.pr, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)
        self.params = OrderedDict(self.named_parameters())
        self.nonlinearity = F.leaky_relu

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, param, hidden_state=None):


        param = param.reshape([-1, self.ls + 1])

        output = F.linear(param, weight=self.params['layer_p1.weight'],
                          bias=self.params['layer_p1.bias'])
        output = self.nonlinearity(output)
        output = F.linear(output, weight=self.params['layer_p2.weight'],
                          bias=self.params['layer_p2.bias'])
        output = self.nonlinearity(output)
        output = F.linear(output, weight=self.params['layer_p3.weight'],
                          bias=self.params['layer_p3.bias'])
        out_p = output.reshape([self.n_agents,self.args.n_actions, self.pr])

        out_p = th.mean(out_p, dim=1)
        #print("1",out_p.shape,  inputs.shape)
        out_p = out_p.repeat(int(inputs.shape[0])*int(inputs.shape[1]), 1)
        #print("2",out_p.shape)
        if actions is not None:
            inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        #print("3",inputs.shape)
        inputs = th.cat([out_p,inputs],-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape