import torch.nn as nn
import torch.nn.functional as F




class QMIXRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(QMIXRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class QMIXRNNAgent_SR(nn.Module):
    def __init__(self, input_shape, args):
        super(QMIXRNNAgent_SR, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        #b, a, e = inputs.size()
        # hh = hh.view(b, a, -1)
        #inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)
        #hh = hh.view(b, a, -1)

        return hh


class QMIXRNNAgent_W(nn.Module):
    def __init__(self, input_shape, args):
        super(QMIXRNNAgent_W, self).__init__()
        self.args = args

        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)


    def forward(self, inputs, shared_stata_embedding):
        #b, a, e = inputs.size()

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(shared_stata_embedding))
        else:
            q = self.fc2(shared_stata_embedding)
        #assert  a == 1
        return q


class FFAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(FFAgent, self).__init__()
        self.args = args

        # Easiest to reuse rnn_hidden_dim variable
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = F.relu(self.fc2(x))
        q = self.fc3(h)
        return q, h