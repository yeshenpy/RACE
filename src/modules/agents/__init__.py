REGISTRY = {}

from .mlp_agent import MLPAgent
from .rnn_agent import RNNAgent
from .comix_agent import CEMAgent, CEMRecurrentAgent
from .qmix_agent import QMIXRNNAgent, FFAgent
from .EA_qmix_agent import  QMIXRNNAgent_SR, QMIXRNNAgent_W
REGISTRY["mlp"] = MLPAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["cem"] = CEMAgent
REGISTRY["cemrnn"] = CEMRecurrentAgent
REGISTRY["qmixrnn"] = QMIXRNNAgent
REGISTRY["ff"] = FFAgent
REGISTRY["qmixrnn_SR"] = QMIXRNNAgent_SR
REGISTRY["qmixrnn_W"] = QMIXRNNAgent_W