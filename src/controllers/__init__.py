from .basic_controller import BasicMAC
from .cqmix_controller import CQMixMAC
from .EA_basic_controller import RL_BasicMAC, Gen_BasicMAC
REGISTRY = {}
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["cqmix_mac"] = CQMixMAC

REGISTRY["RL_basic_mac"] = RL_BasicMAC
REGISTRY["EA_basic_mac"] = Gen_BasicMAC
