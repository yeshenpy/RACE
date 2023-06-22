from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class RL_BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.agent_SR = None
        self.agent_W = []
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        if args.action_selector is not None:
            self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, explore=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, return_logits=(not test_mode))
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode, explore=explore)
        if getattr(self.args, "use_ent_reg", False):
            return chosen_actions, agent_outputs
        return chosen_actions
    def get_hidden_state(self):
        return self.hidden_states.view(self.batch_size,self.n_agents, -1)


    def forward(self, ep_batch, t, return_logits=True):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        self.hidden_states = self.agent_SR(agent_inputs, self.hidden_states)
        self.batch_size = ep_batch.batch_size
       # b, a, e = agent_inputs.size()
       # hh = self.hidden_states.view(b, a, -1)
        # hh = self.hidden_states.view(ep_batch.batch_size, self.n_agents, -1)
        #agent_inputs = agent_inputs.view(ep_batch.batch_size, self.n_agents, -1)
        agent_outs = self.agent_W[0](agent_inputs, self.hidden_states)
        #agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        agent_outs = agent_outs.view(ep_batch.batch_size*self.n_agents, -1)

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            if return_logits:
                return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):

        self.hidden_states = self.agent_SR.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        param_list = []
        for i in range(self.args.n_agents):
            param_list.append(self.agent_W[i].parameters())
        return self.agent_SR.parameters(), param_list

        #return self.agent.parameters()

    def named_parameters(self):

        name_param_list = []
        name_param_list.append( self.agent_SR.named_parameters())
        for W in self.agent_W:
            name_param_list.append(W.named_parameters())

        return  name_param_list#self.agent_SR.named_parameters()

    def load_state(self, other_mac):
        self.agent_SR.load_state_dict(other_mac.agent_SR.state_dict())
        for i in range(self.n_agents):
            self.agent_W[i].load_state_dict(other_mac.agent_W[i].state_dict())

    def load_state_from_state_dict(self, state_dict):
        self.agent_SR.load_state_dict(state_dict)
        for i in range(self.n_agents):
            self.agent_W[i].load_state_dict(state_dict)
        assert  1 == 2

    def cuda(self, device="cuda"):
        self.agent_SR.cuda(device=device)
        for i in range(self.n_agents):
            self.agent_W[i].cuda(device=device)

    def _build_agents(self, input_shape):
        self.agent_SR = agent_REGISTRY[self.args.agent + "_SR"](input_shape, self.args)
        W = agent_REGISTRY[self.args.agent + "_W"](input_shape, self.args)
        for i in range(self.n_agents):
            self.agent_W.append(W)

    def share(self):
        self.agent_SR.share_memory()
        for i in range(self.n_agents):
            self.agent_W[i].share_memory()

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        try:
            inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        except Exception as e:
            pass
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def save_models(self, path):
        th.save(self.agent_SR.state_dict(), "{}/agent_SR.th".format(path))
        for i in range(self.n_agents):
            th.save(self.agent_W[i].state_dict(), "{}/agent_W_{}.th".format(path,str(i)))
    def load_models(self, path):

        self.agent_SR.load_state_dict(th.load("{}/agent_SR.th".format(path), map_location=lambda storage, loc: storage))
        for i in range(self.n_agents):
            self.agent_W[i].load_state_dict(th.load("{}/agent_W_{}.th".format(path,str(i)), map_location=lambda storage, loc: storage))


class Gen_BasicMAC:
    def __init__(self, scheme, agent_SR, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.agent_SR = agent_SR
        self.agent_W = []
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        if args.action_selector is not None:
            self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, explore=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, return_logits=(not test_mode))
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode, explore=explore)
        if getattr(self.args, "use_ent_reg", False):
            return chosen_actions, agent_outputs
        return chosen_actions

    def get_hidden_state(self):
        return self.hidden_states.view(self.batch_size,self.n_agents, -1)

    def forward(self, ep_batch, t, return_logits=True):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        self.batch_size = ep_batch.batch_size
        self.hidden_states = self.agent_SR(agent_inputs, self.hidden_states)

        # print("???????????? , ",agent_inputs.size() , self.hidden_states.size() )
        # b, a = agent_inputs.size()

        if self.args.SAME:

            agent_outs = self.agent_W[0](agent_inputs, self.hidden_states)
        else :

            hh = self.hidden_states.view(ep_batch.batch_size,self.n_agents, -1)
            agent_inputs = agent_inputs.view(ep_batch.batch_size,self.n_agents, -1)
            #print("??????????", hh.size(),agent_inputs.size() )
            agent_outs = []
            for i in range(self.n_agents):
                agent_outs.append(self.agent_W[0](agent_inputs[:,i,:], hh[:,i,:]).unsqueeze(1))
            agent_outs = th.cat(agent_outs, 1)
        agent_outs = agent_outs.squeeze()
        #agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        agent_outs = agent_outs.view(ep_batch.batch_size * self.n_agents, -1)

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            if return_logits:
                return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):

        self.hidden_states = self.agent_SR.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        param_list = []
        for i in range(self.args.n_agents):
            param_list.append(self.agent_W[i].parameters())
        return self.agent_SR.parameters(), param_list

        #return self.agent.parameters()

    def named_parameters(self):

        name_param_list = []
        name_param_list.append( self.agent_SR.named_parameters())
        for W in self.agent_W:
            name_param_list.append(W.named_parameters())

        return  name_param_list#self.agent_SR.named_parameters()

    def load_state(self, other_mac):
        self.agent_SR.load_state_dict(other_mac.agent_SR.state_dict())
        for i in range(self.n_agents):
            self.agent_W[i].load_state_dict(other_mac.agent_W[i].state_dict())

    def load_state_from_state_dict(self, state_dict):
        self.agent_SR.load_state_dict(state_dict)
        for i in range(self.n_agents):
            self.agent_W[i].load_state_dict(state_dict)
        assert  1 == 2

    def cuda(self, device="cuda"):
        self.agent_SR.cuda(device=device)
        for i in range(self.n_agents):
            self.agent_W[i].cuda(device=device)

    def _build_agents(self, input_shape):
        if self.args.SAME:
            W = agent_REGISTRY[self.args.agent + "_W"](input_shape, self.args)
            for i in range(self.n_agents):
                self.agent_W.append(W)
        else :
            for i in range(self.n_agents):
                self.agent_W.append(agent_REGISTRY[self.args.agent + "_W"](input_shape, self.args))

    def share(self):
        self.agent_SR.share_memory()
        for i in range(self.n_agents):
            self.agent_W[i].share_memory()

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        try:
            inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        except Exception as e:
            pass
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def save_models(self, path):
        th.save(self.agent_SR.state_dict(), "{}/agent_SR.th".format(path))
        for i in range(self.n_agents):
            th.save(self.agent_W[i].state_dict(), "{}/agent_W_{}.th".format(path,str(i)))
    def load_models(self, path):

        self.agent_SR.load_state_dict(th.load("{}/agent_SR.th".format(path), map_location=lambda storage, loc: storage))
        for i in range(self.n_agents):
            self.agent_W[i].load_state_dict(th.load("{}/agent_W_{}.th".format(path,str(i)), map_location=lambda storage, loc: storage))
