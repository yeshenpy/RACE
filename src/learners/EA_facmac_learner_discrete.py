import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.facmac import FACMACDiscreteCritic,PeVFA_FACMACDiscreteCritic
#from components.action_selectors import multinomial_entropy
import torch as th
from torch.optim import RMSprop, Adam
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer,PeVFA_QMixer,V_Net
from modules.mixers.qmix_ablations import VDNState, QMixerNonmonotonic
from utils.rl_utils import build_td_lambda_targets
import random
import torch.nn as nn
import  time
import math
from torch.nn import functional as F
import torch
def get_negative_expectation(q_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        #
        Eq = F.softplus(-q_samples) + q_samples - log_2  # Note JSD will be shifted
        # Eq = F.softplus(q_samples) #+ q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        q_samples = torch.clamp(q_samples, -1e6, 9.5)

        # print("neg q samples ",q_samples.cpu().data.numpy())
        Eq = torch.exp(q_samples - 1.)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        assert 1 == 2

    if average:
        return Eq.mean()
    else:
        return Eq


def get_positive_expectation(p_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)  # Note JSD will be shifted
        # Ep =  - F.softplus(-p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples

    elif measure == 'RKL':

        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        assert 1 == 2

    if average:
        return Ep.mean()
    else:
        return Ep


def fenchel_dual_loss(l, m, measure=None):
    '''Computes the f-divergence distance between positive and negative joint distributions.
    Note that vectors should be sent as 1x1.
    Divergences supported are Jensen-Shannon `JSD`, `GAN` (equivalent to JSD),
    Squared Hellinger `H2`, Chi-squeared `X2`, `KL`, and reverse KL `RKL`.
    Args:
        l: Local feature map.
        m: Multiple globals feature map.
        measure: f-divergence measure.
    Returns:
        torch.Tensor: Loss.
    '''
    N, units = l.size()

    # Outer product, we want a N x N x n_local x n_multi tensor.
    u = torch.mm(m, l.t())

    # Since we have a big tensor with both positive and negative samples, we need to mask.
    mask = torch.eye(N).to(l.device)
    n_mask = 1 - mask
    # Compute the positive and negative score. Average the spatial locations.
    E_pos = get_positive_expectation(u, measure, average=False)
    E_neg = get_negative_expectation(u, measure, average=False)
    MI = (E_pos * mask).sum(1)  # - (E_neg * n_mask).sum(1)/(N-1)
    # Mask positive and negative terms for positive and negative parts of loss
    E_pos_term = (E_pos * mask).sum(1)
    E_neg_term = (E_neg * n_mask).sum(1) / (N - 1)
    loss = E_neg_term - E_pos_term
    return loss, MI


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, measure="JSD"):
        super(MINE, self).__init__()
        self.measure = measure
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.nonlinearity = F.leaky_relu
        self.l1 = nn.Linear(self.x_dim, 128)
        self.l2 =  nn.Linear(128, 128)
        self.l3 = nn.Linear(self.y_dim, 128)

    def forward(self, x, y, params=None):

        em_1 = self.nonlinearity(self.l1(x), inplace=True)
        em_1 = self.nonlinearity(self.l2(em_1), inplace=True)

        em_2 = self.nonlinearity(self.l3(y), inplace=True)
        two_agent_embedding = [em_1, em_2]
        loss, MI = fenchel_dual_loss(two_agent_embedding[0], two_agent_embedding[1], measure=self.measure)
        return loss, MI

import numpy as np
class EA_FACMACDiscreteLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.device = th.device('cuda' if args.use_cuda else 'cpu')

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)

        param = mac.parameters()
        if len(param) == 1 :
            self.agent_params = list(mac.parameters())
        else:
            self.agent_params = list(param[0])
            assert len( param[1]) == self.args.n_agents
            for p in param[1]:
                self.agent_params += list(p)


        self.critic = FACMACDiscreteCritic(scheme, args)


        self.MINE = MINE(self.args.rnn_hidden_dim,  int(np.prod(args.state_shape)))
        self.MINE_optimiser = Adam(params=self.MINE.parameters(), lr=args.critic_lr)

        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        if args.EA:
            self.PeVFA_critic = PeVFA_FACMACDiscreteCritic(scheme, args)
            self.PeVFA_params =list(self.PeVFA_critic.parameters())
            self.target_PeVFA_critic = copy.deepcopy(self.PeVFA_critic)

            self.PeVFA_mixer = PeVFA_QMixer(args)
            self.PeVFA_params += list(self.PeVFA_mixer.parameters())
            self.target_PeVFA_mixer = copy.deepcopy(self.PeVFA_mixer)
            self.PeVFA_optimiser = Adam(params=self.PeVFA_params, lr=args.critic_lr,eps=getattr(args, "optimizer_epsilon", 10E-8))
            self.V_Net = V_Net(args)
            self.V_Net_optimiser = Adam(params=self.V_Net.parameters(), lr=args.critic_lr,eps=getattr(args, "optimizer_epsilon", 10E-8))

        self.mixer = None
        if args.mixer is not None and self.args.n_agents > 1:  # if just 1 agent do not mix anything
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "vdn-s":
                self.mixer = VDNState(args)
            elif args.mixer == "qmix-nonmonotonic":
                self.mixer = QMixerNonmonotonic(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))

            self.critic_params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.last_target_update_episode = 0
        self.critic_training_steps = 0

    def train(self, batch: EpisodeBatch, all_teams,  t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        # actions = batch["actions"][:, :]
        actions = batch["actions_onehot"][:, :]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]
        temp_mask = mask

        start = time.time()
        if self.args.EA:
            # Train the critic batched
            index = random.sample(list(range(self.args.pop_size + 1)), 1)[0]
            selected_team = all_teams[index]

            target_mac_out = []
            selected_team.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_act_outs = selected_team.select_actions(batch, t_ep=t, t_env=t_env, test_mode=True)
                target_mac_out.append(target_act_outs)
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat over time

            param_list = []
            for i in range(self.args.n_agents):
                param = nn.utils.parameters_to_vector(list(selected_team.agent_W[i].parameters())).data.cpu().numpy()
                param_list.append(param)
            param_list = th.FloatTensor(param_list).to(self.device)

            q_taken, _ = self.PeVFA_critic(batch["obs"][:, :-1], actions[:, :-1], param_list)




            if self.mixer is not None:
                if self.args.mixer == "vdn":
                    assert 1 == 2
                else:
                    q_taken = self.PeVFA_mixer.forward(q_taken.view(batch.batch_size, -1, 1), batch["state"][:, :-1],param_list)


            target_vals, _ = self.target_PeVFA_critic(batch["obs"][:, :], target_mac_out.detach(),param_list)
            if self.mixer is not None:
                if self.args.mixer == "vdn":
                    assert 1 == 2
                else:
                    target_vals = self.target_PeVFA_mixer.forward(target_vals.view(batch.batch_size, -1, 1), batch["state"][:, :],param_list)

            if self.mixer is not None:
                q_taken = q_taken.view(batch.batch_size, -1, 1)
                target_vals = target_vals.view(batch.batch_size, -1, 1)
            else:
                q_taken = q_taken.view(batch.batch_size, -1, self.n_agents)
                target_vals = target_vals.view(batch.batch_size, -1, self.n_agents)

            targets = build_td_lambda_targets(batch["reward"], terminated, mask, target_vals, self.n_agents,
                                              self.args.gamma, self.args.td_lambda)
            mask =   temp_mask[:, :-1]
            td_error = (q_taken - targets.detach())
            mask = mask.expand_as(td_error)
            masked_td_error = td_error * mask
            ea_loss = (masked_td_error ** 2).sum() / mask.sum()
            self.PeVFA_optimiser.zero_grad()
            ea_loss.backward()
            critic_grad_norm = th.nn.utils.clip_grad_norm_(self.PeVFA_params, self.args.grad_norm_clip)
            self.PeVFA_optimiser.step()

        #print("1 ", time.time()-start)
        start = time.time()

        # Train the critic batched
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_act_outs = self.target_mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=True)
            target_mac_out.append(target_act_outs)
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat over time

        q_taken, _ = self.critic(batch["obs"][:, :-1], actions[:, :-1])
        if self.mixer is not None:
            if self.args.mixer == "vdn":
                q_taken = self.mixer(q_taken.view(-1, self.n_agents, 1), batch["state"][:, :-1])
            else:
                q_taken = self.mixer(q_taken.view(batch.batch_size, -1, 1), batch["state"][:, :-1])

        target_vals, _ = self.target_critic(batch["obs"][:, :], target_mac_out.detach())
        if self.mixer is not None:
            if self.args.mixer == "vdn":
                target_vals = self.target_mixer(target_vals.view(-1, self.n_agents, 1), batch["state"][:, :])
            else:
                target_vals = self.target_mixer(target_vals.view(batch.batch_size, -1, 1), batch["state"][:, :])

        if self.mixer is not None:
            q_taken = q_taken.view(batch.batch_size, -1, 1)
            target_vals = target_vals.view(batch.batch_size, -1, 1)
        else:
            q_taken = q_taken.view(batch.batch_size, -1, self.n_agents)
            target_vals = target_vals.view(batch.batch_size, -1, self.n_agents)

        targets_1 = build_td_lambda_targets(batch["reward"], terminated, mask, target_vals, self.n_agents,
                                          self.args.gamma, self.args.td_lambda)
        mask = temp_mask[:, :-1]
        td_error = (q_taken - targets_1.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        self.critic_training_steps += 1

        #print("2 ", time.time() - start)
        start = time.time()


        V_pre = self.V_Net(batch["state"][:, :-1]).view(batch.batch_size, -1, 1)
        V_error = (V_pre - torch.max(targets_1.detach(),targets.detach()))
        masked_v_error = V_error * mask
        V_loss = (masked_v_error ** 2).sum() / mask.sum()

        self.V_Net_optimiser.zero_grad()
        V_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.V_Net.parameters(), self.args.grad_norm_clip)
        self.V_Net_optimiser.step()

        # Train the actor
        # Use gumbel softmax to reparameterize the stochastic policies as deterministic functions of independent
        # noise to compute the policy gradient (one hot action input to the critic)



        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            act_outs = self.mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=False, explore=False)
            mac_out.append(act_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        chosen_action_qvals, _ = self.critic(batch["obs"][:, :-1], mac_out)

        if self.mixer is not None:
            if self.args.mixer == "vdn":
                chosen_action_qvals = self.mixer(chosen_action_qvals.view(-1, self.n_agents, 1),
                                                 batch["state"][:, :-1])
                chosen_action_qvals = chosen_action_qvals.view(batch.batch_size, -1, 1)
            else:
                chosen_action_qvals = self.mixer(chosen_action_qvals.view(batch.batch_size, -1, 1),
                                                 batch["state"][:, :-1])

        # Compute the actor loss
        pg_loss = - (chosen_action_qvals * mask).sum() / mask.sum()

        #print("3 ", time.time() - start)
        start = time.time()
        MINE_loss = 0
        if self.args.EA:

            V_weight = V_pre.detach()
            V_weight = V_weight.reshape([-1])
            
            V_weight = (V_weight - V_weight.min())/(V_weight.max()-V_weight.min())
            V_weight = V_weight.reshape([batch.batch_size, -1])
            mac_out = []
            index = random.sample(list(range(self.args.pop_size + 1)), 1)[0]
            selected_team = all_teams[index]
            selected_team.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                act_outs = selected_team.select_actions(batch, t_ep=t, t_env=t_env, test_mode=False, explore=False)

                Z = selected_team.get_hidden_state()

                repeat_state = torch.repeat_interleave(torch.unsqueeze(batch["state"][:, t],1), self.args.n_agents ,1)

                repeat_state = repeat_state.reshape([batch.batch_size* self.args.n_agents,-1])

                reshape_Z = Z.reshape([batch.batch_size* self.args.n_agents,-1])

                #print(V_weight[:,t])
                
                
                
                #weight = torch.clamp((V_weight[:,t]- V_weight[:,t].min())/(V_weight[:,t].max() - V_weight[:,t].min()), 1e-8, 1.0 )
                weight = V_weight[:,t].reshape([-1])
                #print(weight.shape, reshape_Z.shape, repeat_state.shape)
                mi_loss, _ = self.MINE.forward(reshape_Z,repeat_state)
                mi_loss = mi_loss.reshape([batch.batch_size, self.args.n_agents])
                mi_loss = mi_loss.mean(1)
                MINE_loss += (weight * mi_loss).mean()
                #MINE_loss += ( mi_loss).mean()

                mac_out.append(act_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            param_list = []
            for i in range(self.args.n_agents):
                param = nn.utils.parameters_to_vector(list(selected_team.agent_W[i].parameters())).data.cpu().numpy()
                param_list.append(param)
            param_list = th.FloatTensor(param_list).to(self.device)
            chosen_action_qvals, _ = self.PeVFA_critic(batch["obs"][:, :-1], mac_out, param_list)

            if self.mixer is not None:
                if self.args.mixer == "vdn":
                    assert 1==2
                else:
                    chosen_action_qvals = self.PeVFA_mixer.forward(chosen_action_qvals.view(batch.batch_size, -1, 1),
                                                     batch["state"][:, :-1],param_list)

            # Compute the actor loss
            ea_pg_loss = - self.args.EA_alpha* (chosen_action_qvals * mask).sum() / mask.sum()  + self.args.state_alpha * MINE_loss
        else :
            ea_pg_loss = 0.0
        #ea_pg_loss = 0.0
        # total_loss = pg_loss + self.args.EA_alpha *ea_pg_loss
        
        total_loss =  self.args.Org_alpha* pg_loss + ea_pg_loss
        # Optimise agents
        self.MINE_optimiser.zero_grad()
        self.agent_optimiser.zero_grad()
        total_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.MINE.parameters(), self.args.grad_norm_clip)
        self.agent_optimiser.step()
        self.MINE_optimiser.step()

        #print("3 ", time.time() - start)
        start = time.time()

        if getattr(self.args, "target_update_mode", "hard") == "hard":
            if (self.critic_training_steps - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
                self._update_targets()
                self.last_target_update_episode = self.critic_training_steps
        elif getattr(self.args, "target_update_mode", "hard") in ["soft", "exponential_moving_average"]:
            self._update_targets_soft(tau=getattr(self.args, "target_update_tau", 0.001))
        else:
            raise Exception(
                "unknown target update mode: {}!".format(getattr(self.args, "target_update_mode", "hard")))

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets_soft(self, tau):

        if self.args.EA:
            for target_param, param in zip(self.target_PeVFA_critic.parameters(), self.PeVFA_critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

            for target_param, param in zip(self.target_PeVFA_mixer.parameters(), self.PeVFA_mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.args.verbose:
            self.logger.console_logger.info("Updated all target networks (soft update tau={})".format(tau))

    def _update_targets(self):

        if self.args.EA:
            self.target_PeVFA_mixer.load_state_dict(self.PeVFA_mixer.state_dict())
            self.target_PeVFA_critic.load_state_dict(self.PeVFA_critic.state_dict())
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated all target networks")

    def cuda(self, device="cuda:0"):
        self.device = device
        self.mac.cuda(device=device)
        self.target_mac.cuda(device=device)
        self.critic.cuda(device=device)
        self.target_critic.cuda(device=device)
        if self.mixer is not None:
            self.mixer.cuda(device=device)
            self.target_mixer.cuda(device=device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))