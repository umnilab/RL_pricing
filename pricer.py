import numpy as np
import torch
import copy
from torch.optim import Adam
from model import ActorMLP, ProbActorMLP, CriticMLP, TwinCriticMLP, ActorCNN, ProbActorCNN, CriticCNN, TwinCriticCNN, SACCriticMLP, SACCriticCNN
from torch.distributions import MultivariateNormal

criterion = torch.nn.MSELoss()
MAX_GRAD_NORM_ACTOR = 0.01
MAX_GRAD_NORM_CRITIC = 0.1
GAMMA = 0.99 # discounted infinite horizon

# Actor Critic Pricer
class Pricer:
    def __init__(self):
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        self.update_time = 0

    def update_value(self):
        pass

    def update_policy(self):
        pass

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def initialize(self, rng):
        self.actor.initialize(rng)
        self.critic.initialize(rng)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def soft_update(self, local_model, target_model, tau = 1e-3):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def random_action(self):
        action = torch.rand((1,self.n_action))
        return action

    def load_weights(self, output, epoch):

        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}actor_{}.pkl'.format(output, epoch))
        )

        self.critic.load_state_dict(
            torch.load('{}critic_{}.pkl'.format(output, epoch))
        )

        self.actor_target.load_state_dict(
            torch.load('{}actor_target_{}.pkl'.format(output, epoch))
        )

        self.critic_target.load_state_dict(
            torch.load('{}critic_target_{}.pkl'.format(output, epoch))
        )

        self.actor_optim.load_state_dict(
            torch.load('{}actor_optim_{}.pkl'.format(output, epoch))
        )

        self.critic_optim.load_state_dict(
            torch.load('{}critic_optim_{}.pkl'.format(output, epoch))
        )


    def save_model(self, output, epoch):
        torch.save(
            self.actor.state_dict(),
            '{}actor_{}.pkl'.format(output, epoch)
        )
        torch.save(
            self.critic.state_dict(),
            '{}critic_{}.pkl'.format(output, epoch)
        )
        torch.save(
            self.actor_target.state_dict(),
            '{}actor_target_{}.pkl'.format(output, epoch)
        )
        torch.save(
            self.critic_target.state_dict(),
            '{}critic_target_{}.pkl'.format(output, epoch)
        )
        torch.save(
            self.actor_optim.state_dict(),
            '{}actor_optim_{}.pkl'.format(output, epoch)
        )
        torch.save(
            self.critic_optim.state_dict(),
            '{}critic_optim_{}.pkl'.format(output, epoch)
        )

class TD3(Pricer):
    def next_udpate_step(self, update_time):
        if self.policy_delay  == 0: # return n/ln(n+1), given the update frequency = 10 and buffer_size = 100800 (small case),
            # for 10000 updates of the value networks, nearly all buffers (10*10000 steps) are the wanted data
            return min(10000, round(update_time / np.log(update_time + 1)))
        elif self.policy_delay == -1: # return n
            return min(10000, update_time)
        else: # constant gap
            return self.policy_delay


    def update_policy(self, batch_size, memory, iter, update_freq=1):
        # Sample batch
        state_batch, state2d_batch, action_batch, reward_batch, next_state_batch, next_state2d_batch, t_batch = memory.sample(batch_size)

        with torch.no_grad():
            # noise = (torch.randn_like(action_batch) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_state_batch, next_state2d_batch, t_batch + 1)# (self.actor_target(next_state_batch, next_state2d_batch, t_batch + 1) + noise).clamp(-self.max_action, self.max_action)
            # Prepare for the target q batch
            target_Q1, target_Q2 = self.critic_target(next_state_batch, next_state2d_batch, next_action, t_batch + 1)
            target_q_batch = reward_batch + GAMMA * torch.minimum(target_Q1, target_Q2) # (t_batch<10079)*next_q_values

        # Critic updates
        current_Q1, current_Q2 = self.critic(state_batch, state2d_batch, action_batch, t_batch)
        value_loss = 1/2 * (criterion(current_Q1, target_q_batch) + criterion(current_Q2, target_q_batch))
        value_loss = value_loss.mean()
        self.writer.add_scalar("TD3_loss/value_loss", value_loss, iter)
        self.critic.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM_CRITIC)
        self.critic_optim.step()
        # Q target update
        self.soft_update(self.critic, self.critic_target)
        # Delay policy updates
        if iter >= self.policy_freq:
            # print(current_Q1)
            # print(target_q_batch)
            # Actor update
            action = self.actor(state_batch, state2d_batch, t_batch)
            # print(action)
            policy_loss = self.critic(state_batch, state2d_batch, action, t_batch)
            policy_loss = - policy_loss[0] #- policy_loss[1])
            # print(policy_loss)
            policy_loss = policy_loss.mean()
            # print(policy_loss)
            self.writer.add_scalar("TD3_loss/policy_loss", policy_loss, iter)
            self.actor.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM_ACTOR)
            self.actor_optim.step()
            self.update_time += 1
            self.policy_freq = iter + self.next_udpate_step(self.update_time) * update_freq
            # policy target update, directly copy, meanwhile forget the top layer of the Q-network
            self.soft_update(self.actor, self.actor_target, tau = 1e-3)
            # print(self.update_time)
            if self.forget and self.update_time % 10 == 0:
                print('Forget!')
                # A few iteration for fine tuning the Q function
                self.critic.forget()
                self.soft_update(self.critic, self.critic_target, tau = 1)
                # self.soft_update(self.actor, self.actor_target, tau=1)
                # reinitialize the optimizer
                self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)
                # self.policy_freq += 1000
                memory.forget()
                # self.update_time = 0

    def select_action(self, state, state2d,  t):
        action = self.actor(state, state2d, t)
        return action.detach()

class TD3_MLP(TD3):
    def __init__(self, num_zone, max_waiting, max_duration, max_traveling, actor_lr = 0.0001, critic_lr = 0.001, writer = None, policy_delay = 30,
                 position_encode = 32, forget = False, max_len = 10080):
        
        self.n_action = num_zone
        self.actor = ActorMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 128, num_zone], n = position_encode, max_len=max_len)
        self.actor_target = ActorMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 128, num_zone], n = position_encode, max_len=max_len)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = TwinCriticMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 128, 1], n = position_encode, max_len=max_len)
        self.critic_target = TwinCriticMLP(num_zone, max_waiting, max_duration, max_traveling, [128,  128, 1], n = position_encode, max_len=max_len)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        self.writer = writer

        self.actor_target.eval()
        self.critic_target.eval()

        self.policy_noise = 0.01
        self.noise_clip = 0.05
        self.policy_freq = 3
        self.max_action = 1
        self.update_time = 0
        self.policy_delay = policy_delay
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr

        self.forget = forget



class TD3_CNN_deep(TD3):
    def __init__(self, num_zone, max_duration, max_traveling, total_channel, kernel_size, stride, row_size, col_size, pooling, actor_lr = 0.0001,
                 critic_lr = 0.001, writer = None, policy_delay = 30, position_encode = 32, forget = False, max_len = 10080):
        self.n_action = num_zone
        self.writer = writer
        self.actor = ActorCNN(row_size,\
            col_size, \
            num_zone * (max_traveling + max_duration), \
            num_zone, \
            channels = [total_channel, 128, 32], \
            kernel_size_conv=kernel_size,\
            stride_size_conv=stride,\
            kernel_size_pool=pooling,\
            stride_size_pool=pooling,\
            shapes = [128, 128, num_zone], n = position_encode, max_len = max_len)
        self.actor_target = ActorCNN(row_size,\
            col_size, \
            num_zone * (max_traveling + max_duration), \
            num_zone, \
            channels = [total_channel, 128, 32], \
            kernel_size_conv=kernel_size,\
            stride_size_conv=stride,\
            kernel_size_pool=pooling,\
            stride_size_pool=pooling,\
            shapes = [128, 128, num_zone], n = position_encode, max_len = max_len)
        self.actor_optim  = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = TwinCriticCNN(row_size,\
            col_size, \
            num_zone * (max_traveling + max_duration + 1), \
            num_zone,\
            channels = [total_channel, 128, 32], \
            kernel_size_conv=kernel_size,\
            stride_size_conv=stride,\
            kernel_size_pool=pooling,\
            stride_size_pool=pooling,\
            shapes = [128, 128, 1], n = position_encode, max_len = max_len)
        self.critic_target = TwinCriticCNN(row_size,\
            col_size, \
            num_zone * (max_traveling + max_duration + 1), \
            num_zone,\
            channels = [total_channel, 128, 32], \
            kernel_size_conv=kernel_size,\
            stride_size_conv=stride,\
            kernel_size_pool=pooling,\
            stride_size_pool=pooling,\
            shapes = [128, 128, 1], n = position_encode, max_len = max_len)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_target.eval()
        self.critic_target.eval()

        self.policy_noise = 0.01
        self.noise_clip = 0.05
        self.policy_freq = 1
        self.max_action = 1
        self.update_time = 0
        self.policy_delay = policy_delay
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr

        self.forget = forget

class PPO(Pricer):
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.n_action,), new_action_std * new_action_std)

    def update_policy(self, batch_size, memory, epoches, horizon_size = 500, update_freq=1):
        # train for 1 epoch
        old_log_probs = []
        old_values = []
        # print("Estimate rt")
        with torch.no_grad():
            for state_batch, state2d_batch, action_batch, reward_batch, t_batch in memory.iteration(
                    horizon_size):
                log_prob , _ = self.evaluate_actions(state_batch, state2d_batch, action_batch, t_batch)
                old_value = torch.sum(reward_batch.view(-1) * torch.tensor([(GAMMA ** i) for i in range(horizon_size)]).to(reward_batch.device))
                old_log_probs.append(log_prob.detach())
                old_values.append(old_value.detach())
        # Do a complete pass on the rollout buffer
        print("Updating policy")
        # continue_training = True
        for epoch in range(epoches):
            # print("Epoch" + str(epoch) + "/" + str(iter))
            total_loss = 0
            policy_loss = 0
            value_loss = 0
            for state_batch, state2d_batch, action_batch, t_batch, old_values_batch, old_log_probs_batch in memory.iteration2(horizon_size, batch_size, old_values, old_log_probs):
                # evaluate the policy
                # calculate Advantage, and odd of probability
                values = self.critic(state_batch, state2d_batch, t_batch)
                advantages = old_values_batch - values.detach()
                new_log_probs, new_dist_entropy = self.evaluate_actions(state_batch, state2d_batch, action_batch, t_batch)
                ratio = torch.exp(new_log_probs - old_log_probs_batch.detach())
                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.noise_clip, 1 + self.noise_clip)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                value_loss = criterion(values, old_values_batch.unsqueeze(1))

                # Entropy loss favor exploration
                entropy_loss = -0.01 * torch.mean(-new_log_probs)
                loss = policy_loss + 0.5 * value_loss + entropy_loss
                total_loss += loss.detach()
                # with torch.no_grad():
                #     log_ratio = new_log_probs - old_log_probs_batch
                    # approx_kl_div = torch.mean((torch.exp(log_ratio)-1) - log_ratio).cpu().numpy()

                # limit the KL divergence of the update, disable it as it does not occur in the vanilla PPO
                # if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                #     continue_training = False
                #     break
                self.actor.zero_grad()
                self.critic.zero_grad()

                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM_CRITIC)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM_ACTOR)
                self.critic_optim.step()
                self.actor_optim.step()

            self.writer.add_scalar("PPO_loss/policy_loss", policy_loss, self.update_time)
            self.writer.add_scalar("PPO_loss/value_loss", value_loss, self.update_time)

            # if not continue_training:
            #     break
            self.update_time += 1
            if self.forget and self.update_time % 10 == 0:
                print('Forget!')
                # A few iteration for fine tuning the Q function
                self.critic.forget()
                # self.soft_update(self.critic, self.critic_target, tau = 1)
                # self.soft_update(self.actor, self.actor_target, tau=1)
                # reinitialize the optimizer
                self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)
                # self.policy_freq += 1000
                # memory.forget()
                # self.update_time = 0

        # clear the memory
        memory.clear()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def cuda(self):
        self.actor.cuda()
        self.critic.cuda()

    def initialize(self, rng):
        self.actor.initialize(rng)
        self.critic.initialize(rng)

    def save_model(self, output, epoch):
        torch.save(
            self.actor.state_dict(),
            '{}actor_{}.pkl'.format(output, epoch)
        )
        torch.save(
            self.critic.state_dict(),
            '{}critic_{}.pkl'.format(output, epoch)
        )
        torch.save(
            self.actor_optim.state_dict(),
            '{}actor_optim_{}.pkl'.format(output, epoch)
        )
        torch.save(
            self.critic_optim.state_dict(),
            '{}critic_optim_{}.pkl'.format(output, epoch)
        )

    def load_weights(self, output, epoch):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}actor_{}.pkl'.format(output, epoch))
        )

        self.critic.load_state_dict(
            torch.load('{}critic_{}.pkl'.format(output, epoch))
        )

        self.actor_optim.load_state_dict(
            torch.load('{}actor_optim_{}.pkl'.format(output, epoch))
        )

        self.critic_optim.load_state_dict(
            torch.load('{}critic_optim_{}.pkl'.format(output, epoch))
        )

class PPO_MLP(PPO):
    def __init__(self, num_zone, max_waiting, max_duration, max_traveling, actor_lr = 0.0001, critic_lr = 0.001, writer = None, position_encode = 32,
                 forget = False, max_len = 10080):
        self.n_action = num_zone
        self.actor = ProbActorMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 128, num_zone], n = position_encode, max_len = max_len)
        # self.actor_target = ProbActorMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 32, 2*num_zone], n = position_encode)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = CriticMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 128, 1], n = position_encode, max_len = max_len)
        # self.critic_target = CriticMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 32, 1], n = position_encode)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_lr = critic_lr
        self.writer = writer
        # self.actor_target.eval()
        # self.critic_target.eval()
        self.noise_clip = 0.1 # This is for odds truncating
        self.max_action = 1
        self.target_kl = 0.005
        self.update_time = 0
        self.forget = forget

    def evaluate_actions(self, states, state2d, actions, ts):
        action_mean = self.actor(states, ts)
        action_var = self.action_var.expand_as(action_mean).to(action_mean.device)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    def select_action(self, state, state2d,  t):
        action_mean = self.actor(state, t)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).to(action_mean.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        return action.detach()

class PPO_CNN_deep(PPO):
    def __init__(self, num_zone, max_duration, max_traveling, total_channel, kernel_size, stride, row_size, col_size, pooling, actor_lr = 0.0001,
                 critic_lr = 0.001, writer = None, position_encode = 32, forget = False, max_len = 10080):
        self.n_action = num_zone
        self.writer = writer
        self.actor = ProbActorCNN(row_size,\
            col_size, \
            num_zone * (max_traveling + max_duration), \
            num_zone, \
            channels = [total_channel, 128, 32], \
            kernel_size_conv=kernel_size,\
            stride_size_conv=stride,\
            kernel_size_pool=pooling,\
            stride_size_pool=pooling,\
            shapes = [128, 128, num_zone], n = position_encode, max_len = max_len)
        # self.actor_target = ActorCNN(row_size,\
        #     col_size, \
        #     num_zone * (max_traveling + max_duration), \
        #     channels = [total_channel, 128, 32], \
        #     kernel_size_conv=kernel_size,\
        #     stride_size_conv=stride,\
        #     kernel_size_pool=pooling,\
        #     stride_size_pool=pooling,\
        #     shapes = [128, 32, 2*num_zone], n = position_encode)
        self.actor_optim  = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = CriticCNN(row_size,\
            col_size, \
            num_zone * (max_traveling + max_duration), \
            num_zone, \
            channels = [total_channel, 128, 32], \
            kernel_size_conv=kernel_size,\
            stride_size_conv=stride,\
            kernel_size_pool=pooling,\
            stride_size_pool=pooling,\
            shapes = [128, 128, 1], n = position_encode, max_len = max_len)
        # self.critic_target = TwinCriticCNN(row_size,\
        #     col_size, \
        #     num_zone * (max_traveling + max_duration + 1), \
        #     channels = [total_channel, 128, 32], \
        #     kernel_size_conv=kernel_size,\
        #     stride_size_conv=stride,\
        #     kernel_size_pool=pooling,\
        #     stride_size_pool=pooling,\
        #     shapes = [128, 32, 1], n = position_encode)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_lr = critic_lr

        # self.actor_target.eval()
        # self.critic_target.eval()

        self.noise_clip = 0.1
        self.max_action = 1
        self.update_time = 0
        self.target_kl = 0.005
        
        self.forget = forget

    def evaluate_actions(self, states, state2d, actions, ts):
        action_mean = self.actor(states, state2d, ts)
        action_var = self.action_var.expand_as(action_mean).to(action_mean.device)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    def select_action(self, state, state2d, t):
        action_mean = self.actor(state, state2d, t)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).to(action_mean.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        return action.detach()

class SAC(Pricer):
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.n_action,), new_action_std * new_action_std)

    def update_policy(self, batch_size, memory, iter, update_freq=1):
         # Sample batch
        state_batch, state2d_batch, action_batch, reward_batch, next_state_batch, next_state2d_batch, t_batch = memory.sample(batch_size)

        with torch.no_grad():
            # Prepare for the target q batch
            _, target_V = self.critic_target(next_state_batch, next_state2d_batch, action_batch, t_batch + 1)
            target_q_batch = reward_batch + GAMMA * target_V # (t_batch<10079)*next_q_values
        
        current_Q, _ = self.critic(state_batch, state2d_batch, action_batch, t_batch)
        q_loss = 1/2 * criterion(current_Q, target_q_batch)
        q_loss = q_loss.mean()
        self.writer.add_scalar("SAC_loss/q_loss", q_loss, iter)
        self.critic.zero_grad()
        q_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM_CRITIC)
        self.critic_optim.step()

        # v updates
        with torch.no_grad():
            action = self.select_action(state_batch, state2d_batch, t_batch)
            new_log_probs, _ = self.evaluate_actions(state_batch, state2d_batch, action, t_batch)
            # Prepare for the target q batch
            target_Q, _ = self.critic(state_batch, state2d_batch, action, t_batch)
            target_v_batch = target_Q - self.temperature * new_log_probs.unsqueeze_(1)
        
        _, current_V = self.critic(state_batch, state2d_batch, action, t_batch)
        value_loss = 1/2 * criterion(current_V, target_v_batch)
        value_loss = value_loss.mean()
        self.writer.add_scalar("SAC_loss/value_loss", value_loss, iter)
        self.critic.zero_grad()
        value_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM_CRITIC)
        self.critic_optim.step()
        
        # Policy update
        # policy_loss = self.critic(state_batch, state2d_batch, action, t_batch)
        policy_loss, _ = self.critic(state_batch, state2d_batch, action, t_batch)
        policy_loss = - policy_loss + (self.temperature * new_log_probs).unsqueeze_(1)
        policy_loss = policy_loss.mean()
        # print(policy_loss)
        self.writer.add_scalar("SAC_loss/policy_loss", policy_loss, iter)
        self.actor.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM_ACTOR)
        self.actor_optim.step()
        self.update_time += 1
        # policy target update, directly copy, meanwhile forget the top layer of the Q-network
        # self.soft_update(self.actor, self.actor_target, tau = 1e-3)
        # print(self.update_time)
        # Temperature update
        temperature_loss = new_log_probs + self.n_action
        self.temperature -= 0.0001 * temperature_loss.mean()
        self.temperature = max(self.temperature, 0)

        # V target update
        self.soft_update(self.critic, self.critic_target)

        if self.forget and self.update_time % 100 == 0:
            print('Forget!')
            # A few iteration for fine tuning the Q function
            self.critic.forget()
            self.soft_update(self.critic, self.critic_target, tau = 1)
            # self.soft_update(self.actor, self.actor_target, tau=1)
            # reinitialize the optimizer
            self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)
            # self.policy_freq += 1000
            memory.forget()
            # self.update_time = 0

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def initialize(self, rng):
        self.actor.initialize(rng)
        self.critic.initialize(rng)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def load_weights(self, output, epoch):

        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}actor_{}.pkl'.format(output, epoch))
        )

        self.critic.load_state_dict(
            torch.load('{}critic_{}.pkl'.format(output, epoch))
        )

        self.critic_target.load_state_dict(
            torch.load('{}critic_target_{}.pkl'.format(output, epoch))
        )

        self.actor_optim.load_state_dict(
            torch.load('{}actor_optim_{}.pkl'.format(output, epoch))
        )

        self.critic_optim.load_state_dict(
            torch.load('{}critic_optim_{}.pkl'.format(output, epoch))
        )


    def save_model(self, output, epoch):
        torch.save(
            self.actor.state_dict(),
            '{}actor_{}.pkl'.format(output, epoch)
        )
        torch.save(
            self.critic.state_dict(),
            '{}critic_{}.pkl'.format(output, epoch)
        )
        torch.save(
            self.critic_target.state_dict(),
            '{}critic_target_{}.pkl'.format(output, epoch)
        )
        torch.save(
            self.actor_optim.state_dict(),
            '{}actor_optim_{}.pkl'.format(output, epoch)
        )
        torch.save(
            self.critic_optim.state_dict(),
            '{}critic_optim_{}.pkl'.format(output, epoch)
        )
    
class SAC_MLP(SAC):
    def __init__(self, num_zone, max_waiting, max_duration, max_traveling, actor_lr = 0.0001, critic_lr = 0.001, writer = None, position_encode = 32,
                 forget = False, max_len = 10080):
        self.n_action = num_zone
        self.actor = ProbActorMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 128, num_zone], n = position_encode, max_len = max_len)
        # self.actor_target = ProbActorMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 32, 2*num_zone], n = position_encode)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = SACCriticMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 128, 1], n = position_encode, max_len = max_len)
        self.critic_target = SACCriticMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 128, 1], n = position_encode, max_len = max_len)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_lr = critic_lr
        self.max_action = 1
        self.temperature = 0.1
        self.writer = writer

        self.update_time = 0

        self.forget = forget

    def evaluate_actions(self, states, state2d, actions, ts):
        action_mean = self.actor(states, ts)
        action_var = self.action_var.expand_as(action_mean).to(action_mean.device)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    def select_action(self, state, state2d,  t):
        action_mean = self.actor(state, t)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).to(action_mean.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        return action.detach()

class SAC_CNN(SAC):
    def __init__(self, num_zone, max_duration, max_traveling, total_channel, kernel_size, stride, row_size, col_size, pooling, actor_lr = 0.0001,
                 critic_lr = 0.001, writer = None, position_encode = 32, forget = False, max_len = 10080):
        self.n_action = num_zone
        self.actor = ProbActorCNN(row_size,\
            col_size, \
            num_zone * (max_traveling + max_duration), \
            num_zone, \
            channels = [total_channel, 128, 32], \
            kernel_size_conv=kernel_size,\
            stride_size_conv=stride,\
            kernel_size_pool=pooling,\
            stride_size_pool=pooling,\
            shapes = [128, 128, num_zone], n = position_encode, max_len = max_len)
        # self.actor_target = ProbActorMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 32, 2*num_zone], n = position_encode)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)        
        self.critic = SACCriticCNN(row_size,\
            col_size, \
            num_zone * (max_traveling + max_duration + 1), \
            num_zone,\
            channels = [total_channel, 128, 32], \
            kernel_size_conv=kernel_size,\
            stride_size_conv=stride,\
            kernel_size_pool=pooling,\
            stride_size_pool=pooling,\
            shapes = [128, 128, 1], n = position_encode, max_len = max_len)
        
        self.critic_target = SACCriticCNN(row_size,\
            col_size, \
            num_zone * (max_traveling + max_duration + 1), \
            num_zone,\
            channels = [total_channel, 128, 32], \
            kernel_size_conv=kernel_size,\
            stride_size_conv=stride,\
            kernel_size_pool=pooling,\
            stride_size_pool=pooling,\
            shapes = [128, 128, 1], n = position_encode, max_len = max_len)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_lr = critic_lr
        self.max_action = 1
        self.temperature = 0.1
        self.writer = writer

        self.update_time = 0 

        self.forget = forget

    def evaluate_actions(self, states, state2d, actions, ts):
        action_mean = self.actor(states, state2d, ts)
        action_var = self.action_var.expand_as(action_mean).to(action_mean.device)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    def select_action(self, state, state2d, t):
        action_mean = self.actor(state, state2d, t)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).to(action_mean.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        return action.detach()

    

