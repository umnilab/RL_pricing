import numpy as np
import torch
import copy
from torch.optim import Adam
from model import ActorMLP, ProbActorMLP, CriticMLP, TwinCriticMLP, ActorCNN, ProbActorCNN, CriticCNN, TwinCriticCNN
from torch.distributions import MultivariateNormal

criterion = torch.nn.MSELoss()

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

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}actor_mlp.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}critic_mlp.pkl'.format(output))
        )

    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}critic.pkl'.format(output)
        )
        torch.save(
            self.actor_optim.state_dict(),
            '{}actor_optim.pkl'.format(output)
        )
        torch.save(
            self.critic_optim.state_dict(),
            '{}critic_optim.pkl'.format(output)
        )

class TD3(Pricer):
    def update_policy(self, batch_size, memory, iter):
        # Sample batch
        state_batch, state2d_batch, action_batch, reward_batch, next_state_batch, next_state2d_batch, t_batch = memory.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action_batch) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state_batch, next_state2d_batch, t_batch + 1) + noise).clamp(-self.max_action, self.max_action)

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
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optim.step()
        self.soft_update(self.critic, self.critic_target)
        # Delay policy updates
        if iter % self.policy_freq == 0:
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
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
            self.actor_optim.step()
            self.update_time += 1
            self.policy_freq += round(self.update_time/np.log(self.update_time+1))

            # target update, use interpolation
            self.soft_update(self.actor, self.actor_target, tau = 1e-3 * max(round(5* (1 - np.log(5) + np.log(self.update_time))),1))

    def select_action(self, state, state2d,  t):
        action = self.actor(state, state2d, t)
        return action.detach()

class TD3_MLP(TD3):
    def __init__(self, num_zone, max_waiting, max_duration, max_traveling, actor_lr = 0.0001, critic_lr = 0.001, writer = None):
        self.n_action = num_zone
        self.actor = ActorMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 64, 32, num_zone])
        self.actor_target = ActorMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 64, 32, num_zone])
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = TwinCriticMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 64, 32, 1])
        self.critic_target = TwinCriticMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 64, 32, 1])
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        self.writer = writer

        self.actor_target.eval()
        self.critic_target.eval()

        self.policy_noise = 0.02
        self.noise_clip = 0.05
        self.policy_freq = 3
        self.max_action = 1
        self.update_time = 0



class TD3_CNN_deep(TD3):
    def __init__(self, num_zone, max_duration, max_traveling, total_channel, kernel_size, stride, row_size, col_size, pooling, actor_lr = 0.0001,
                 critic_lr = 0.001, writer = None):
        self.n_action = num_zone
        self.writer = writer
        self.actor = ActorCNN(row_size,\
            col_size, \
            num_zone * (max_traveling + max_duration), \
            channels = [total_channel, 32], \
            kernel_size_conv=kernel_size,\
            stride_size_conv=stride,\
            kernel_size_pool=pooling,\
            stride_size_pool=pooling,\
            shapes = [128, 64, 32, num_zone])
        self.actor_target = ActorCNN(row_size,\
            col_size, \
            num_zone * (max_traveling + max_duration), \
            channels = [total_channel, 32], \
            kernel_size_conv=kernel_size,\
            stride_size_conv=stride,\
            kernel_size_pool=pooling,\
            stride_size_pool=pooling,\
            shapes = [128, 64, 32, num_zone])
        self.actor_optim  = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = TwinCriticCNN(row_size,\
            col_size, \
            num_zone * (max_traveling + max_duration + 1), \
            channels = [total_channel, 32], \
            kernel_size_conv=kernel_size,\
            stride_size_conv=stride,\
            kernel_size_pool=pooling,\
            stride_size_pool=pooling,\
            shapes = [128, 64, 32, 1])
        self.critic_target = TwinCriticCNN(row_size,\
            col_size, \
            num_zone * (max_traveling + max_duration + 1), \
            channels = [total_channel, 32], \
            kernel_size_conv=kernel_size,\
            stride_size_conv=stride,\
            kernel_size_pool=pooling,\
            stride_size_pool=pooling,\
            shapes = [128, 64, 32, 1])
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_target.eval()
        self.critic_target.eval()

        self.policy_noise = 0.05
        self.noise_clip = 0.1
        self.policy_freq = 1
        self.max_action = 1
        self.update_time = 0

class PPO(Pricer):
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.n_action,), new_action_std * new_action_std)
    def update_policy(self, batch_size, memory, epoches, horizon_size = 500):
        # train for 1 epoch
        old_log_probs = []
        old_values = []
        # print("Estimate rt")
        with torch.no_grad():
            for state_batch, state2d_batch, action_batch, reward_batch, t_batch in memory.iteration(
                    horizon_size):
                log_prob , _ = self.evaluate_actions(state_batch, action_batch, t_batch)
                old_value = torch.sum(reward_batch.view(-1) * torch.tensor([(GAMMA ** i) for i in range(horizon_size)]).to(reward_batch.device))
                old_log_probs.append(log_prob.detach())
                old_values.append(old_value.detach())
        # Do a complete pass on the rollout buffer
        print("Updating policy")
        continue_training = True
        for epoch in range(epoches):
            # print("Epoch" + str(epoch) + "/" + str(iter))
            total_loss = 0
            for state_batch, state2d_batch, action_batch, t_batch, old_values_batch, old_log_probs_batch in memory.iteration2(horizon_size, batch_size, old_values, old_log_probs):
                # evaluate the policy
                # calculate Advantage, and odd of probability
                values = self.critic(state_batch, state2d_batch, t_batch)
                advantages = old_values_batch - values.detach()
                new_log_probs, new_dist_entropy = self.evaluate_actions(state_batch, action_batch, state2d_batch, t_batch)
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

                with torch.no_grad():
                    log_ratio = new_log_probs - old_log_probs_batch
                    approx_kl_div = torch.mean((torch.exp(log_ratio)-1) - log_ratio).cpu().numpy()

                # limit the KL divergence of the update
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.actor.zero_grad()
                self.critic.zero_grad()

                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
                self.critic_optim.step()
                self.actor_optim.step()
            self.writer.add_scalar("PPO_loss/policy_loss", policy_loss, self.update_time)
            self.writer.add_scalar("PPO_loss/value_loss", value_loss, self.update_time)

            if not continue_training:
                break
            self.update_time += 1

        # clear the memory
        memory.clear()

class PPO_MLP(PPO):
    def __init__(self, num_zone, max_waiting, max_duration, max_traveling, actor_lr = 0.0001, critic_lr = 0.001, writer = None):
        self.n_action = num_zone
        self.actor = ProbActorMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 64, 32, num_zone])
        # self.actor_target = ProbActorMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 32, 2*num_zone])
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = CriticMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 64, 32, 1])
        # self.critic_target = CriticMLP(num_zone, max_waiting, max_duration, max_traveling, [128, 32, 1])
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        self.writer = writer
        # self.actor_target.eval()
        # self.critic_target.eval()
        self.noise_clip = 0.1
        self.max_action = 1
        self.target_kl = 0.005
        self.update_time = 0

    def evaluate_actions(self, states, state2d, actions, ts):
        action_mean = self.actor(states, ts)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    def select_action(self, state, state2d,  t):
        action_mean = self.actor(state, t)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        return action.detach()

class PPO_CNN_deep(PPO):
    def __init__(self, num_zone, max_duration, max_traveling, total_channel, kernel_size, stride, row_size, col_size, pooling, actor_lr = 0.0001,
                 critic_lr = 0.001, writer = None):
        self.n_action = num_zone
        self.writer = writer
        self.actor = ProbActorCNN(row_size,\
            col_size, \
            num_zone * (max_traveling + max_duration), \
            channels = [total_channel, 128, 32], \
            kernel_size_conv=kernel_size,\
            stride_size_conv=stride,\
            kernel_size_pool=pooling,\
            stride_size_pool=pooling,\
            shapes = [128, 32, num_zone])
        # self.actor_target = ActorCNN(row_size,\
        #     col_size, \
        #     num_zone * (max_traveling + max_duration), \
        #     channels = [total_channel, 128, 32], \
        #     kernel_size_conv=kernel_size,\
        #     stride_size_conv=stride,\
        #     kernel_size_pool=pooling,\
        #     stride_size_pool=pooling,\
        #     shapes = [128, 32, 2*num_zone])
        self.actor_optim  = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = CriticCNN(row_size,\
            col_size, \
            num_zone * (max_traveling + max_duration), \
            channels = [total_channel, 128, 32], \
            kernel_size_conv=kernel_size,\
            stride_size_conv=stride,\
            kernel_size_pool=pooling,\
            stride_size_pool=pooling,\
            shapes = [128, 32, 1])
        # self.critic_target = TwinCriticCNN(row_size,\
        #     col_size, \
        #     num_zone * (max_traveling + max_duration + 1), \
        #     channels = [total_channel, 128, 32], \
        #     kernel_size_conv=kernel_size,\
        #     stride_size_conv=stride,\
        #     kernel_size_pool=pooling,\
        #     stride_size_pool=pooling,\
        #     shapes = [128, 32, 1])
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        # self.actor_target.eval()
        # self.critic_target.eval()

        self.noise_clip = 0.1
        self.max_action = 1
        self.update_time = 0
        self.target_kl = 0.005

    def evaluate_actions(self, states, state2d, actions, ts):
        action_mean = self.actor(states, state2d, ts)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    def select_action(self, state, state2d, t):
        action_mean = self.actor(state, state2d, t)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        return action.detach()
