import numpy as np
import torch
from collections import namedtuple, deque
from random import sample
from lapsolver import solve_dense
from pricer import TD3_MLP, TD3_CNN_deep, PPO_MLP, PPO_CNN_deep
import pickle
from scipy.optimize import linprog

MAXP = 3
MINP = 1/3

class Platform:
    def __init__(self, travel_distance, travel_time, num_zone, max_waiting = 5, max_step = 6, option = 'TD3_MLP', permutation = None, permutation2 = None,
        kernel_size = 3, stride = 1, pooling = 2, device = 'cpu', actor_lr=0.0001, critic_lr=0.001, update_freq = 1, writer = None, od_permutation = True,
        veh_num = 1, demand_mean = None, demand_std = None, wage_mean = None, wage_std = None, searching = 'Gaussian', policy_delay = 30, position_encode = 64,
                  auto_correlation = False, forget = False, T_train = 10080):
        self.num_zone = num_zone
        self.veh_num = veh_num
        if demand_mean is not None:
            self.demand_mean = demand_mean
        else:
            self.demand_mean = np.zeros((num_zone,num_zone))

        if demand_std is not None:
            self.demand_std = demand_std
        else:
            self.demand_std = np.ones((num_zone,num_zone))

        if wage_mean is not None:
            self.wage_mean = wage_mean
        else:
            self.wage_mean = 0

        if wage_std is not None:
            self.wage_std = wage_std
        else:
            self.wage_std = 1


        self.travel_distance = travel_distance
        self.travel_time = travel_time
        self.writer = writer
        self.update_freq = update_freq

        # num of channel = max_waiting + 1
        if option == 'PPO_CNN':
            if od_permutation:
                self.pricer = PPO_CNN_deep(num_zone, max(max_waiting//update_freq,1), max_step + 1, 2*max_waiting, \
                                           kernel_size, stride, max(permutation[0])+1, \
                                       max(permutation[1])+1, pooling = pooling, actor_lr = actor_lr, critic_lr = critic_lr,
                                       writer = writer, position_encode = position_encode, max_len = T_train)
            else:
                self.pricer = PPO_CNN_deep(num_zone, max(max_waiting//update_freq,1), max_step + 1, max_waiting, \
                                           kernel_size, stride, num_zone, num_zone, \
                                       pooling = pooling, actor_lr = actor_lr, critic_lr = critic_lr,
                                       writer = writer, position_encode = position_encode, max_len = T_train)
        elif option == 'PPO_MLP':
            self.pricer = PPO_MLP(num_zone, max_waiting, max(max_waiting // update_freq, 1), max_step + 1,
                                  actor_lr=actor_lr, critic_lr=critic_lr,
                                  writer=writer, position_encode = position_encode, max_len = T_train)
        elif option == 'TD3_CNN':
            if od_permutation:
                self.pricer = TD3_CNN_deep(num_zone, max(max_waiting//update_freq, 1), max_step + 1, 2*max_waiting, \
                                           kernel_size, stride, max(permutation[0])+1, \
                                       max(permutation[1])+1, pooling = pooling, actor_lr = actor_lr, critic_lr = critic_lr,
                                       writer = writer, policy_delay = policy_delay, position_encode = position_encode, forget = forget,\
                                        max_len = T_train)
            else:
                self.pricer = TD3_CNN_deep(num_zone, max(max_waiting//update_freq, 1), max_step + 1, max_waiting, \
                                           kernel_size, stride, num_zone, num_zone, \
                                       pooling = pooling, actor_lr = actor_lr, critic_lr = critic_lr,
                                       writer = writer, policy_delay = policy_delay, position_encode = position_encode, forget = forget,\
                                           max_len = T_train)
        else:
            self.pricer = TD3_MLP(num_zone, max_waiting, max(max_waiting//update_freq, 1), max_step + 1, actor_lr = actor_lr, critic_lr = critic_lr,
                                   writer = writer, policy_delay = policy_delay, position_encode = position_encode, forget = forget,\
                                  max_len = T_train)
        print(option)
        if option.startswith('TD3'):
            self.buffer = Memory(size = (10*T_train)//update_freq, device=device)
        else:
            self.buffer = Memory(size = T_train//update_freq * 2, device=device)

        self.prices = torch.zeros((1, self.num_zone))

        self.last_state = None
        self.last_state2d = None

        self.auto_correlation = auto_correlation

        # For TD3
        self.epsilon = 0.5
        self.alpha = 0.9
        self.epsilon_min = 0.1

        if self.auto_correlation:
            self.noise = torch.zeros((10, self.num_zone))

        # For PPO
        self.action_std = 0.5
        self.action_std_decay_rate = 0.01
        self.min_action_std = 0.1

        if option.startswith('PPO'):
            self.pricer.set_action_std(self.action_std)

        # For permutation
        self.permutation = permutation
        self.permutation2 = permutation2

        self.option = option
        self.device = device
        self.searching = searching
        
    def update_price(self, pass_count, veh_count, ongoing_veh_count, avg_profit, zone_profits, price_multipliers,  \
                     t, mode, od_permutation = True,\
                     policy_constr = False, last_pricing = None): # price for trips departure from certain place
        # normalize here
        pass_count, veh_count, ongoing_veh_count,  price_multipliers, avg_profit, zone_profits= \
            self.preprocess(pass_count, veh_count, ongoing_veh_count,  price_multipliers, avg_profit, zone_profits)
        if len(price_multipliers.shape) < 2:
            price_multipliers = price_multipliers[None, :]

        # print(np.mean(pass_count))
        # print(np.mean(veh_count))
        # print(np.mean(ongoing_veh_count))
        # store the memory
        if self.option.endswith('CNN'):
            state2d = torch.from_numpy(pass_count).type(torch.FloatTensor)
            state = torch.from_numpy(
                np.concatenate([veh_count, ongoing_veh_count.flatten(), price_multipliers.flatten(), np.array([avg_profit]), zone_profits.flatten()])).type(
                torch.FloatTensor)
            if od_permutation:
                tmp = torch.zeros((state2d.size(0) * 2, max(self.permutation[0]) + 1, max(self.permutation[1]) + 1))
                tmp[:state2d.size(0), self.permutation[0], self.permutation[1]] += state2d.view(state2d.size(0), -1)
                tmp[state2d.size(0):, self.permutation2[0], self.permutation2[1]] += state2d.view(state2d.size(0), -1)
                state2d = tmp
        else:
            state = torch.from_numpy(np.concatenate([pass_count.flatten(), veh_count, ongoing_veh_count.flatten(),
                                                     price_multipliers.flatten(), np.array([avg_profit]), zone_profits.flatten()])).type(
                torch.FloatTensor)
            state2d = torch.zeros(1)

        # print("--------------State for decision-----------------")
        # print(state.size())
        # print(torch.sum(state,[1,2]))
        # update the price
        if mode == 'train':
            # print(self.searching)
            if self.searching == 'Gaussian' and self.option.startswith("TD3"):
                # gaussian noise, does not work since it can be easily trapped to a plateau when the number of zones is large.
                self.prices = self.pricer.select_action(torch.unsqueeze(state, 0).type(torch.FloatTensor).to(self.device), \
                                                        torch.unsqueeze(state2d, 0).type(torch.FloatTensor).to(self.device),\
                                                        torch.unsqueeze(torch.tensor([t]), 0).to(self.device)).cpu()
                # self.prices = self.prices + torch.randn(self.prices.shape)*max(self.epsilon,self.epsilon_min)
                # self.prices = torch.abs(self.prices - MINP) + MINP
                # self.prices = MAXP - torch.abs(MAXP- self.prices)
                # self.prices = self.prices.clamp(min = MINP, max = MAXP)
                if self.auto_correlation:
                    self.noise[0:9, :] = self.noise[1:10, :].clone()
                    self.noise[-1, :] = torch.randn(self.prices.shape) * max(self.epsilon, self.epsilon_min)/ 3.1623
                    self.prices += torch.sum(self.noise, 0).squeeze(0)
                else:
                    self.prices += torch.randn(self.prices.shape) * max(self.epsilon, self.epsilon_min)
                # self.epsilon *= self.alpha
            # epsilon-greedy
            elif self.searching == 'Greedy' and self.option.startswith("TD3"):
                if np.random.random() < max(self.epsilon,self.epsilon_min):
                    self.prices = self.pricer.random_action()
                    # print("-----------Price random----------------")
                    # print(self.prices)
                else:
                    self.prices = self.pricer.select_action(torch.unsqueeze(state, 0).type(torch.FloatTensor).to(self.device), \
                                                            torch.unsqueeze(state2d, 0).type(torch.FloatTensor).to(self.device),\
                                                            torch.unsqueeze(torch.tensor([t]), 0).to(self.device)).cpu()
                    # print("-----------Price after----------------")
                    # print(self.prices)
                # self.epsilon *= self.alpha
                # print(self.prices)
            else:
                self.prices = self.pricer.select_action(
                    torch.unsqueeze(state, 0).type(torch.FloatTensor).to(self.device), \
                    torch.unsqueeze(state2d, 0).type(torch.FloatTensor).to(self.device), \
                    torch.unsqueeze(torch.tensor([t]), 0).to(self.device)).cpu()
        else:
            self.prices = self.pricer.select_action(torch.unsqueeze(state, 0).type(torch.FloatTensor).to(self.device), \
                                                    torch.unsqueeze(state2d, 0).type(torch.FloatTensor).to(self.device),\
                                                    torch.unsqueeze(torch.tensor([t]), 0).to(self.device)).cpu()
        if policy_constr:
            res = (MINP * 0.1 * self.update_freq)**(self.prices.squeeze(0).numpy().flatten().clip(min=-1, max=1)) + last_pricing
            return res.clip(min = MINP, max = MAXP)
        else:
            # print(self.prices)
            return (MAXP**(self.prices.squeeze(0).numpy().flatten().clip(min=-1, max=1)))

    def preprocess(self, pass_count, veh_count, ongoing_veh_count, price_multipliers, avg_price, zone_prices):
        pass_count = (pass_count - self.demand_mean[None, :, :]) / (self.demand_std + 1)[None, :, :]
        veh_count = veh_count / self.veh_num * self.num_zone
        ongoing_veh_count = ongoing_veh_count / self.veh_num * self.num_zone
        price_multipliers = price_multipliers / MAXP
        avg_price = (avg_price - self.wage_mean) / self.wage_std
        zone_prices = (zone_prices + 1e-4) / (np.sum(zone_prices + 1e-4))
        return pass_count, veh_count, ongoing_veh_count, price_multipliers, avg_price, zone_prices

    def add_memory(self, pass_count, veh_count, ongoing_veh_count, avg_profit, zone_profits, price_multipliers, reward, t, od_permutation = True):
        # normalize here
        pass_count, veh_count, ongoing_veh_count, price_multipliers, avg_profit, zone_profits = \
            self.preprocess(pass_count, veh_count, ongoing_veh_count, price_multipliers, avg_profit, zone_profits)
        if len(price_multipliers.shape) < 2:
            price_multipliers = price_multipliers[None, :]

        if self.option.endswith('CNN'):
            state2d = torch.from_numpy(pass_count).type(torch.FloatTensor)
            state = torch.from_numpy(
                np.concatenate(
                    [veh_count, ongoing_veh_count.flatten(), price_multipliers.flatten(), np.array([avg_profit]),
                     zone_profits.flatten()])).type(
                torch.FloatTensor)
            if od_permutation:
                tmp = torch.zeros((state2d.size(0) * 2, max(self.permutation[0]) + 1, max(self.permutation[1]) + 1))
                tmp[:state2d.size(0), self.permutation[0], self.permutation[1]] += state2d.view(state2d.size(0), -1)
                tmp[state2d.size(0):, self.permutation2[0], self.permutation2[1]] += state2d.view(state2d.size(0), -1)
                state2d = tmp
        else:
            state = torch.from_numpy(np.concatenate([pass_count.flatten(), veh_count, ongoing_veh_count.flatten(),
                                                     price_multipliers.flatten(), np.array([avg_profit]),
                                                     zone_profits.flatten()])).type(
                torch.FloatTensor)
            state2d = torch.zeros(1)

        # state *= self.num_zone # scale up by sqrt (number of OD pairs)
        if self.last_state is None:
            self.last_state = state
            self.last_state2d = state2d
        else:
            self.buffer.push(self.last_state, self.last_state2d, self.prices, state, state2d, reward, t)
            self.last_state = state
            self.last_state2d = state2d

    def dummy_price(self):
        self.prices = torch.zeros((1,self.num_zone)).type(torch.FloatTensor)
        return np.power(2, self.prices.squeeze(0).numpy()).flatten()

    def random_price(self):
        self.prices = torch.rand((1,self.num_zone))*2-1
        return np.power(2, self.prices.squeeze(0).numpy()).flatten()

    def equilibrium_price(self, pass_count, veh_num, pass_count_future, data_folder):
        # if 'grid_small_static' in data_folder:
        #     # knowning the average trip length is 2.5, the average relocation trip length is also 2.5, calculate the idea pass_rate per zone
        #     ideal_pass_rate = veh_num / 5 / 16
        #     tmp = sum(pass_count_future) # in the static case, no attention is needed for the current pass_count 
        #     tmp = np.sum(tmp, axis = 1)
        #     tmp = tmp/(len(pass_count_future)*ideal_pass_rate)
        #     # print(veh_num)
        #     self.prices = torch.from_numpy(tmp[None,:]).type(torch.FloatTensor)
        #     return self.prices.squeeze(0).numpy()
        #     # if policy_constr:
        #     #     res = (self.prices.squeeze(0).numpy()- \
        #     #            last_pricing).clip(min=-(MINP * 0.1 * self.update_freq), max=(MINP * 0.1 * self.update_freq))+last_pricing
        #     #     return res
        #     # else:
        #     #     return self.prices.squeeze(0).numpy()
        #     # self.prices = 1 * torch.ones((1, self.num_zone)).type(torch.FloatTensor)
        #     # return 1.5*self.prices.squeeze(0).numpy().flatten()
        # elif 'grid_small_dynamic' in data_folder:
        #     ideal_pass_rate = veh_num / 5 / 16
        #     tmp = sum(pass_count_future) # in the static case, no attention is needed for the current pass_count 
        #     tmp = np.sum(tmp, axis = 1)
        #     tmp = tmp/(len(pass_count_future)*ideal_pass_rate)
        #     # print(tmp)
        #     self.prices = torch.from_numpy(tmp[None,:]).type(torch.FloatTensor)
        #     return self.prices.squeeze(0).numpy()
        # else:
            # use linear program to solve the dynamic pricing strategy
            # avg_travel_time is unkown, estimated as the (trip length) * 2
        tmp = sum(pass_count_future)
        avg_travel_time = 2*(np.sum(self.travel_time*(pass_count + tmp))/np.sum(pass_count + tmp))
        # print(avg_travel_time)
        ideal_pass_rate = max(min((veh_num / avg_travel_time)*len(pass_count_future), np.sum(tmp)*3),
        np.sum(tmp)/3)  # this becomes pass rate in toal, not per zone
        # the obj is to meet the idle_pass_rate while minimizing the sum of demand gaps in each zone
        # x is a 2n vector, the first n is the demand/inverse price multiplier, the second n are demand/supply imbalance 
        A_ub = []
        b_ub = []
        
        pass_gap = - np.sum(pass_count, axis = 0) + np.sum(pass_count, axis = 1)  # departure - arrival
        for i in range(pass_count.shape[0]):
            tmp_i = tmp[:,i].copy() # arrival
            tmp_i[i] -= np.sum(tmp[i,:]) # arrival - departure
            tmp_j = np.zeros(pass_count.shape[0]) 
            tmp_j[i] = -1
            A_ub.append(np.concatenate((tmp_i, tmp_j)))
            b_ub.append(pass_gap[i])
            A_ub.append(np.concatenate((-tmp_i, tmp_j)))
            b_ub.append(-pass_gap[i])

        A_eq = [np.concatenate((np.sum(tmp, axis = 1), np.zeros(pass_count.shape[0])))]
        b_eq = [ideal_pass_rate]

        x_min = 1/3
        x_max = 3 
        c = np.concatenate((np.zeros(len(pass_count)), np.ones(len(pass_count))))
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = [(x_min, x_max)] * pass_count.shape[0] + [(0, None)] * pass_count.shape[0])

        if(res.x[0] < 1/3):
            print(np.sum(A_ub,0))
            print(np.sum(b_ub))
            print(c)
            # print(A_ub)
            # print(b_ub)
            print(A_eq)
            print(np.sum(A_eq))
            print(b_eq)

        x = 1/res.x[:pass_count.shape[0]]
        self.prices = torch.from_numpy(x[None, :]).type(torch.FloatTensor)
        return self.prices.squeeze(0).numpy() 


    def batch_matching(self, pass_count, veh_count):
        # return: reposition cost and matching result
        # first match all veh and pass in the same zone
        veh_schedule = []
        same_zone = np.minimum(pass_count,veh_count).astype(int)
        for i in range(self.num_zone):
            for j in range(same_zone[i]):
                veh_schedule.append((i,i))
        # then solve integer program
        ## get the cost matrix
        total_cost = 0
        r_count = (veh_count-same_zone)
        c_count = (pass_count-same_zone)
        r_sum = np.sum(r_count)
        c_sum = np.sum(c_count)
        if r_sum > 0 and c_sum > 0:
            rlist = []
            clist = []
            for i in range(self.num_zone):
                rlist += [i]*min(sum(c_count),r_count[i])
                clist += [i]*min(sum(r_count),c_count[i])
            costs = np.zeros((len(rlist), len(clist)))
            for i in range(len(rlist)):
                costs[i,:] = self.travel_time[rlist[i], clist]
            for j in range(len(clist)):
                costs[:, j] = self.travel_time[rlist, clist[j]]
            costs[costs>15] = np.nan
            # costs[rlist, clist] = travel_time[rlist, clist]
            # for i in range(len(rlist)):
            #     for j in range(len(clist)):
            #         if self.travel_time[rlist[i], clist[j]] > 15: # greater than 15 mins, unreachable
            #             costs[i,j] = np.nan
            #         else:
            #             costs[i,j] = self.travel_time[rlist[i], clist[j]]
            # print(costs.shape)
            if costs.shape[0] < costs.shape[1]:
                rids, cids = solve_dense(costs)
            else:
                cids, rids = solve_dense(costs.T)
            for r,c in zip(rids, cids):
                veh_schedule.append((rlist[r], clist[c]))
                total_cost += costs[r,c]
        # veh_schedule is a three element tuple
        return total_cost, veh_schedule

    # def train_pricing_value(self, epoches = 100, batch_size = 32):
    #     # coarsely learn the value function to reduce the variance in the beginning
    #     # idea based on the policy iteration
    #     # unused
    #     self.pricer.update_value(epoches, batch_size, self.buffer)
        
    def train_pricing_policy(self, iter, thresholds, batch_size = 32, update_frequency = 1):
        if thresholds <= len(self.buffer):
            self.pricer.update_policy(batch_size, self.buffer, iter, update_freq = update_frequency)


    def decay_searching_variance(self):
        if self.option.startswith('PPO'):
            self.action_std = max(self.action_std - self.action_std_decay_rate, self.min_action_std)
            self.pricer.set_action_std(self.action_std)
        elif self.option.startswith('TD3'):
            self.epsilon = max(self.epsilon * self.alpha, self.epsilon_min)



Transition = namedtuple('Transition',
                        ('state', 'state2d', 'action', 'next_state', 'next_state2d', 'reward', 't'))

class Memory:
    def __init__(self, size, device='cpu'):
        self.memory = deque([], maxlen = size)
        self.size = size
        self.device = device
        print(f"Memory size {size}.")

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transition_batch = sample(list(self.memory), batch_size)
        state_batch = torch.stack([t[0] for t in transition_batch]).type(torch.FloatTensor)
        state2d_batch = torch.stack([t[1] for t in transition_batch]).type(torch.FloatTensor)
        action_batch = torch.concat([t[2] for t in transition_batch])
        reward_batch = torch.tensor([torch.tensor(t[5]) for t in transition_batch]).unsqueeze(1).type(torch.FloatTensor)
        next_state_batch = torch.stack([t[3] for t in transition_batch]).type(torch.FloatTensor)
        next_state2d_batch = torch.stack([t[4] for t in transition_batch]).type(torch.FloatTensor)
        t_batch = torch.tensor([torch.tensor(t[6]) for t in transition_batch]).unsqueeze(1)

        # if normalized: # only works for a small network, the problem is quite obvious,
        # looking for the different output with the same input...
        #     # think of transfer learning
        #     # normalize state and next state via their norm, a very rough way to perform batch normalization
        #     mu_state = torch.norm(state_batch, dim = 0) + 1e-4
        #     mu_next_state = torch.norm(next_state_batch, dim=0) + 1e-4
        #
        #     state_batch /= mu_state
        #     next_state_batch /= mu_next_state
        return state_batch.to(self.device), state2d_batch.to(self.device), action_batch.to(self.device), \
               reward_batch.to(self.device), next_state_batch.to(self.device), next_state2d_batch.to(self.device), t_batch.to(self.device)

    def iteration(self, horizon_size):
        perm = list(range(len(self.memory))) # list(np.random.permutation(len(self.memory)).astype(int))
        for i in range(len(perm)-horizon_size):
            transition_batch = [list(self.memory)[j] for j in perm[i:(i+horizon_size)]]
            state_batch = torch.stack([t[0] for t in transition_batch[:1]]).type(torch.FloatTensor)
            state2d_batch = torch.stack([t[1] for t in transition_batch[:1]]).type(torch.FloatTensor)
            action_batch = torch.concat([t[2] for t in transition_batch[:1]])
            reward_batch = torch.tensor([torch.tensor(t[5]) for t in transition_batch]).unsqueeze(1).type(
                torch.FloatTensor)
            # next_state_batch = torch.stack([t[3] for t in transition_batch]).type(torch.FloatTensor)
            # next_state2d_batch = torch.stack([t[4] for t in transition_batch]).type(torch.FloatTensor)
            t_batch = torch.tensor([torch.tensor(t[6]) for t in transition_batch[:1]]).unsqueeze(1)
            yield state_batch.to(self.device), state2d_batch.to(self.device), action_batch.to(self.device), \
               reward_batch.to(self.device), t_batch.to(self.device)

    def iteration2(self,horizon_size, batch_size, old_values, old_log_probs):
        perm = list(np.random.permutation(len(self.memory)-horizon_size).astype(int))
        for i in range(len(perm) // batch_size):
            transition_batch = [list(self.memory)[j] for j in perm[(i * batch_size):((i + 1) * batch_size)]]
            state_batch = torch.stack([t[0] for t in transition_batch]).type(torch.FloatTensor)
            state2d_batch = torch.stack([t[1] for t in transition_batch]).type(torch.FloatTensor)
            action_batch = torch.concat([t[2] for t in transition_batch])
            # reward_batch = torch.tensor([torch.tensor(t[5]) for t in transition_batch]).unsqueeze(1).type(
            #     torch.FloatTensor)
            # next_state_batch = torch.stack([t[3] for t in transition_batch]).type(torch.FloatTensor)
            # next_state2d_batch = torch.stack([t[4] for t in transition_batch]).type(torch.FloatTensor)
            t_batch = torch.tensor([torch.tensor(t[6]) for t in transition_batch]).unsqueeze(1)
            old_value_batch = torch.tensor([old_values[j] for j in perm[(i * batch_size):((i + 1) * batch_size)]])
            old_prob_batch = torch.tensor(
                [old_log_probs[j] for j in perm[(i * batch_size):((i + 1) * batch_size)]])
            # yield state_batch.to(self.device), state2d_batch.to(self.device), action_batch.to(self.device), \
            #       reward_batch.to(self.device), next_state_batch.to(self.device), next_state2d_batch.to(
            #     self.device), t_batch.to(self.device), old_value_batch.to(self.device), old_prob_batch.to(self.device)
            # yield state_batch.to(self.device), state2d_batch.to(self.device), action_batch.to(self.device), \
            #       reward_batch.to(self.device), t_batch.to(self.device), old_value_batch.to(self.device), old_prob_batch.to(self.device)
            yield state_batch.to(self.device), state2d_batch.to(self.device), action_batch.to(self.device), \
                t_batch.to(self.device), old_value_batch.to(
                self.device), old_prob_batch.to(self.device)

    def clear(self):
        self.memory = deque([], maxlen=self.size)

    def forget(self):
        # forget the old memory
        [self.memory.popleft() for i in range(9*len(self.memory)//10)]

    def save(self, folder, epoch):
        # save the memory
        with open(folder + '/memory_' + str(epoch) + '.pickle', 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, folder, epoch):
        with open(folder + '/memory_' + str(epoch) + '.pickle', 'rb') as f:
            self.memory = pickle.load(f)

    def __len__(self):
        return len(self.memory)
