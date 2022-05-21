# input: parameters
# output: trainded model under "train mode", tested results under "test mode"
from sim import Environment
from center import Platform
import sys
import numpy as np
import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from runner import train, test


# usage example
# python main.py -d grid_small_dynamic/3/0/ -e 5 -v
# python main.py -d nyc_large/ -e 5 -v -p dummy -m test
# python main.py -d nyc_large/ -e 5 -v -p ddpg_CNN -m all -k 5 -s 2 -f 10 -u 100
# python main.py -d nyc_large/ -e 5 -v -p ddpg_MLP -m all -f 10 -u 100

# TODO:
# Check the policy
# Implement PPO


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='SpatioTemporal Pricing')
    parser.add_argument('-d', '--data_folder',
                        help='the folder that contains all the input data')
    parser.add_argument('-e', '--n_epochs', type=int, default=10,
                        help='number of epochs (DEFAULT: 10)')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='batch_size. (DEFAULT: 32)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-m', '--mode', type=str, default='all',
                        help='running mode, test or all which means include both train and test. (DEFAULT: all)')
    parser.add_argument('-p', '--pricing_alg', type=str, default='dummy',
                        help='pricing algorithms, can be ddpg_MLP, ddpg_CNN, dummy, or equilibrium')
    parser.add_argument('-k', '--kernel_size', type=int, default=3,
                        help='kernel size of CNN layer')
    parser.add_argument('-s', '--stride', type=int, default=2,
                        help='stride of CNN layer')
    parser.add_argument('-c', '--device', type=str, default='cpu',
                        help='cpu or gpu (cuda)')
    parser.add_argument('-f', '--frequency', type=int, default=1,
                        help='pricing freqency')
    parser.add_argument('-u', '--update_frequency', type=int, default=10,
                        help='training_frequency')
    parser.add_argument('-o', '--od_permutation', action='store_false', default=True,
                        help='enable the permutation of OD features')
    parser.add_argument('-n', '--name', type=str, default='0', help='name_of the run')
    parser.add_argument('-ps', '--pooling', type=int, default=2, help='kernel/stride size of pooling layer for downsampling')
    parser.add_argument('-alr', '--actor_lr', type=float, default=0.0001, help='learning rate of actor')
    parser.add_argument('-clr', '--critic_lr', type=float, default=0.001, help='learning rate of critic')
    parser.add_argument('-sa', '--searching', type=str, default='Greedy', help='searching mechanism')
    parser.add_argument('-rs', '--seed', type=int, default=47, help='random seed')
    parser.add_argument('-pc', '--policy_constr', action='store_true', default=False,
                        help='constrained pricing change speed')
    parser.add_argument('-pd', '--policy_delay', type = int, default = 30,
                        help='incremental policy delay')


    args = parser.parse_args(argv)

    return args


if __name__ == '__main__':

    # loading arguments
    args = get_arguments(sys.argv[1:])

    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    permutation_tag = '' if args.od_permutation else '_no_perm'
    constraint_tag = '' if args.policy_constr else '_no_constr'

    scenario_tag = args.data_folder.replace("/", "_")
    settings_tag = args.pricing_alg + permutation_tag + constraint_tag + "_" +args.searching + "_" + str(args.policy_delay) + "_" + args.name
    hyperp_tag = str(args.batch_size) + "_" + str(args.actor_lr) + "_" + str(args.critic_lr)

    # prepare the folders for storing the results
    store_run_folder = "runs/" + scenario_tag
    store_model_folder = 'models/' + scenario_tag + '/' + settings_tag + '/' + hyperp_tag
    store_res_folder = "results/" + scenario_tag + '/' + settings_tag + '/' + hyperp_tag

    if (not os.path.isdir(
            store_run_folder + "/")):
        os.makedirs(
            store_run_folder + "/")

    if (not os.path.isdir(
            store_model_folder + "/")):
        os.makedirs(
            store_model_folder + "/")

    if (not os.path.isdir(
            store_res_folder + "/")):
        os.makedirs(
            store_res_folder + "/")

    args.store_run_folder = store_run_folder
    args.store_model_folder = store_model_folder
    args.store_res_folder = store_res_folder

    # load data
    if args.verbose:
        print("Loading data")

    # time_tag = datetime.datetime.now().strftime("%m-%d-%H-%M")
    writer = None
    if args.mode == 'all':
        writer = SummaryWriter(store_run_folder + "/" + settings_tag + "_" + hyperp_tag)

    td = np.load(args.data_folder + 'td.npy')
    tt = np.load(args.data_folder + 'tt.npy')
    vp = np.load(args.data_folder + 'vp.npy')
    dd_train = np.load(args.data_folder + 'dd_train.npy')
    dd_test = np.load(args.data_folder + 'dd_test.npy')
    permutation = np.load(args.data_folder + 'permutation.npy')
    permutation2 = np.load(args.data_folder + 'permutation2.npy')
    args.num_zone = td.shape[0]
    T_train = dd_train.shape[0]
    T_test = dd_test.shape[0]

    # initialize environment
    if args.verbose:
        print("Initialize environment, vehicle num: " + str(len(vp)))
    if args.pricing_alg == 'PPO_CNN':
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, option='PPO_CNN',
                              permutation=permutation, permutation2=permutation2, kernel_size=args.kernel_size,
                              stride=args.stride, pooling=args.pooling, device=args.device, writer=writer,
                              od_permutation=args.od_permutation, update_freq=args.frequency,
                              veh_num=len(vp), \
                              demand_mean=np.mean(dd_train, axis=0), demand_std=np.std(dd_train, axis=0),
                              searching=args.searching)
    elif args.pricing_alg == 'PPO_MLP':
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, option='PPO_MLP',
                              device=args.device,
                              writer=writer, update_freq=args.frequency, veh_num=len(vp),
                              demand_mean=np.mean(dd_train, axis=0), demand_std=np.std(dd_train, axis=0),
                              searching=args.searching)
    elif args.pricing_alg == 'TD3_CNN':
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, option='TD3_CNN',
                              permutation=permutation, permutation2=permutation2, kernel_size=args.kernel_size,
                              stride=args.stride, pooling = args.pooling, device=args.device, writer=writer,
                              od_permutation=args.od_permutation, update_freq = args.frequency, veh_num = len(vp),\
                              demand_mean = np.mean(dd_train, axis = 0), demand_std = np.std(dd_train, axis = 0),
                              searching = args.searching, policy_delay = args.policy_delay)
    else:
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, device=args.device,
                              writer=writer, update_freq = args.frequency, veh_num= len(vp),
                              demand_mean = np.mean(dd_train, axis = 0), demand_std = np.std(dd_train, axis = 0),
                              searching = args.searching, policy_delay = args.policy_delay)

    env = Environment(td, tt, vp, args.num_zone, frequency=args.frequency)

    # TODO: relocate this to runners
    # input: travel distance, travel time matrices, initial vehicle distribution
    # output: time step, profit, reposition cost
    if args.mode == 'all':
        baseline = env.alpha0 * np.sum(dd_train, axis=(1, 2)) + (env.alpha1 - env.alpha1_1 - env.alpha2) * np.sum(
            dd_train * td[None, :, :], axis=(1, 2)) + \
                   (env.beta1 - env.beta1_1 -env.beta2) * np.sum(dd_train * tt[None, :, :], axis=(1, 2))
        train(env, controller, dd_train, T_train, baseline, writer, args)
        test(env, controller, dd_train, dd_test, T_test, args, 0)
    else:
        # load the model
        if args.verbose:
            print("Load model")
        if args.pricing_alg.startswith('TD3'):
            controller.pricer.load_weights(
                args.store_model_folder + "/")
            if args.device == 'cuda':
                controller.pricer.cuda()
        if args.pricing_alg.startswith('PPO'):
            controller.pricer.load_weights(
                args.store_model_folder + "/")
            controller.pricer.set_action_std(controller.min_action_std)
            if args.device == 'cuda':
                controller.pricer.cuda()
        # run exp
        for k in range(1, 6):
            test(env, controller, dd_train, dd_test, T_test, args, k)
    # collect result
    if args.mode == 'all':
        writer.close()