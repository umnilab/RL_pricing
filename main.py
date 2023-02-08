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
# python main.py -d grid_small_dynamic/3/0/ --pricing_alg TD3_MLP -alr 0.00001 -clr 0.001 --n_epochs 50 -m all -n 0 --batch_size 32 --seed 5 -sa Gaussian -pd 0 -pe 0 -ac

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='SpatioTemporal Pricing')
    # Simulation settings
    parser.add_argument('-d', '--data_folder',
                        help='the folder that contains all the input data')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-m', '--mode', type=str, default='all',
                        help='running mode, test or all which means include both train and test. (DEFAULT: all)')
    parser.add_argument('-n', '--name', type=str, default='0', help='name_of the run')
    parser.add_argument('-rs', '--seed', type=int, default=47, help='random seed')
    # Training settings
    parser.add_argument('-e', '--n_epochs', type=int, default=10,
                        help='number of epochs (DEFAULT: 10)')
    parser.add_argument('-p', '--pricing_alg', type=str, default='dummy',
                        help='pricing algorithms, can be ddpg_MLP, ddpg_CNN, dummy, or equilibrium')
    parser.add_argument('-c', '--device', type=str, default='cpu',
                        help='cpu or gpu (cuda)')
    parser.add_argument('-f', '--frequency', type=int, default=1,
                        help='pricing freqency')  # update the price every minute
    parser.add_argument('-pc', '--policy_constr', action='store_true', default=False,
                        help='constrained pricing change speed')  # constrained policy
    parser.add_argument('-re', '--resume', type=int, default=0,
                        help='resume_prev_training')
    # Hyper parameters for RL algorithms
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch_size. (DEFAULT: 32)')
    parser.add_argument('-k', '--kernel_size', type=int, default=3,
                        help='kernel size of CNN layer')
    parser.add_argument('-s', '--stride', type=int, default=2,
                        help='stride of CNN layer')
    parser.add_argument('-o', '--od_permutation', action='store_false', default=True,
                        help='enable the permutation of OD features')
    parser.add_argument('-alr', '--actor_lr', type=float, default=0.00001, help='learning rate of actor')
    parser.add_argument('-clr', '--critic_lr', type=float, default=0.001, help='learning rate of critic')
    parser.add_argument('-u', '--update_frequency', type=int, default=25,
                        help='base frequency for policy update') 
    parser.add_argument('-ps', '--pooling', type=int, default=2, help='kernel/stride size of pooling layer for downsampling')
    parser.add_argument('-sa', '--searching', type=str, default='Gaussian', help='searching mechanism')
    parser.add_argument('-pd', '--policy_delay', type = int, default = 0,
                        help='incremental policy delay') # -1: n; 0: n/ln(n); other values: constant
    parser.add_argument('-pe', '--position_encode', type=int, default= 64,\
                        help='whether to use NN for positional encoding')
    parser.add_argument('-ac', '--auto_correlation', action='store_true', default=False,
                        help='whether to use auto-correlated noise for exploration')
    parser.add_argument('-fg', '--forget', action='store_true', default=False,
                        help='whether to use forget in the critic network')
    parser.add_argument('-pm', '--pricing_mode', action='store_true', default=False,
                        help='demand sensitivity to price, either inverse or linear')
    args = parser.parse_args(argv)

    return args


if __name__ == '__main__':

    # loading arguments
    args = get_arguments(sys.argv[1:])

    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    permutation_tag = '' if not args.od_permutation else '_perm'
    constraint_tag = '' if not args.policy_constr else '_constr' # unused for this study
    auto_tag = '' if not args.auto_correlation else '_auto'
    forget_tag = '' if not args.forget else '_forget'

    scenario_tag = args.data_folder.replace("/", "_")
    settings_tag = args.pricing_alg + permutation_tag + constraint_tag + auto_tag + forget_tag + "_" +args.searching + "_" +\
                   str(args.policy_delay) +"_"+str(args.position_encode) + "_" + args.name
    hyperp_tag = str(args.batch_size) + "_" + str(args.actor_lr) + "_" + str(args.critic_lr) + "_" + str(args.kernel_size)

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

    if args.mode == 'all' and (os.path.exists(args.store_res_folder+"/"+"train_log_"+str(args.n_epochs)+".csv") or \
                               os.path.exists(args.store_res_folder+"/"+"test_log_1.csv")):
        print("Files are already there, exit!")
        sys.exit()

    if (args.mode == 'test') and (len(os.listdir(args.store_model_folder)) == 0) and (args.pricing_alg not in ['dummy','equilibrium']):
        print("No trained model can be found, exit!")
        sys.exit()

    if (args.mode == 'test') and os.path.exists(args.store_res_folder+"/"+"test_log_1.csv"):
        print("Files are already there, exit!")
        sys.exit()

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
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis = 0),
                              searching=args.searching,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train
                              )
    elif args.pricing_alg == 'PPO_MLP':
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, option='PPO_MLP',
                              device=args.device,
                              writer=writer, update_freq=args.frequency, veh_num=len(vp),
                              demand_mean=np.mean(dd_train, axis=0), demand_std=np.std(dd_train, axis=0),
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis=0),
                              searching=args.searching,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train
                              )
    elif args.pricing_alg == 'TD3_CNN':
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, option='TD3_CNN',
                              permutation=permutation, permutation2=permutation2, kernel_size=args.kernel_size,
                              stride=args.stride, pooling = args.pooling, device=args.device, writer=writer,
                              od_permutation=args.od_permutation, update_freq = args.frequency, veh_num = len(vp),\
                              demand_mean = np.mean(dd_train, axis = 0), demand_std = np.std(dd_train, axis = 0),
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis=0),
                              searching = args.searching, policy_delay = args.policy_delay,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train
                              )
    else:
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, device=args.device,
                              writer=writer, update_freq = args.frequency, veh_num= len(vp),
                              demand_mean = np.mean(dd_train, axis = 0), demand_std = np.std(dd_train, axis = 0),
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis=0),
                              searching = args.searching, policy_delay = args.policy_delay,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train
                              )

    env = Environment(td, tt, vp, args.num_zone, frequency=args.frequency)

    # input: travel distance, travel time matrices, initial vehicle distribution
    # output: time step, profit, reposition cost
    if args.mode == 'all':
        if args.resume > 0:
            if args.verbose:
                print("Load model")
            print("Resume training from epoch " + str(args.resume))
            controller.pricer.load_weights(
                args.store_model_folder + "/", args.resume)
            controller.buffer.load(args.store_model_folder + "/", args.resume)
            if args.device == 'cuda':
                controller.pricer.cuda()

            # update the searching variance
            for i in range(args.resume):
                controller.decay_searching_variance()

        baseline = (1 - env.delta) * (env.alpha0 * np.sum(dd_train, axis=(1, 2)) + env.alpha1 * np.sum(
            dd_train * td[None, :, :], axis=(1, 2)) + \
                   env.beta1 * np.sum(dd_train * tt[None, :, :], axis=(1, 2)))
        train(env, controller, dd_train, T_train, baseline, writer, args)
        # test(env, controller, dd_train, dd_test, T_test, args, 0)
    else:
        # load the model
        if args.verbose:
            print("Load model")
        if args.pricing_alg.startswith('TD3'):
            controller.pricer.load_weights(
                args.store_model_folder + "/", args.n_epochs)
            if args.device == 'cuda':
                controller.pricer.cuda()
        if args.pricing_alg.startswith('PPO'):
            controller.pricer.load_weights(
                args.store_model_folder + "/", args.n_epochs)
            controller.pricer.set_action_std(controller.min_action_std)
            if args.device == 'cuda':
                controller.pricer.cuda()
        # run exp
        for k in range(1, 6):
            test(env, controller, dd_train, dd_test, T_test, args, k)
    # collect result
    if args.mode == 'all':
        writer.close()
