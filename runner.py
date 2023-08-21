import numpy as np
import pandas as pd
import torch

def price_sensitivity(pricing_multipliers, mode = False):
    # A monotonic decreasing function that describe the price elasticity
    # To make the problem challenging (and more fun), we choose the nonlinear function such that the increase of profit
    # can only come from the improvement of service efficiency (serving more passengers) with less empty mileage
    if mode:
        return max(2.0 - pricing_multipliers, 0)
    else:
        return 1.0 / pricing_multipliers

def train(env, controller, dd_train, T_train, baseline, writer, args):
    thrng = torch.Generator("cpu")
    # print(np.sum(baseline))
    controller.pricer.initialize(thrng)
    if args.device == 'cuda':
        controller.pricer.cuda()
    if args.verbose:
        print("Start training")

    res = []  # pd.DataFrame(columns = 'epoch', 't', 'policy', 'profit', 'welfare')

    # one day to warm up to obtain the initial active vehicle distribution
    for t in range(max(T_train-1440, 0), T_train):
        # print(t)
        temp_cost, temp_schedule = controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)),
                                                             env.veh_count)
        ongoing_veh = env.get_ongoing_veh()
        if t % args.frequency == 0:
            if args.pricing_alg == 'dummy':
                # no need to train
                ...
            elif args.pricing_alg == 'equilibrium':
                ...
            else:
                temp_p = controller.update_price(env.pass_count, env.veh_count, \
                                             ongoing_veh, env.avg_profit, env.zone_profit,
                                             env.pricing_multipliers[0:max(5 // args.frequency, 1), :],
                                             t, 'train',
                                             od_permutation=args.od_permutation,
                                             policy_constr=args.policy_constr,
                                             last_pricing=env.pricing_multipliers[0, :])
            dd_train_ = np.zeros(dd_train[t:(t + args.frequency), :, :].shape)
            ind = dd_train[t:(t + args.frequency), :, :] > 0
            dd_train_[ind] = np.random.poisson(dd_train[t:(t + args.frequency), :, :][ind])
            temp_demands = np.floor(dd_train_ * price_sensitivity(temp_p[None, :, None], args.pricing_mode))
            temp_demands += np.random.binomial(1, (
                        dd_train_ * price_sensitivity(temp_p[None, :, None], args.pricing_mode) - temp_demands))
        temp_demand = temp_demands[t % args.frequency]
        env.step(temp_demand, temp_schedule, temp_p)

    for e in range(args.resume):
        controller.decay_searching_variance()

    for e in range(args.resume, args.n_epochs):
        total_time_step = 0
        total_profit = 0
        total_reposition = 0
        total_expense = 0
        total_served_pass = 0
        total_left_pass = 0

        total_occu_mile = 0
        total_occu_minute = 0
        total_empty_mile = 0
        total_empty_minute = 0

        temp_reward = 0
        temp_demands = np.zeros((args.frequency, args.num_zone))

        if args.verbose:
            print("Epoch " + str(e))
        for t in range(T_train):
            with torch.no_grad():
                temp_cost, temp_schedule = controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)),
                                                                     env.veh_count)
                if t % args.frequency == 0:
                    if args.pricing_alg == 'dummy':
                        # no need to train
                        ...
                    elif args.pricing_alg == 'equilibrium':
                        ...
                    else:
                        ongoing_veh = env.get_ongoing_veh()
                        controller.add_memory(env.pass_count, env.veh_count, ongoing_veh, env.avg_profit, env.zone_profit,\
                                              env.pricing_multipliers[0:max(5 // args.frequency, 1), :], temp_reward, t,
                                              args.od_permutation)
                        temp_p = controller.update_price(env.pass_count, env.veh_count, \
                                                         ongoing_veh, env.avg_profit, env.zone_profit,
                                                         env.pricing_multipliers[0:max(5 // args.frequency, 1), :],
                                                         t, 'train',
                                                         od_permutation=args.od_permutation,
                                                         policy_constr=args.policy_constr,
                                                         last_pricing=env.pricing_multipliers[0, :])

                    dd_train_ = np.zeros(dd_train[t:(t + args.frequency), :, :].shape)
                    ind = dd_train[t:(t + args.frequency), :, :] > 0
                    dd_train_[ind] = np.random.poisson(dd_train[t:(t + args.frequency), :, :][ind])
                    temp_demands = np.floor(dd_train_ * price_sensitivity(temp_p[None, :, None], args.pricing_mode))
                    temp_demands += np.random.binomial(1, (
                            dd_train_ * price_sensitivity(temp_p[None, :, None], args.pricing_mode) - temp_demands))
                    temp_reward = 0  # refresh the temp_reward

                temp_demand = temp_demands[t % args.frequency]
                temp_profit, temp_expense, temp_served_pass, temp_left_pass, occu_mile, occu_minute, empty_mile, empty_minute = env.step(temp_demand, temp_schedule, temp_p)
                temp_reward += temp_profit - temp_expense - baseline[t]
            # train the model
            # if t == 0 and e == 1:
            #     controller.train_pricing_value(1000, args.batch_size)
            if args.pricing_alg.startswith('TD3') and (e * T_train + t + 1) % (args.frequency) == 0: 
                controller.train_pricing_policy(((e - 1) * T_train + t + 1) //args.frequency, \
                                                T_train // args.frequency, args.batch_size, args.update_frequency)
            elif args.pricing_alg.startswith('PPO') and (e * T_train + t + 1) % (args.frequency * args.update_frequency) == 0:
                controller.train_pricing_policy(50, \
                                                args.update_frequency, args.batch_size)
            elif args.pricing_alg.startswith('SAC') and (e * T_train + t + 1) % (args.frequency * args.update_frequency) == 0:
                controller.train_pricing_policy(((e - 1) * T_train + t + 1) //(args.frequency * args.update_frequency), \
                                                args.update_frequency, args.batch_size)

            total_time_step += 1
            total_profit += temp_profit
            total_reposition += temp_cost
            total_expense += temp_expense
            total_served_pass += temp_served_pass
            total_left_pass += temp_left_pass
            total_occu_mile += occu_mile
            total_occu_minute += occu_minute
            total_empty_mile += empty_mile
            total_empty_minute += empty_minute

            if t % 120 == 0:
                writer.flush()
                if args.verbose:
                    # print(np.diag(env.veh_count))
                    print("Time step: " + str(t) + ", income: " + str(temp_profit) + \
                          ", expense: " + str(temp_expense) + \
                          ", active veh: " + str(env.active_veh) + \
                          ", served_pass: " + str(temp_served_pass) + \
                          ", left_pass: " + str(temp_left_pass) + \
                          ", new demand: " + str(np.sum(temp_demand)))

        controller.decay_searching_variance()
        print("Total time step: " + str(total_time_step) + ", reward: " + str(
            total_profit - total_expense) + ", profit: " + str(total_profit) + \
              ", expense: " + str(total_expense) + ", served_pass: " + str(
            total_served_pass) + ", left_pass: " + str(total_left_pass))
        res.append([e, total_profit, total_expense, total_served_pass, total_left_pass,\
                    total_occu_mile, total_occu_minute, total_empty_mile, total_empty_minute])
        if env.active_veh == 0: # run into an absorbing stage in which no drive wants to work, need to reset the env
            env.reset()
            controller.buffer.clear()
        writer.add_scalar("Reward/train", total_profit - total_expense, e)
        writer.add_scalar("Served_passenger/train", total_served_pass, e)
        writer.add_scalar("Left_passenger/train", total_left_pass, e)

    res = pd.DataFrame(res, columns=['epoch', 'income', 'expense', 'served', 'left', 'occu_mile', 'occu_minute', 'empty_mile', 'empty_minute'])
    res.to_csv(
        args.store_res_folder+"/"+"train_log_"+str(args.n_epochs)+".csv",
        index=None)

    # save the model in the end
    controller.pricer.save_model(
        args.store_model_folder + "/", args.n_epochs)
    controller.buffer.save(args.store_model_folder + "/", args.n_epochs)

    # store the model
    # print("Store model")
    # if (not os.path.isdir(
    #         'models/' + args.name+'/'+ args.data_folder + args.pricing_alg + permutation_tag + "/" + str(
    #             args.batch_size) + "_" + str(
    #             args.actor_lr) + "_" + str(args.critic_lr) + str(args.epoch) + "/")):
    #     os.makedirs('models/' +  args.name+'/'+args.data_folder + args.pricing_alg + permutation_tag + "/" + str(
    #         args.batch_size) + "_" + str(
    #         args.actor_lr) + "_" + str(args.critic_lr) + str(args.epoch) + "/")
    # controller.pricer.save_model(
    #     'models/' +args.name+'/' + args.data_folder + args.pricing_alg + permutation_tag + "/" + str(args.batch_size) + "_" + str(
    #         args.actor_lr) + "_" + str(args.critic_lr) + str(args.epoch) + "/")

def test(env, controller, dd_train, dd_test, T_test, args, k):
    if args.verbose:
        print("Start testing")
    res = []
    controller.pricer.eval()
    total_time_step = 0
    total_profit = 0
    total_reposition = 0
    total_expense = 0
    total_served_pass = 0
    total_left_pass = 0
    temp_demands = np.zeros((args.frequency, args.num_zone))

    for t in range(max(T_test-1440, 0), T_test):
        print(t)
        temp_cost, temp_schedule = controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)),
                                                             env.veh_count)
        # temp_p = controller.dummy_price()
        if t % args.frequency == 0:
            if args.pricing_alg == 'dummy':
                # no need to train
                temp_p = controller.dummy_price()
            elif args.pricing_alg == 'equilibrium':
                temp_p = controller.equilibrium_price(np.sum(env.pass_count, axis=(0)), env.active_veh ,
                                                          [dd_train[(t + i) % T_test, :, :] for i in range(10)], args.data_folder)
            else:
                ongoing_veh = env.get_ongoing_veh()
                temp_p = controller.update_price(env.pass_count, env.veh_count, \
                                                 ongoing_veh, env.avg_profit, env.zone_profit,
                                                 env.pricing_multipliers[0:max(5 // args.frequency, 1), :],
                                                 t, 'train',
                                                 od_permutation=args.od_permutation,
                                                 policy_constr=args.policy_constr,
                                                 last_pricing=env.pricing_multipliers[0, :])
            dd_train_ = np.zeros(dd_train[t:(t + args.frequency), :, :].shape)
            ind = dd_train[t:(t + args.frequency), :, :] > 0
            dd_train_[ind] = np.random.poisson(dd_train[t:(t + args.frequency), :, :][ind])
            temp_demands = np.floor(dd_train_ * price_sensitivity(temp_p[None, :, None], args.pricing_mode))
            temp_demands += np.random.binomial(1, (
                    dd_train_ * price_sensitivity(temp_p[None, :, None], args.pricing_mode) - temp_demands))
        temp_demand = temp_demands[t % args.frequency]
        dd_train_ = np.random.poisson(dd_train[t, :, :]).astype(int)
        temp_demand = dd_train_
        # temp_demand += np.random.binomial(1, (dd_train_ / temp_p[:, None] - temp_demand))
        env.step(temp_demand, temp_schedule, temp_p)

    for t in range(T_test):
        print(t)
        temp_cost, temp_schedule = controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)),
                                                             env.veh_count)
        if t % args.frequency == 0:
            if args.pricing_alg == 'dummy':
                temp_p = controller.dummy_price()
            elif args.pricing_alg == 'equilibrium':
                temp_p = controller.equilibrium_price(np.sum(env.pass_count, axis=(0)),env.active_veh ,
                                                          [dd_train[(t + i) % T_test, :, :] for i in range(10)], args.data_folder)
            else:
                ongoing_veh = env.get_ongoing_veh()
                temp_p = controller.update_price(env.pass_count, env.veh_count, \
                                                         ongoing_veh, env.avg_profit, env.zone_profit,
                                                         env.pricing_multipliers[0:max(5 // args.frequency, 1), :],
                                                         t, 'test',
                                                         od_permutation=args.od_permutation,
                                                         policy_constr=args.policy_constr,
                                                         last_pricing=env.pricing_multipliers[0, :])
            temp_demands = np.floor(dd_test[t:(t + args.frequency), :, :] * price_sensitivity(temp_p[None, :, None], args.pricing_mode))
            temp_demands += np.random.binomial(1, (
                    dd_test[t:(t + args.frequency), :, :] * price_sensitivity(temp_p[None, :, None], args.pricing_mode) - temp_demands))

        temp_demand = temp_demands[t % args.frequency]
        temp_profit, temp_expense, temp_served_pass, temp_left_pass, occu_mile, occu_minute, empty_mile, empty_minute = env.step(temp_demand, temp_schedule, temp_p)
        total_time_step += 1
        total_profit += temp_profit
        total_reposition += temp_cost
        total_expense += temp_expense
        total_served_pass += temp_served_pass
        total_left_pass += temp_left_pass

        # writer.add_scalar("Reward/test", temp_profit + temp_welfare, t)
        # writer.add_scalar("Served_passenger/test", temp_served_pass, t)
        # writer.add_scalar("Left_passenger/test", temp_left_pass, t)

        res.append([t, temp_p, temp_profit, temp_expense, env.active_veh, temp_served_pass, temp_left_pass, occu_mile, occu_minute, empty_mile, empty_minute])
        if t % 120 == 0:
            # writer.flush()
            if args.verbose:
                # print(np.diag(env.veh_count))
                print("Time step: " + str(t) + ", profit: " + str(temp_profit) + ", expense: " + str(
                    temp_expense) + \
                      ", active veh: " + str(env.active_veh) + \
                      ", served_pass: " + str(temp_served_pass) + \
                      ", left_pass: " + str(temp_left_pass) + \
                      ", new demand: " + str(np.sum(temp_demand)))
    print("Total time step: " + str(total_time_step) + ", reward: " + str(
        total_profit - total_expense) + ", profit: " + str(total_profit) + \
          ", expense: " + str(total_expense) + ", served_pass: " + str(
        total_served_pass) + ", left_pass: " + str(
        total_left_pass))
    res = pd.DataFrame(res, columns=['t', 'policy', 'profit', 'expense', 'act_veh', 'served', 'left', 'occu_mile', 'occu_minute', 'empty_mile', 'empty_minute'])
    res.to_csv(
        args.store_res_folder + "/" + f"test_log_{k}.csv",
        index=None)
    env.reset()