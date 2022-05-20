#
CPU='cpu'
GPU='cuda' 
EPOCHS=50

SEEDS=(5 8 47 94 106)


pricing_alg1="TD3_MLP"
pricing_alg2="TD3_CNN"
pricing_alg3="PPO_MLP"
pricing_alg4="PPO_CNN"

#
data_folder=""

for arg in $*; do
    #
    if [[ ${arg} == "grid_3_static" ]]; then
        #
        data_folder="grid_small_static/3/"
    fi
    if [[ ${arg} == "grid_3_dynamic" ]]; then
        #
        data_folder="grid_small_dynamic/3/"
    fi
    if [[ ${arg} == "grid_4_static" ]]; then
        #
        data_folder="grid_small_static/4/"
    fi
    if [[ ${arg} == "grid_4_dynamic" ]]; then
        #
        data_folder="grid_small_dynamic/4/"
    fi
    if [[ ${arg} == "grid_5_static" ]]; then
        #
        data_folder="grid_small_static/5/"
    fi
    if [[ ${arg} == "grid_5_dynamic" ]]; then
        #
        data_folder="grid_small_dynamic/5/"
    fi
    if [[ ${arg} == "nyc_data" ]]; then
        #
        data_folder="nyc_large/"
    fi
done


# python main.py --data_folder ${small_data3_dynamic} --pricing_alg dummy \
#         --device ${CPU} --n_epochs ${EPOCHS} -m test

# python main.py --data_folder ${small_data1_dynamic} --pricing_alg equilibrium \
#         --device ${CPU} --n_epochs ${EPOCHS} -m test


for name in 0 1 2 3 4; do
    for searching in 'Gaussian'; do
        for batch_size in 32; do
            for actor_lr in 0.00001; do
                for critic_lr in 0.001; do
                    # YOU SHOULD FILL IN THIS FUNCTION
                    python main.py --data_folder ${data_folder}${name}/ --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                        --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -sa ${searching}
                done
            done
        done
    done
done

for name in 0 1 2 3 4; do
    for searching in 'Gaussian'; do
        for batch_size in 32; do
            for actor_lr in 0.00001; do
                for critic_lr in 0.001; do
                    # YOU SHOULD FILL IN THIS FUNCTION
                    python main.py --data_folder ${data_folder}${name}/ --pricing_alg ${pricing_alg3} -alr ${actor_lr} -clr ${critic_lr}\
                        --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 2000 -sa PPO
                done
            done
        done
    done
done


# for name in 0; do
#     for searching in 'Gaussian'; do
#         if [[ ${nyc_data} -gt 0 ]]; then
#             # YOU SHOULD FILL IN THIS FUNCTION
#             python main.py --data_folder ${nyc_large} --pricing_alg ${pricing_alg1} \
#                 --device ${CPU} --n_epochs ${EPOCHS2} -m all -n ${name} --batch_size 64 --seed ${SEEDS[${name}]} -f 10 -u 100 -sa ${searching}
#             python main.py --data_folder ${nyc_large} --pricing_alg ${pricing_alg2} \
#                 --device ${GPU} --n_epochs ${EPOCHS2} -o -m all -n ${name} -k 5 -s 2 --batch_size 64  --seed ${SEEDS[${name}]} -f 10 -u 100 -sa ${searching}
#             python main.py --data_folder ${nyc_large} --pricing_alg ${pricing_alg2} \
#                 --device ${GPU} --n_epochs ${EPOCHS2} -m all -n ${name} -k 5 -s 2 --batch_size 64  --seed ${SEEDS[${name}]} -f 10 -u 100 -sa ${searching}
#         fi  
#     done
# done

# python main.py -d nyc_large/ -e 5 -v -p ddpg_MLP -m all -f 10 -u 100 


# if [[ ${nyc_large} -gt 0 ]]; then
#     # YOU SHOULD FILL IN THIS FUNCTION
#     python main.py --data_folder ${large_data} --pricing_alg ${pricing_alg1} \
#         --device ${CPU} --n_epochs ${EPOCHS} --frequency ${LARGE_FREQ} --update_frequency ${LARGE_UPDATE}
#     python main.py --data_folder ${large_data} --pricing_alg ${pricing_alg2} \
#         --device ${GPU} --n_epochs ${EPOCHS} --frequency ${LARGE_FREQ} --update_frequency ${LARGE_UPDATE}  -k 3 -s 2 -o
# fi
# if [[ ${minibatch} -gt 0 ]]; then
#     # YOU SHOULD FILL IN THIS FUNCTION
#     for bsz in 32 64 128 256; do
#         #
#         for actor_lr in 1e-3 1e-4 1e-5; do
#             #
#             for critic_lr in 1e-3 1e-4 1e-5; do
#                 #
#                 python main.py --data_folder ${small_data2} --batch_size ${bsz} --pricing_alg ${pricing_alg1}\
#                     --actor_lr ${actor_lr} --critic_lr ${critic_lr} --gpu_id ${DEVICE} --n_epochs ${EPOCHS}
#             done
#         done
#     done
# fi

# if [[ ${minibatch} -gt 0 ]]; then
#     # YOU SHOULD FILL IN THIS FUNCTION
#     for bsz in 32 64 128 256; do
#         #
#         for actor_lr in 1e-3 1e-4 1e-5; do
#             #
#             for critic_lr in 1e-3 1e-4 1e-5; do
#                 #
#                 python main.py --data_folder ${small_data1} --batch_size ${bsz} --pricing_alg ${pricing_alg2}\
#                     --actor_lr ${actor_lr} --critic_lr ${critic_lr} --gpu_id ${DEVICE} --n_epochs ${EPOCHS}
#             done
#         done
#     done
# fi

# if [[ ${minibatch} -gt 0 ]]; then
#     # YOU SHOULD FILL IN THIS FUNCTION
#     for bsz in 32 64 128 256; do
#         #
#         for actor_lr in 1e-3 1e-4 1e-5; do
#             #
#             for critic_lr in 1e-3 1e-4 1e-5; do
#                 #
#                 python main.py --data_folder ${small_data2} --batch_size ${bsz} --pricing_alg ${pricing_alg2}\
#                     --actor_lr ${actor_lr} --critic_lr ${critic_lr} --gpu_id ${DEVICE} --n_epochs ${EPOCHS}
#             done
#         done
#     done
# fi
