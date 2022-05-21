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
small_cases=("grid_small_static/" "grid_small_dynamic/")
small_sizes=(3 4 5)
large_case="nyc_large/"
large_size=10

full_exp=0
ablation_exp=0
large_exp=0
small_exp=1

for arg in $*; do
    #
    if [[ ${arg} == "large" ]]; then
        #
        large_exp=1
        small_exp=0
    fi
    if [[ ${arg} == "full" ]]; then
        #
        full_exp=1
    fi
    if [[ ${arg} == "ablation" ]]; then
        #
        ablation_exp=1
    fi

done


if [ ${small_exp} -gt 0 ]; then
    if [ ${full_exp} -gt 0 ]; then
        for small_case in ${small_cases}; do
            for small_size in ${small_sizes}; do
                for name in 0; do
                    python main.py --data_folder ${small_case}${small_size}/${name}/  --pricing_alg dummy \
                    --device ${CPU} --n_epochs ${EPOCHS} -m test -n ${name} --seed ${SEEDS[${name}]} 
                    python main.py --data_folder ${small_case}${small_size}/${name}/  --pricing_alg equilibrium \
                    --device ${CPU} --n_epochs ${EPOCHS} -m test -n ${name} --seed ${SEEDS[${name}]} 
                    for searching in 'Gaussian'; do
                        for batch_size in 32; do
                            for actor_lr in 0.0001 0.00001; do
                                for critic_lr in 0.001 0.0001; do
                                    # YOU SHOULD FILL IN THIS FUNCTION
                                    python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                                        --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -sa ${searching} -pd 0
                                    python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                        --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -sa ${searching} -pd 0 -k ${small_size} -s 2
                                    python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg3} -alr ${actor_lr} -clr ${critic_lr}\
                                        --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -sa ${searching} -u 2000
                                    python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg4} -alr ${actor_lr} -clr ${critic_lr}\
                                        --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -sa ${searching} -k ${small_size} -s 2 -u 2000
                                done
                            done
                        done
                    done
                done
            done
        done
    fi



    if [ ${ablation_exp} -gt 0 ]; then
        small_case=${small_cases[1]}
        small_size=${small_sizes[1]}
        for name in 0; do
            for searching in 'Gaussian'; do
                for batch_size in 32; do
                    for actor_lr in 0.00001; do
                        for critic_lr in 0.001; do
                            for pd in -1 30; do
                                # YOU SHOULD FILL IN THIS FUNCTION
                                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                                        --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -sa ${searching} -pd ${pd}
                                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                    --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -sa ${searching} -pd ${pd} -k ${small_size} -s 2
                            done
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                    --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -sa ${searching} -pd 0 -k ${small_size} -s 2 -o 
                        done
                    done
                done
            done
        done
    fi

fi

if [ ${large_exp} -gt 0 ]; then
    if [ ${full_exp} -gt 0 ]; then
        for name in 0; do
            python main.py --data_folder ${large_case}  --pricing_alg dummy \
            --device ${CPU} --n_epochs ${EPOCHS} -m test -n ${name} --seed ${SEEDS[${name}]} 
            python main.py --data_folder ${large_case}  --pricing_alg equilibrium \
            --device ${CPU} --n_epochs ${EPOCHS} -m test -n ${name} --seed ${SEEDS[${name}]} 
            for searching in 'Gaussian'; do
                for batch_size in 32; do
                    for actor_lr in 0.0001 0.00001; do
                        for critic_lr in 0.001 0.0001; do
                            # YOU SHOULD FILL IN THIS FUNCTION
                            python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -sa ${searching} -pd 0 -f 10 -u 100 
                            python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -sa ${searching} -pd 0 -k ${large_size} -s 2 -f 10 -u 100 
                            python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg3} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -sa ${searching} -f 10
                            python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg4} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -sa ${searching} -k ${large_size} -s 2 -f 10 -u 2000
                        done
                    done
                done
            done
        done
    fi



    if [ ${ablation_exp} -gt 0 ]; then
        small_case=${small_cases[1]}
        small_size=${small_sizes[1]}
        for name in 0; do
            for searching in 'Gaussian'; do
                for batch_size in 32; do
                    for actor_lr in 0.00001; do
                        for critic_lr in 0.001; do
                            for pd in -1 30; do
                                # YOU SHOULD FILL IN THIS FUNCTION
                                python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                                        --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -sa ${searching} -pd ${pd} -f 10 -u 100 
                                python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                    --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -sa ${searching} -pd ${pd} -k ${large_size} -s 2 -f 10 -u 100 
                            done
                            python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                    --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -sa ${searching} -pd 0 -k ${large_size} -s 2 -o -f 10 -u 100 
                        done
                    done
                done
            done
        done
    fi

fi
# python main.py -d nyc_large/ -e 5 -v -p ddpg_MLP -m all -f 10 -u 100 

