#
CPU='cpu'
GPU='cuda' 
EPOCHS=500
EPOCHS2=500

SEEDS=(5 8 47 94 106)

pricing_alg1="TD3_MLP"
pricing_alg2="TD3_CNN"
pricing_alg3="PPO_MLP"
pricing_alg4="PPO_CNN"

#
small_case="grid_small_dynamic/" #("grid_small_static/" "grid_small_dynamic/")
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
        small_case="grid_small_static/"
        for small_size in 4; do
            for name in 4; do # too expensive so we just run one round
                python main.py --data_folder ${small_case}${small_size}/${name}/  --pricing_alg dummy \
                --device ${CPU} --n_epochs ${EPOCHS} -m test -n ${name} --seed ${SEEDS[${name}]} 
                python main.py --data_folder ${small_case}${small_size}/${name}/  --pricing_alg equilibrium \
                --device ${CPU} --n_epochs ${EPOCHS} -m test -n ${name} --seed ${SEEDS[${name}]} 
                for batch_size in 32; do
                    for actor_lr in 0.00001; do
                        for critic_lr in 0.001; do
                            # YOU SHOULD FILL IN THIS FUNCTION
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -fg
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -k ${small_size} -fg
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg3} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 10000 
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg4} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -k ${small_size} -u 10000
                        done
                    done
                done
            done
        done
    fi

    if [ ${full_exp} -gt 0 ]; then
        small_case="grid_small_dynamic/"
        for small_size in 4; do
            for name in 4; do
                python main.py --data_folder ${small_case}${small_size}/${name}/  --pricing_alg dummy \
                --device ${CPU} --n_epochs ${EPOCHS} -m test -n ${name} --seed ${SEEDS[${name}]} 
                python main.py --data_folder ${small_case}${small_size}/${name}/  --pricing_alg equilibrium \
                --device ${CPU} --n_epochs ${EPOCHS} -m test -n ${name} --seed ${SEEDS[${name}]} 
                for batch_size in 32; do
                    for actor_lr in 0.00001; do
                        for critic_lr in 0.001; do
                            # YOU SHOULD FILL IN THIS FUNCTION
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -fg
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -k ${small_size} -fg
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg3} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 10000 
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg4} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -k ${small_size} -u 10000
                        done
                    done
                done
            done
        done
    fi



    if [ ${ablation_exp} -gt 0 ]; then
        small_case="grid_small_static/"
        small_size=4
        for name in 4; do
            for batch_size in 32; do
                for actor_lr in 0.00001; do
                    for critic_lr in 0.001; do
                        for pd in -1 25; do
                            # YOU SHOULD FILL IN THIS FUNCTION
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                                    --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd}
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -k ${small_size}
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -fg
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -k ${small_size} -fg
                        done
                        python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -k ${small_size} -o -fg
                    done
                done
            done
        done
    fi

    if [ ${ablation_exp} -gt 0 ]; then
        small_case="grid_small_dynamic/"
        small_size=4
        for name in 4; do
            for batch_size in 32; do
                for actor_lr in 0.00001; do
                    for critic_lr in 0.001; do
                        for pd in -1 25; do
                            # YOU SHOULD FILL IN THIS FUNCTION
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                                    --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd}
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -k ${small_size} 
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                                    --device ${CPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -fg
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -k ${small_size} -fg
                        done
                        python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${GPU} --n_epochs ${EPOCHS} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -k ${small_size} -o -fg
                    done
                done
            done
        done
    fi

fi

if [ ${large_exp} -gt 0 ]; then
    if [ ${full_exp} -gt 0 ]; then
        for name in 4; do
            python main.py --data_folder ${large_case}  --pricing_alg dummy \
            --device ${CPU} --n_epochs ${EPOCHS2} -m test -n ${name} --seed ${SEEDS[${name}]} 
            python main.py --data_folder ${large_case}  --pricing_alg equilibrium \
            --device ${CPU} --n_epochs ${EPOCHS2} -m test -n ${name} --seed ${SEEDS[${name}]} 
            for batch_size in 32; do
                for actor_lr in 0.00001; do
                    for critic_lr in 0.001; do
                        # YOU SHOULD FILL IN THIS FUNCTION
                        python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                            --device ${CPU} --n_epochs ${EPOCHS2} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -f 10 -u 5 -fg
                        python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                            --device ${GPU} --n_epochs ${EPOCHS2} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -k ${large_size} -s 2 -f 10 -u 5 -fg
                        python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg3} -alr ${actor_lr} -clr ${critic_lr}\
                            --device ${CPU} --n_epochs ${EPOCHS2} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -f 10 -u 1000 
                        python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg4} -alr ${actor_lr} -clr ${critic_lr}\
                            --device ${GPU} --n_epochs ${EPOCHS2} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -k ${large_size} -s 2 -f 10 -u 1000 
                    done
                done
            done
        done
    fi


    if [ ${ablation_exp} -gt 0 ]; then
        for name in 4; do
            for batch_size in 32; do
                for actor_lr in 0.00001; do
                    for critic_lr in 0.001; do
                        for pd in -1 25; do
                            python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                                    --device ${CPU} --n_epochs ${EPOCHS2} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -f 10 -u 5 
                            python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${GPU} --n_epochs ${EPOCHS2} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -k ${large_size} -s 2 -f 10 -u 5
                            python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                                    --device ${CPU} --n_epochs ${EPOCHS2} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -f 10 -u 5 -fg
                            python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${GPU} --n_epochs ${EPOCHS2} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -k ${large_size} -s 2 -f 10 -u 5 -fg 
                        done
                        python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${GPU} --n_epochs ${EPOCHS2} -m all -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -k ${large_size} -s 2 -o -f 10 -u 5 -fg
                    done
                done
            done
        done
    fi
fi
# python main.py -d nyc_large/ -e 5 -v -p ddpg_MLP -m all -f 10 -u 100 

