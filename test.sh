# for test
device='cuda'
EPOCHS=100

SEEDS=(5 8 47 94 106)

pricing_alg1="TD3_MLP"
pricing_alg2="TD3_CNN"
pricing_alg3="PPO_MLP"
pricing_alg4="PPO_CNN"
pricing_alg5="SAC_MLP"
pricing_alg6="SAC_CNN"


#
small_case="grid_small_dynamic/" #("grid_small_static/" "grid_small_dynamic/")
large_case="nyc_small/"
small_size=4
large_size=5

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
        for name in 4; do # this is for test, which does not matter for training
            python main.py --data_folder ${small_case}${small_size}/${name}/  --pricing_alg dummy \
            --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --seed ${SEEDS[${name}]}
            python main.py --data_folder ${small_case}${small_size}/${name}/  --pricing_alg equilibrium \
            --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --seed ${SEEDS[${name}]}
            for batch_size in 64; do
                for actor_lr in 0.00001 0.00005 0.0001 0.0005 0.001; do
                    for critic_lr in 0.001 0.005 0.01; do
                        if (( $(echo "$actor_lr > $critic_lr" | bc -l) )); then
                            :
                        else
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                                 --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -fg
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg3} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 2000
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg5} -alr ${actor_lr} -clr ${critic_lr}\
                                 --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 50
                            for filter_size in 3 4; do
                                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                    --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -k ${filter_size} -fg
                                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg4} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -k ${small_size} -u 2000
                                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg6} -alr ${actor_lr} -clr ${critic_lr}\
                                    --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 50 -k ${filter_size}
                            done
                        fi
                    done
                done
            done
        done
    fi

    if [ ${full_exp} -gt 0 ]; then
        small_case="grid_small_dynamic/"
        for name in 4; do
            python main.py --data_folder ${small_case}${small_size}/${name}/  --pricing_alg dummy \
            --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --seed ${SEEDS[${name}]}
            python main.py --data_folder ${small_case}${small_size}/${name}/  --pricing_alg equilibrium \
            --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --seed ${SEEDS[${name}]}
            for batch_size in 64; do
                for actor_lr in 0.00001 0.00005 0.0001 0.0005 0.001; do
                    for critic_lr in 0.001 0.005 0.01; do
                        if (( $(echo "$actor_lr > $critic_lr" | bc -l) )); then
                            :
                        else
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -fg
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg3} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 2000
                            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg5} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 50
                            for filter_size in 3 4; do
                                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -k ${filter_size} -fg
                                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg4} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -k ${filter_size} -u 2000
                                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg6} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 50 -k ${filter_size}
                            done
                        fi
                    done
                done
            done
        done
    fi


    # ablation study, choose the best parameter batch_size, actor_lr and critic_lr
    batch_size=64
    actor_lr=5e-5
    if [ ${ablation_exp} -gt 0 ]; then
        small_case="grid_small_static/"
        small_size=3
        for name in 4; 
            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg3} -alr 0.0005 -clr 0.001\
                    --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 2000 -fg
            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg5} -alr 0.0001 -clr 0.001\
                     --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 50 -fg
            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg4} -alr 0.0005 -clr 0.001\
                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -k 4 -u 2000 -fg
            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg6} -alr 0.0001 -clr 0.001\
                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 50 -k 3 -fg
            for pd in 0; do
                # without forget mechanism
                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr 0.001\
                        --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd}
                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr 0.01\
                    --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -k ${small_size}
            done
            for pd in 1 10; do
                # without incremental delay
                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr 0.001\
                        --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -fg
                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr 0.01\
                    --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -k ${filter_size} -fg
            done
            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr  0.01\
                    --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -k ${small_size} -o -fg
        done
    fi
    # batch_size, actor_lr, criticl_lr
    actor_lr=0.0001
    if [ ${ablation_exp} -gt 0 ]; then
        small_case="grid_small_dynamic/"
        small_size=4
        for name in 4; do
            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg3} -alr 0.0005 -clr 0.005\
                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 2000 -fg
            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg5} -alr 0.0001 -clr 0.001\ 
                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 50 -fg
            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg4} -alr 0.001 -clr 0.005\
                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -k 3 -u 2000 -fg
            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg6} -alr 0.0001 -clr 0.001\
                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 50 -k 3 -fg
            for pd in 0; do
                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr 0.005\
                        --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd}
                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr 0.01\
                    --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -k ${small_size}
            done
            # no incremental delay
            for pd in 1 10; do
                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr 0.005\
                        --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -fg
                python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr 0.01\
                    --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -k ${small_size} -fg
            done
            python main.py --data_folder ${small_case}${small_size}/${name}/ --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr 0.01\
            --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -k ${small_size} -o -fg
        done
    fi

fi

if [ ${large_exp} -gt 0 ]; then
    if [ ${full_exp} -gt 0 ]; then
        for name in 4; do
            python main.py --data_folder ${large_case}  --pricing_alg dummy \
            --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --seed ${SEEDS[${name}]}
            python main.py --data_folder ${large_case}  --pricing_alg equilibrium \
            --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --seed ${SEEDS[${name}]}
            for batch_size in 64; do
                for actor_lr in 0.00001 0.00005 0.0001 0.001; do
                    for critic_lr in 0.001 0.005 0.01; do
                        python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                            --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -f 10 -fg
                        python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg3} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -f 10 -u 2000
                        python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg5} -alr ${actor_lr} -clr ${critic_lr}\
                            --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 50 -f 10
                        for filter_size in 5; do
                            python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                            --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -k ${filter_size} -f 10 -fg
                            python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg4} -alr ${actor_lr} -clr ${critic_lr}\
                                --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -k ${filter_size} -f 10 -u 2000
                            python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg6} -alr ${actor_lr} -clr ${critic_lr}\
                            --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 50 -k ${filter_size} -f 10 -fg
                        done
                    done
                done
            done
        done
    fi


    if [ ${ablation_exp} -gt 0 ]; then
        actor_lr=0.0001
        critic_lr=0.005
        large_size=5
        for name in 4; do
            for batch_size in 64; do
                python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg3} -alr 0.0001 -clr 0.01\
                    --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -f 10 -u 2000 -fg
                python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg5} -alr 0.0001 -clr 0.001\
                    --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 50 -f 10 -fg
                python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg4} -alr 0.00001 -clr 0.005\
                    --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -k 5 -f 10 -u 2000 -fg
                python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg6} -alr 0.001 -clr 0.005\
                    --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -u 50 -k 5 -f 10 -fg
                
                # no forget
                for pd in 0; do
                    python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                            --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -f 10 
                    python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                            --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -k ${large_size} -f 10 
                done
                for pd in 1 10; do
                    python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg1} -alr ${actor_lr} -clr ${critic_lr}\
                            --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -f 10 -fg
                    python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                            --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd ${pd} -k ${large_size} -f 10-fg
                done
                python main.py --data_folder ${large_case} --pricing_alg ${pricing_alg2} -alr ${actor_lr} -clr ${critic_lr}\
                        --device ${device} --n_epochs ${EPOCHS} -m test -n ${name} --batch_size ${batch_size} --seed ${SEEDS[${name}]} -pd 0 -k ${large_size} -o -f 10 -fg
            done
        done
    fi
fi

