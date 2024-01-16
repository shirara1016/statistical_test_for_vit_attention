# bash

for d in 64 32 16 8; do
    for noise in iid corr; do
        for seed in {0..9}; do
            python experiment/permute_experiment.py \
                --d $d \
                --noise $noise \
                --seed $seed \
                >> result.txt 2>&1
        done
    done
done

for architecture in huge large small; do
    for noise in iid corr; do
        for seed in {0..9}; do
            python experiment/permute_experiment.py \
                --architecture $architecture \
                --noise $noise \
                --seed $seed \
                >> result.txt 2>&1
        done
    done
done
