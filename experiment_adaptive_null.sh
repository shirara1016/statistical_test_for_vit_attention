# bash

for architecture in huge large small; do
    for noise in iid corr; do
        for seed in {0..9}; do
            python experiment/adaptive_experiment.py \
                --architecture $architecture \
                --noise $noise \
                --seed $seed \
                >> result.txt 2>&1
        done
    done
done

for d in 64 32 16 8; do
    for noise in iid corr; do
        for seed in {0..9}; do
            python experiment/adaptive_experiment.py \
                --d $d \
                --noise $noise \
                --seed $seed \
                >> result.txt 2>&1
        done
    done
done
