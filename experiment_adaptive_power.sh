# bash

for signal in 4.0 3.0 2.0 1.0; do
    for noise in iid corr; do
        for seed in {0..9}; do
            python experiment/adaptive_experiment.py \
                --signal $signal \
                --noise $noise \
                --seed $seed \
                >> result.txt 2>&1
        done
    done
done
