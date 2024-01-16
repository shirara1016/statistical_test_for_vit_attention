# bash

for architecture in huge large small; do
    for noise in iid corr; do
        for seed in {0..1}; do
            python experiment/deterministic_experiment.py \
                --architecture $architecture \
                --noise $noise \
                --seed $seed \
                --mode fine \
                >> result.txt 2>&1
        done
    done
done

for d in 64 32 16 8; do
    for noise in iid corr; do
        for seed in {0..1}; do
            python experiment/deterministic_experiment.py \
                --d $d \
                --noise $noise \
                --seed $seed \
                --mode fine \
                >> result.txt 2>&1
        done
    done
done

for architecture in huge large small; do
    for noise in iid corr; do
        for seed in {0..1}; do
            python experiment/deterministic_experiment.py \
                --architecture $architecture \
                --noise $noise \
                --seed $seed \
                --mode combi \
                >> result.txt 2>&1
        done
    done
done

for d in 64 32 16 8; do
    for noise in iid corr; do
        for seed in {0..1}; do
            python experiment/deterministic_experiment.py \
                --d $d \
                --noise $noise \
                --seed $seed \
                --mode combi \
                >> result.txt 2>&1
        done
    done
done
