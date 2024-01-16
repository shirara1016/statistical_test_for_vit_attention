# bash

for noise in iid corr; do
    for mode in image architecture; do
        for timer in 0 1; do
            python visualize/plot.py \
                --experiment null \
                --noise $noise \
                --mode $mode \
                --timer $timer \
                >> result.txt 2>&1
        done
    done
done

for noise in iid corr; do
    python visualize/plot.py \
        --experiment alter \
        --noise $noise \
        --mode $mode \
        --timer 0 \
        >> result.txt 2>&1
done

for noise in iid corr; do
    for mode in image architecture; do
        python visualize/plot.py \
            --experiment multi \
            --noise $noise \
            --mode $mode \
            >> result.txt 2>&1
    done
done
