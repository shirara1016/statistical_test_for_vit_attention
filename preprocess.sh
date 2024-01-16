# bash

for i in {0..4}; do
    python visualize/preprocess.py \
        --num $i \
        >> result.txt 2>&1
done
