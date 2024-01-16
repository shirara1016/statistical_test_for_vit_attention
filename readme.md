# Statistical Test for Attention Map in Vision Transformer
This pacakge is the implementation of the paper "Statistical Test for Attention Map in Vision Transformer" for experiments.

## Installation & Requirements
This pacakage has the following dependencies:
- Python (version 3.10 or higher, we use 3.10.11)
- sicore (we use version 1.0.0)
- tensorflow (we use version 2.11.1)
- tqdm

Please install these dependencies by pip.
```
pip install sicore
pip install tensorflow
pip install tqdm
```

## Reproducibility

Since we have already got the results in advance, you can reproduce the figures by running following code. The results will be saved in "/image" folder.
```
sh plot.sh
```

To reproduce the results, please see the following instructions after installation step.
The results will be saved in "./results_{iid,corr,permute}" folder as pickle file.

For reproducing the Figures 4 and 5 (type I error rate).
The first line is for our proposed method (adaptive) and simultaneously for bonferroni correction and naive test.
The second line is for permutation test.
```
sh experiment_adaptve_null.sh
sh experiment_permute.sh
```

For reproducing the Figure 6 (power).
This is for our proposed method (adaptive) and simultaneously for bonferroni correction.
```
sh experiment_adaptive_power.sh
```

For reproducing the Figures 7 and 8 (computation time).
This is for other grid search options (fixed and combination).
```
sh experiment_time.sh
```

For visualization of the reproduced results.
```
sh preprocess.sh
sh plot.sh
```
