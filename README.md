# HGRU4Rec
Code for our ACM RecSys 2017 paper "Personalizing Session-based Recommendation with Hierarchical Recurrent Neural Networks". 
See the paper: [https://arxiv.org/abs/1706.04148](https://arxiv.org/abs/1706.04148)

## Setup
This code is based of GRU4Rec ([https://github.com/hidasib/GRU4Rec](https://github.com/hidasib/GRU4Rec)).
As the original code, it is written in Python 3.4 and requires Theano 0.8.0+ to run efficiently on GPU.
In addition, this code uses H5Py and PyTables for efficient I/O operations.

We suggest to use `virtualenv` or `conda` (preferred) together with `requirements.txt` to set up a virtual environment before running the code.

## Experiments on the XING dataset
This repository comes with the code necessary to reproduce the experiments on the XING dataset.
This dataset was released to the participants of the 2016 Recsys Challenge.

1) Download the dataset (see [here](http://2016.recsyschallenge.com/), though it is no longer available.  See format in [this comment](https://github.com/mquad/hgru4rec/issues/1#issuecomment-381060517)). You will only need the file `interactions.csv`.

2) `cd data/xing`, then run `python build_dataset.py <path_to_interactions>` to build the dataset. It will be saved under `data/xing/dense/last-session-out/sessions.hdf`.

3) To run HGRU on this dataset, go to `scripts` folder.
Then run `sh xing_dense_small.sh` to execute _small_ HRNN networks, or run `sh xing_dense_large.sh` to execute _large_ HRNN networks. See the paper for further details (notice that we used random seeds in \{0..9\} in our experiments).

NOTE: These experiments run quite efficiently on CPU too (small networks train and evaluate in ~20 minutes on a 8-core Intel(R) Xeon(R) CPU E3-1246 v3 @ 3.50GHz).
