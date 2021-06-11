# Computationally Efficient Optimization of Plackett-Luce Ranking Models for Relevance and Fairness
This repository contains the code used for the experiments in "Computationally Efficient Optimization of Plackett-Luce Ranking Models for Relevance and Fairness" published at SIGIR 2021 ([preprint available](https://harrieo.github.io//publication/2021-plrank)).

Citation
--------

If you use this code to produce results for your scientific publication, or if you share a copy or fork, please refer to our SIGIR 2021 paper:
```
@inproceedings{oosterhuis2021plrank,
  Author = {Oosterhuis, Harrie},
  Booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR`21)},
  Organization = {ACM},
  Title = {Computationally Efficient Optimization of Plackett-Luce Ranking Models for Relevance and Fairness},
  Year = {2021}
}
```

License
-------

The contents of this repository are licensed under the [MIT license](LICENSE). If you modify its contents in any way, please link back to this repository.

Usage
-------

This code makes use of [Python 3](https://www.python.org/), the [numpy](https://numpy.org/) and the [tensorflow](https://www.tensorflow.org/) packages, make sure they are installed.

A file is required that explains the location and details of the LTR datasets available on the system, for the Yahoo! Webscope, MSLR-Web30k, and Istella datasets an example file is available. Copy the file:
```
cp example_datasets_info.txt local_dataset_info.txt
```
Open this copy and edit the paths to the folders where the train/test/vali files are placed.

Here are some command-line examples that illustrate how the results in the paper can be replicated.
First create a folder to store the resulting models:
```
mkdir local_output
```
To optimize NDCG use *run.py* with the *--loss* flag to indicate the loss to use (PL_rank_1/PL_rank_2/lambdaloss/pairwise/policygradient/placementpolicygradient); *--cutoff* indicates the top-k that is being optimized, e.g. 5 for NDCG@5; *--num_samples* the number of samples to use per gradient estimation (with *dynamic* for the dynamic strategy); *--dataset* indicates the dataset name, e.g. *Webscope_C14_Set1*.
The following command optimizes NDCG@5 with PL-Rank-2 and the dynamic sampling strategy on the Yahoo! dataset:
```
python3 run.py local_output/yahoo_ndcg5_dynamic_plrank2.txt --num_samples dynamic --loss PL_rank_2 --cutoff 5 --dataset Webscope_C14_Set1
```
To optimize the disparity metric for exposure fairness use *fairrun.py* this has the additional flag *--num_exposure_samples* for the number of samples to use to estimate exposure (this must always be a greater number than *--num_samples*).
The following command optimizes disparity with PL-Rank-2 and the dynamic sampling strategy on the Yahoo! dataset with 1000 samples for estimating exposure:
```
python3 fairrun.py local_output/yahoo_fairness_dynamic_plrank2.txt --num_samples dynamic --loss PL_rank_2 --cutoff 5 --num_exposure_samples 1000 --dataset Webscope_C14_Set1
```