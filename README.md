# Explainable Graph Nueral Architecture Search for Homophilic and Heterophilic Graphs
This is a an implementation of a graph NAS method with Monte-Carlo tree search.

## Setup
- python=3.9
- CUDA11.1
- pytorch=1.9.0
- torchvision=0.10.0

## Run
`python3 main.py -train_size $train_size -val_size $val_size -test_size $test_size -epoch $epoch -dataset_name $dataset_name`
- `$train_size` set train data size
- `$val_size` set validation data size
- `$test_size` set test data size
- `$epoch` set the number of epochs
- `$dataset_name` set the detaset name

ex) `python3 main.py -train_size 0.5 -val_size 0.25 -test_size 0.25 -epoch 500 -dataset_name cora`

CSV files are generated at result directory.

