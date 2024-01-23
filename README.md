# Graph Neural Architecture Search for Homophilic and Heterophilic Graphs
This is an implementation of a graph NAS method with Monte-Carlo tree search.

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
- `$num_models` set the number of architectures
- `$search threshold` set the visit times to generate child nodes
- `$mcts_score_sqrt` set c of ucs
- `$eval_type` set the mode of search algorithm (max or avg)
- `$eval_mode` set the mode of evaluation (acc or auc)

ex) `python3 main.py -train_size 0.6 -val_size 0.2 -test_size 0.2 -epoch 500 -trial 5 -num_models 1000 -search_threshold 10 -mcts_score_sqrt 2 -eval_type avg -eval_mode acc -dataset_name cora`

CSV files are generated at results directory.

