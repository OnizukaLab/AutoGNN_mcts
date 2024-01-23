import argparse
import csv
import os
import random
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from torch_geometric.datasets import Amazon, WikiCS, Actor, Coauthor
from torch_geometric.utils import to_undirected

from DatasetLoader.DatasetCiteseer import DatasetCiteseer
from DatasetLoader.DatasetCora import DatasetCora
from DatasetLoader.DatasetGeneral import DatasetGeneral
from DatasetLoader.DatasetLINKX import DatasetLINKX
from DatasetLoader.DatasetPubMed import DatasetPubMed
from DatasetLoader.DatasetWebkb import DatasetWebkb
from DatasetLoader.DatasetWikipediaNetwork import DatasetWikipediaNetwork
from mcts import MctsNode
from mcts import ModelConfig
from models import selectModel

import subprocess
import platform
from datetime import datetime


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# model params
def params_count(model):
    return sum([p.numel() for p in model.parameters()])

# GPU info
DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [line.strip() for line in lines if line.strip() != '']

    return [{k: v for k, v in zip(keys, line.split(', '))} for line in lines]

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def make_edges_for_gcn(dataset):

    return to_undirected(dataset.data.edge_index)

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

class ModelRunner:

    def __init__(self, feature_size, num_classes, device, epoch, model_path, writer, trial):
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.device = device
        self.epoch = epoch
        self.model_path = model_path
        self.writer = writer
        self.trial = trial
        self.model_run_count = 0
        self.model_run_time = 0

    def set_dataset(self, train_data, train_node_list, val_data, val_node_list, test_data, test_node_list):
        self.train_data = train_data
        self.train_node_list = train_node_list
        self.val_data = val_data
        self.val_node_list = val_node_list
        self.test_data = test_data
        self.test_node_list = test_node_list

    def set_answer(self, train_ans, train_ans_bin, val_ans, val_ans_bin, test_ans, test_ans_bin):
        self.train_ans = train_ans
        self.train_ans_bin = train_ans_bin
        self.val_ans = val_ans
        self.val_ans_bin = val_ans_bin
        self.test_ans = test_ans
        self.test_ans_bin = test_ans_bin

    def run_model(self, comb):

        start_time = time.time()

        feature_size = self.feature_size
        num_classes = self.num_classes
        device = self.device
        epoch = self.epoch
        model_path = self.model_path

        train_data = self.train_data
        train_node_list = self.train_node_list
        val_data = self.val_data
        val_node_list = self.val_node_list
        train_ans = self.train_ans
        train_ans_bin = self.train_ans_bin
        val_ans = self.val_ans
        val_ans_bin = self.val_ans_bin

        model = selectModel(comb, feature_size, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), model.getLr(), weight_decay=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        losses = []
        max_val_auc = 0
        early_stopping_count = 0

        for e in range(epoch):
            model.train()
            optimizer.zero_grad()
            train_pred, train_gnn_time, train_mlp_time = model(train_data, train_node_list)
            loss = loss_fn(train_pred, train_ans)
            losses.append(loss)
            loss.backward()
            optimizer.step()

            if (e + 1) % 1 == 0:
                if eval_mode == 'auc':
                    train_AUC = roc_auc_score(train_ans_bin, train_pred.detach().cpu().numpy())
                else:
                    train_AUC = eval_acc(train_ans.view(-1, 1), train_pred)

                model.eval()
                val_pred, val_gnn_time, val_mlp_time = model(val_data, val_node_list)
                val_loss = loss_fn(val_pred, val_ans)

                if eval_mode == 'auc':
                    val_AUC = roc_auc_score(val_ans_bin, val_pred.detach().cpu().numpy())
                else:
                    val_AUC = eval_acc(val_ans.view(-1, 1), val_pred)

                if (e + 1) % 10 == 0:
                    if eval_mode == 'auc':
                        print("epoch{}".format(e + 1), "train loss : {}".format(loss), "val loss : {}".format(val_loss),
                              "train AUC : {}".format(train_AUC), "val AUC : {}".format(val_AUC))
                    else:
                        print("epoch{}".format(e + 1), "train loss : {}".format(loss), "val loss : {}".format(val_loss),
                              "train ACC : {}".format(train_AUC), "val ACC : {}".format(val_AUC))

                # early stopping
                if max_val_auc < val_AUC:
                    max_val_auc = val_AUC
                    early_stopping_count = 0
                    torch.save(model.state_dict(), model_path)
                else:
                    early_stopping_count += 1

                if early_stopping_count == patience:
                    print("early stopping {}epoch".format(e + 1))
                    break

        self.model_run_count += 1
        self.model_run_time += time.time() - start_time

        return model, e, max_val_auc


if __name__ == "__main__":
    import platform
    pf = platform.system()
    if pf != 'Windows':
        import sys
        import resource

        sys.setrecursionlimit(2000)
        resource.setrlimit(resource.RLIMIT_STACK, (-1, -1))

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_name', '--dataset_name', type=str)
    parser.add_argument('-train_size', '--train_size', default=0.6, type=float)
    parser.add_argument('-val_size', '--val_size', default=0.2, type=float)
    parser.add_argument('-test_size', '--test_size', default=0.2, type=float)
    parser.add_argument('-epoch', '--epoch', default=500, type=int)
    parser.add_argument('-patience', '--patience', default=10, type=int)
    parser.add_argument('-dropout', '--dropout', default=0.2, type=float)
    parser.add_argument('-trial', '--trial', default=5, type=int)
    parser.add_argument('-datasave', action='store_true')
    parser.add_argument('-eval_mode', '--eval_mode', type=str, help='auc or acc')
    
    parser.add_argument('-num_models', '--num_models', default=1000, type=int)
    parser.add_argument('-search_threshold', '--search_threshold', default=10, type=int)
    parser.add_argument('-mcts_score_sqrt', '--mcts_score_sqrt', default=2, type=int)
    parser.add_argument('-eval_type', '--eval_type', default='avg', type=str)
    parser.add_argument('-max_search_time', '--max_search_time', default=172800, type=int)
    
    args = parser.parse_args()

    dataset_name = args.dataset_name
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    epoch = args.epoch
    patience = args.patience
    dropout = args.dropout
    trial = args.trial
    datasave = args.datasave
    eval_mode = args.eval_mode
    
    num_models = args.num_models
    search_threshold = args.search_threshold
    mcts_score_sqrt = args.mcts_score_sqrt
    eval_type = args.eval_type
    max_search_time = args.max_search_time

    fix_seed(42)

    # data load
    if os.path.isfile("saved_data/{}/x.pt".format(dataset_name)):
        features = torch.load("saved_data/{}/x.pt".format(dataset_name))
        labels = torch.load("saved_data/{}/y.pt".format(dataset_name))
        edges = torch.load("saved_data/{}/edge_index.pt".format(dataset_name))
        dataset = DatasetGeneral(dataset_name, features, labels, edges)
    elif datasave:
        if dataset_name == "amazon_computer":
            dataset = Amazon(root='/tmp/{}'.format(dataset_name), name='computers')
        elif dataset_name == "amazon_photo":
            dataset = Amazon(root='/tmp/{}'.format(dataset_name), name='photo')
        elif dataset_name == "Coauthor_CS":
            dataset = Coauthor(root='/tmp/{}'.format(dataset_name), name='cs')
        elif dataset_name == "WikiCS":
            dataset = WikiCS(root='/tmp/{}'.format(dataset_name))
        elif dataset_name == "cora":
            dataset = DatasetCora(root ='/tmp/{}'.format(dataset_name))
        elif dataset_name == "CiteSeer":
            dataset = DatasetCiteseer(root ='/tmp/{}'.format(dataset_name))
        elif dataset_name == "cornell" or dataset_name =="texas" or  dataset_name == "washington" or  dataset_name == "wisconsin":
            dataset = DatasetWebkb(root ='/tmp/WebKB_{}'.format(dataset_name), name=dataset_name)
        elif dataset_name == "pubmed":
            dataset = DatasetPubMed(root='/tmp/{}'.format(dataset_name))
        elif dataset_name == "actor":
            dataset = Actor(root='/tmp/{}'.format(dataset_name))
        elif dataset_name == "chameleon" or dataset_name == "squirrel":
            dataset = DatasetWikipediaNetwork(root='/tmp/WikipediaNetwork_{}'.format(dataset_name), name=dataset_name)
        elif dataset_name == "penn94" or dataset_name== "genius":
            dataset = DatasetLINKX(root='/tmp/LINKX_{}'.format(dataset_name), name=dataset_name)
        else:
            print("not implemented")
            exit()
        os.makedirs("saved_data/{}".format(dataset_name), exist_ok=True)
        torch.save(dataset.data.x, "saved_data/{}/x.pt".format(dataset_name))
        torch.save(dataset.data.y, "saved_data/{}/y.pt".format(dataset_name))
        torch.save(dataset.data.edge_index, "saved_data/{}/edge_index.pt".format(dataset_name))
    else:
        print("error datafile not exist")
        exit()
    print("data loaded:", get_time())

    feature_size = dataset.num_node_features

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = dataset.num_classes
    classes_array = [ i for i in range(num_classes)]

    num_nodes = dataset.data.num_nodes
    node_list = list(range(0, num_nodes))

    if not os.path.exists('./np_save/{}/train_node_list_{}.npy'.format(dataset_name, 1)):
        for i in range(trial):
            random.shuffle(node_list)

            # train, val, test
            train_node_list = node_list[:int(len(node_list)*train_size)]
            val_node_list = node_list[int(len(node_list)*train_size):int(len(node_list)*(train_size+val_size))]
            test_node_list = node_list[int(len(node_list)*(train_size+val_size)):]

            arr_train_node_list = np.array(train_node_list)
            arr_val_node_list = np.array(val_node_list)
            arr_test_node_list = np.array(test_node_list)

            os.makedirs("np_save/{}".format(dataset_name), exist_ok=True)
            np.save('./np_save/{}/train_node_list_{}'.format(dataset_name, i+1), arr_train_node_list)
            np.save('./np_save/{}/val_node_list_{}'.format(dataset_name, i+1), arr_val_node_list)
            np.save('./np_save/{}/test_node_list_{}'.format(dataset_name, i+1), arr_test_node_list)

    y = np.array(dataset.data.y).tolist()

    if dataset_name == "penn94":
        with open('config_penn94.yml', 'r') as yml:
            params = yaml.safe_load(yml)
    else:
        with open('config.yml', 'r') as yml:
            params = yaml.safe_load(yml)

    config = ModelConfig(params)

    dataset.data.edge_index = make_edges_for_gcn(dataset)

    train_data = dataset[0].to(device)
    val_data = train_data
    test_data = train_data

    print("learning")

    model_path = "model/{}_best_model.pth".format(dataset_name)

    results = []

    f = open('results/ModelOrder_{}_{}_{}.csv'.format(dataset_name, eval_type, eval_mode), 'a', newline="")
    writer = csv.writer(f)
    writer.writerow(["dataset_name", "eval_type"])
    writer.writerow([dataset_name, eval_type])
    if eval_mode == 'auc':
        writer.writerow(
                ["trial", "model_order", "gnn_layer_size", "gnn_emb_size", "activation_func", "attention_mech",
                 "jumping_knowledge", "preMLP_layer_size", "preMLP_emb_size", "postMLP_hidden_layer",
                 "postMLP_hidden_size", "lr", "jknet_preMLP", "val_auc", "model_train_time", "params_num"])
    else:
        writer.writerow(
                ["trial", "model_order", "gnn_layer_size", "gnn_emb_size", "activation_func", "attention_mech",
                 "jumping_knowledge", "preMLP_layer_size", "preMLP_emb_size", "postMLP_hidden_layer",
                 "postMLP_hidden_size", "lr", "jknet_preMLP", "val_acc", "model_train_time", "params_num"])

    f2 = open('results/MCTS_NodeInfo_{}_{}_{}.csv'.format(dataset_name, eval_type, eval_mode), 'a', newline="")
    writer2 = csv.writer(f2)
    if eval_mode == 'auc':
        writer2.writerow(["", "trial", "model_params", "trial_count", "auc_mean", "exec_time_mean"])
    else:
        writer2.writerow(["", "trial", "model_params", "trial_count", "acc_mean", "exec_time_mean"])

    for t in range(trial):
        print("trial{}:".format(t+1), get_time())

        arr_train_node_list = np.load('./np_save/{}/train_node_list_{}.npy'.format(dataset_name, t+1))
        arr_val_node_list = np.load('./np_save/{}/val_node_list_{}.npy'.format(dataset_name, t+1))
        arr_test_node_list = np.load('./np_save/{}/test_node_list_{}.npy'.format(dataset_name, t+1))

        train_node_list = arr_train_node_list.tolist()
        val_node_list = arr_val_node_list.tolist()
        test_node_list = arr_test_node_list.tolist()

        train_ans = torch.tensor([int(y[i]) for i in arr_train_node_list], dtype=torch.long).to(device)
        val_ans = torch.tensor([int(y[i]) for i in arr_val_node_list], dtype=torch.long).to(device)
        test_ans = torch.tensor([int(y[i]) for i in arr_test_node_list], dtype=torch.long).to(device)

        if num_classes > 2:
            train_ans_bin = label_binarize(train_ans.detach().cpu().numpy(), classes=classes_array)
            val_ans_bin = label_binarize(val_ans.detach().cpu().numpy(), classes=classes_array)
            test_ans_bin = label_binarize(test_ans.detach().cpu().numpy(), classes=classes_array)
        else:
            train_ans_bin = label_binarize(train_ans.detach().cpu().numpy(), classes=[0, 1, 1])[:, 0:2]
            val_ans_bin = label_binarize(val_ans.detach().cpu().numpy(), classes=[0, 1, 1])[:, 0:2]
            test_ans_bin = label_binarize(test_ans.detach().cpu().numpy(), classes=[0, 1, 1])[:, 0:2]

        start_train = time.time()

        runner = ModelRunner(feature_size, num_classes, device, epoch, model_path, writer, t+1)
        runner.set_dataset(train_data, train_node_list, val_data, val_node_list, test_data, test_node_list)
        runner.set_answer(train_ans, train_ans_bin, val_ans, val_ans_bin, test_ans, test_ans_bin)

        root = MctsNode(config, runner, None, None, num_models, max_search_time, search_threshold, False, mcts_score_sqrt, eval_type)

        root.search()

        search_total_time = time.time() - start_train
        model_run_count = runner.model_run_count
        model_run_time = runner.model_run_time
        search_algorithm_time = search_total_time - model_run_time

        auc, comb, seed = root.select_max_auc_param_list()
        print("MCTS Best Model: ", auc, comb, seed)

        fix_seed(seed)

        start_eval = time.time()

        model, e, auc = runner.run_model(comb)
        print("MCTS Best Model ReRun: ", comb)

        params_num = params_count(model)

        auc_param_list, deep_node = root.get_auc_param_list()

        for x in auc_param_list:
            # x[0]:auc, x[1]:params, x[2]:offset, x[3]:len(auc_list), x[4]:mean(exec_time_list)
            writer2.writerow(["MCTS Node : ", t+1, x[1], x[3], x[0], x[4]])
        print("Best Model: ", comb)

        eval_time = time.time() - start_eval

        model.load_state_dict(torch.load(model_path)) #load best model
        model.eval()
        test_pred, test_gnn_time, test_mlp_time = model(test_data, test_node_list)
        train_pred, train_best_gnn_time, train_best_mlp_time = model(train_data, train_node_list)
        val_pred, val_best_gnn_time, val_best_mlp_time = model(val_data, val_node_list)

        if eval_mode == 'auc':
            train_AUC = roc_auc_score(train_ans_bin, train_pred.detach().cpu().numpy())
            val_AUC = roc_auc_score(val_ans_bin, val_pred.detach().cpu().numpy())
            test_AUC = roc_auc_score(test_ans_bin, test_pred.detach().cpu().numpy())
        else:
            train_ans2 = train_ans.view(-1, 1)
            val_ans2 = val_ans.view(-1, 1)
            test_ans2 = test_ans.view(-1, 1)

            train_AUC = eval_acc(train_ans2, train_pred)
            val_AUC = eval_acc(val_ans2, val_pred)
            test_AUC = eval_acc(test_ans2, test_pred)

        if e+1 == epoch:
            best_epoch = e+1
        else:
            best_epoch = e+1-patience

        if eval_mode == 'auc':
            print("best_epoch{}".format(best_epoch), "test AUC : {}".format(test_AUC), "train AUC : {}".format(train_AUC), "val AUC : {}".format(val_AUC))
        else:
            print("best_epoch{}".format(best_epoch), "test ACC : {}".format(test_AUC), "train ACC : {}".format(train_AUC), "val ACC : {}".format(val_AUC))

        model_param = "gnn_layer_size: {}".format(comb[0]),\
                      "gnn_emb_size: {}".format(comb[1]), \
                      "activation_func: {}".format(comb[2]),\
                      "attention_mech: {}".format(comb[3]),\
                      "jumping_knowledge: {}".format(comb[4]),\
                      "preMLP_layer_sizes: {}".format(comb[5]),\
                      "preMLP_emb_sizes: {}".format(comb[6]),\
                      "postMLP_hidden_layers: {}".format(comb[7]),\
                      "postMLP_hidden_sizes: {}".format(comb[8]),\
                      "lr: {}".format(comb[9]),\
                      "jknet_preMLP: {}".format(comb[10])

        pf = platform.system()
        gpu_usage = 0
        if pf != 'Windows':
            gpu_usage = float(get_gpu_info()[0]['memory.used'])

        results.append([t + 1, model_param, best_epoch, train_AUC, val_AUC, test_AUC,
                        search_total_time, model_run_count, model_run_time, search_algorithm_time,
                        test_gnn_time, test_mlp_time, len(test_node_list), params_num, gpu_usage])

    f.close()

    mean_best_epoc = np.mean([row[2] for row in results])
    mean_train = np.mean([row[3] for row in results])
    mean_val = np.mean([row[4] for row in results])
    mean_test = np.mean([row[5] for row in results])
    variance_train = np.var([row[3] for row in results])
    variance_val = np.var([row[4] for row in results])
    variance_test = np.var([row[5] for row in results])

    mean_search_total_time = np.mean([row[6] for row in results])
    mean_model_run_count = np.mean([row[7] for row in results])
    mean_model_run_time = np.mean([row[8] for row in results])
    mean_search_algorithm_time = np.mean([row[9] for row in results])

    mean_gnn_time = np.mean([row[10] for row in results])
    mean_mlp_time = np.mean([row[11] for row in results])
    mean_mlp_time_per_node = np.mean([row[11] / row[12] for row in results])

    mean_params_num = np.mean([row[13] for row in results])
    mean_gpu_usage = np.mean([row[14] for row in results])

    statistics = ["", "", mean_best_epoc, mean_train, variance_train, mean_val, variance_val, mean_test, variance_test,
                  mean_search_total_time, mean_model_run_count, mean_model_run_time, mean_search_algorithm_time,
                  mean_gnn_time, mean_mlp_time, mean_mlp_time_per_node, mean_params_num, mean_gpu_usage]

    with open('results/{}_{}_{}.csv'.format(dataset_name, eval_type, eval_mode), 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_name", "eval_type"])
        writer.writerow([dataset_name, eval_type])

        if eval_mode == 'auc':
            writer.writerow(["trial", "model_param", "best_epoch", "train_AUC", "val_AUC", "test_AUC",
                             "search_total_time", "model_run_count", "model_run_time", "search_algorithm_time",
                             "gnn_time", "mlp_time", "test_node_list", "params_num", "gpu_usage"])
        else:
            writer.writerow(["trial", "model_param", "best_epoch", "train_ACC", "val_ACC", "test_ACC",
                             "search_total_time", "model_run_count", "model_run_time", "search_algorithm_time",
                             "gnn_time", "mlp_time", "test_node_list", "params_num", "gpu_usage"])

        writer.writerows(results)

        if eval_mode == 'auc':
            writer.writerow(
                ["mean", "", "best_epoc_mean", "train_AUC_mean", "train_AUC_variance", "val_AUC_mean", "val_AUC_variance", "test_AUC_mean", "test_AUC_variance",
                 "search_total_time_mean", "model_run_count_mean", "model_run_time_mean", "search_algorithm_time_mean",
                 "gnn_time_mean", "mlp_time_mean", "mlp_time_mean_per_node", "params_num_mean", "gpu_usage_mean"])
        else:
            writer.writerow(
                ["mean", "", "best_epoc_mean", "train_ACC_mean", "train_ACC_variance", "val_ACC_mean",
                 "val_ACC_variance", "test_ACC_mean", "test_ACC_variance",
                 "search_total_time_mean", "model_run_count_mean", "model_run_time_mean", "search_algorithm_time_mean",
                 "gnn_time_mean", "mlp_time_mean", "mlp_time_mean_per_node", "params_num_mean", "gpu_usage_mean"])

        writer.writerow(statistics)
        writer.writerow("")

    print("Finish: ", get_time())
