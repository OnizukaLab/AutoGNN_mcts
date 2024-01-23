import random
import copy
import numpy as np
from statistics import mean
import math
import yaml
import torch
import time

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

class ModelConfig:
    '''
    1.  gnn layer size
    2.  PreMLP
    3.  PreMLP JKnet
    4.  JKnet
    5.  1 layer's activation func
    6.  1 layer's attention
    7.  postMLP
    8.  pre MLP emb size
    9.  post MLP emb size
    10. 1 layer's emd size
    11. 2 layer's activation func
    12. 2 layer's attention
    13. 2 layer's emd size
    14. 3 layer's activation func
    15. 3 layer's attention
    16. 3 layer's emd size
    17. lr
    '''
    def __init__(self, params):
        self.params = params

    def get_params(self, fix_params):

        for i in range(len(fix_params), 17):
            fix_params.append(random.choice(self.get_choices(fix_params)))
        return fix_params

    def get_choices(self, fix_params):

        i = len(fix_params)
        if i == 0: # gnn layer size
            return self.params["gnn"]["layer_sizes"]
        elif i == 1: # PreMLP
            return self.params["preMLP"]["layer_sizes"]
        elif i == 2: # PreMLP JKnet
            if fix_params[1] == 0:
                return [0]
            else:
                return self.params["gnn"]["jknet_preMLP"]
        elif i == 3: # JKnet
            if fix_params[0] == 1 and fix_params[2] == 0:
                return ['none']
            else:
                return self.params["gnn"]["jumping_knowledges"]
        elif i == 4: # 1 layer's activation func
            return self.params["gnn"]["activation_functions"]
        elif i == 5: # 1 layer's attention
            return self.params["gnn"]["attention_mechanisms"]
        elif i == 6: # postMLP
            plist = copy.copy(self.params["postMLP"]["hidden_layers"])
            if (fix_params[3] != 'none' or (fix_params[0] == 1 and fix_params[4] != 'none')) and 0 in plist:
                plist.remove(0)
            return plist
        elif i == 7: # pre MLP emb size
            if fix_params[1] == 0:
                return [0]
            else:
                if fix_params[3] == "max":
                    new_list = set(self.params["preMLP"]["emb_sizes"]) & set(self.params["gnn"]["emb_sizes"])
                    return list(new_list)
                else:
                    return self.params["preMLP"]["emb_sizes"]
        elif i == 8: # post MLP emb size
            if fix_params[6] == 0:
                return [0]
            else:
                return self.params["postMLP"]["hidden_sizes"]
        elif i == 9: # 1 layer's emd size
            if fix_params[3] == "max" and fix_params[7] > 0:
                return [fix_params[7]]
            else:
                return self.params["gnn"]["emb_sizes"]
        elif i == 10: # 2 layer's activation func
            if fix_params[0] >= 2:
                if fix_params[6] == 0 and fix_params[0] == 2:
                    return ['none']
                else:
                    return self.params["gnn"]["activation_functions"]
            else:
                return ['']
        elif i == 11: # 2 layer's attention
            if fix_params[0] >= 2:
                return self.params["gnn"]["attention_mechanisms"]
            else:
                return ['']
        elif i == 12: # 2 layer's emd size
            if fix_params[3] == "max":
                return [fix_params[9]]
            else:
                return self.params["gnn"]["emb_sizes"]
        elif i == 13: # 3 layer's activation func
            if fix_params[0] >= 2:
                if fix_params[0] >= 3:
                    if fix_params[6] == 0 and fix_params[0] == 3:
                        return ['none']
                    else:
                        return self.params["gnn"]["activation_functions"]
                else:
                    return ['']
            else:
                return [0]
        elif i == 14: # 3 layer's attention
            if fix_params[0] >= 3:
                return self.params["gnn"]["attention_mechanisms"]
            else:
                return ['']
        elif i == 15: # 3 layer's emd size
            if fix_params[0] >= 3:
                if fix_params[3] == "max":
                    return [fix_params[9]]
                else:
                    return self.params["gnn"]["emb_sizes"]
            else:
                return [0]
        elif i == 16: # 3 learning rate
            return self.params["general"]["lr"]
        else:
            return None

    def convert_model_param(self, params):

        l = len(params)
        if l < 17:
            params.extend([ '' for _ in range(17 - l)])

        gnn_layer_size = params[0]

        if gnn_layer_size == 1:
            gnn_emb_size = (params[9],)
            activation_func = (params[4],)
            attention_mech = (params[5],)
        elif gnn_layer_size == 2:
            gnn_emb_size = (params[9], params[12])
            activation_func = (params[4], params[10])
            attention_mech = (params[5], params[11])
        elif gnn_layer_size == 3:
            gnn_emb_size = (params[9], params[12], params[15])
            activation_func = (params[4], params[10], params[13])
            attention_mech = (params[5], params[11], params[14])
        jumping_knowledge = params[3]
        preMLP_layer_sizes = params[1]
        preMLP_emb_sizes = params[7]
        postMLP_hidden_layers = params[6]
        postMLP_hidden_sizes = params[8]
        lr = params[16]
        jknet_preMLP = params[2]

        comb = [gnn_layer_size, gnn_emb_size, activation_func, attention_mech, jumping_knowledge, preMLP_layer_sizes, preMLP_emb_sizes, postMLP_hidden_layers, postMLP_hidden_sizes, lr, jknet_preMLP]
        return comb


class MctsNode:

    def __init__(self, config, runner, parent, choice, max_try_count=40, max_search_time=3600, expand_threshold=20, stepMode=True, mcts_score_sqrt = 2, eval_type = "avg"):

        self.config = config
        self.runner = runner
        self.parent = parent
        self.choice = choice
        self.try_count = 0
        self.child_try_count = 0
        self.auc_list = []
        self.exec_time_list = []
        self.run_param_list = []
        self.child_trees = []
        self.stepMode = stepMode

        if parent is not None:
            self.max_try_count = parent.max_try_count
            self.max_search_time = parent.max_search_time
            self.expand_threshold = parent.expand_threshold
            self.mcts_score_sqrt = parent.mcts_score_sqrt
            self.eval_type = parent.eval_type
        else:
            self.max_try_count = max_try_count
            self.max_search_time = max_search_time
            self.expand_threshold = expand_threshold
            self.mcts_score_sqrt = mcts_score_sqrt
            self.eval_type = eval_type

    def get_params(self):

        if self.parent is None:
            if self.choice is None:
                return []
            return [self.choice]
        else:
            p = self.parent.get_params()
            p.append(self.choice)
            return p

    def best_node(self, all_count):

        uct_list = [x.uct(all_count) for x in self.child_trees]
        maxIndex = random.choice([i for i, x in enumerate(uct_list) if x == max(uct_list)])
        node = self.child_trees[maxIndex]
        return node

    def real_best_node(self):

        auc_list = [x.auc() for x in self.child_trees]
        maxIndex = random.choice([i for i, x in enumerate(auc_list) if x == max(auc_list)])
        node = self.child_trees[maxIndex]
        return node

    def init_child_trees(self):

        model_params = self.get_params()
        choices = self.config.get_choices(model_params)
        if choices == None:
            return

        for choice in choices:
            node = MctsNode(self.config, self.runner, self, choice)
            self.child_trees.append(node)

    def search(self):

        search_start_time = time.time()

        if len(self.child_trees) == 0:
            self.init_child_trees()
        if len(self.child_trees) == 0:
            return

        for i in range(self.child_try_count, self.max_try_count):
            # UCT get
            node = self.best_node(i)
            node.play()
            self.child_try_count += 1

            search_time = time.time() - search_start_time

            if search_time > self.max_search_time:
                print('Search time exceed the limit. time:', search_time)
                break

        if self.stepMode == False:
            return

        node = self.real_best_node()
        node.search()

    def play(self):

        self.try_count += 1

        start_time = time.time()
        auc = -1
        if self.try_count > self.expand_threshold:

            auc = self.expand_play()
        if auc == -1:

            model_params = self.get_params()

            seed = len(model_params) * 100 + self.try_count
            fix_seed(seed)
            model_params = self.config.get_params(model_params)
            mode_prams_conv = self.config.convert_model_param(model_params)

            print('Start model run:', mode_prams_conv)

            model_train_start_time = time.time()
            try:
                model, e, auc = self.runner.run_model(mode_prams_conv)
            except MemoryError:
                print('Memory Error:', mode_prams_conv)
                auc = 0
            else:
                self.run_param_list.append([auc, mode_prams_conv, seed])
                print('End model run:', mode_prams_conv, 'max_val_AUC:', auc, 'model_run_count:', self.runner.model_run_count)

                model_train_time = time.time() - model_train_start_time

                params_num = params_count(model)

                self.runner.writer.writerow([self.runner.trial, self.runner.model_run_count, mode_prams_conv[0], mode_prams_conv[1],
                                             mode_prams_conv[2], mode_prams_conv[3], mode_prams_conv[4], mode_prams_conv[5], mode_prams_conv[6], mode_prams_conv[7], mode_prams_conv[8],
                                             mode_prams_conv[9], mode_prams_conv[10], auc, model_train_time, params_num])

        exec_time = time.time() - start_time

        self.auc_list.append(auc)
        self.exec_time_list.append(exec_time)
        return auc


    def expand_play(self):

        if len(self.child_trees) == 0:
            self.init_child_trees()
        if len(self.child_trees) == 0:
            return -1

        node = self.best_node(self.child_try_count)
        auc = node.play()
        self.child_try_count += 1
        return auc

    def auc(self):

        # mean or max
        if self.eval_type == "avg":
            return mean(self.auc_list)
        else:
            return max(self.auc_list)

    def uct(self, all_count):

        if all_count == 0:
            return np.inf
        if len(self.auc_list) == 0:
            return np.inf

        c = math.sqrt(self.mcts_score_sqrt)
        
        # mean or max
        if self.eval_type == "avg":
            value = mean(self.auc_list) + c * math.sqrt(math.log(all_count)/self.try_count)
        else:
            value = max(self.auc_list) + c * math.sqrt(math.log(all_count)/self.try_count)

        return value

    def get_deepest_node(self):

        if len(self.get_params()) == 17:
            return self

        if len(self.child_trees) == 0:
            return None
        for child in self.child_trees:
            e = child.get_deepest_node()
            if e is not None:
                return e
        return None

    def collect_run_param_list(self):

        param_list = []
        for child in self.child_trees:
            plist = child.collect_run_param_list()
            param_list.extend(plist)
        param_list.extend(self.run_param_list)
        return param_list

    def select_max_auc_param_list(self):

        plist = self.collect_run_param_list()
        max_auc = max([ x[0] for x in plist ])
        param_list = [ x for x in plist if x[0] == max_auc ]
        return param_list[0]

    def collect_auc_param_list(self):

        param_list = []
        if len(self.auc_list) > 0:
            params = self.get_params()
            offset = len(params)
            param_list.append([self.auc(), params, offset, len(self.auc_list), np.mean(self.exec_time_list)])
        for child in self.child_trees:
            plist = child.collect_auc_param_list()
            param_list.extend(plist)
        return param_list

    def get_auc_param_list(self):

        plist = self.collect_auc_param_list()
        deep_list = [ x for x in plist if len(x[1]) == 17 ]
        deep_list = [ x for x in deep_list if x[0] == max([ x[0] for x in deep_list ]) ]

        if len(deep_list) == 0:
            return plist, None
        return plist, deep_list[0]





class dummy_runner:
    def run_model(self, comb):
        return None, 0, random.random()

if __name__ == "__main__":
        with open('config.yml', 'r') as yml:
            params = yaml.safe_load(yml)
        config = ModelConfig(params)

        root = MctsNode(config, dummy_runner(), None, None, 1000, 10, False)
        root.search()
        # deepest_node = root.get_deepest_node()

        auc_param_list, deep_node = root.get_auc_param_list()
        for x in auc_param_list:
            # x[2]:offset, x[0]:auc, x[3]:len(auc_list), x[1]:params, x[4]:mean(exec_time_list)
            print("MCTS Node : ", "  " * x[2], x[0], x[3], x[1], x[4])

        auc, comb, seed = root.select_max_auc_param_list()
        print("Best Node: ", comb, auc)
        # print("Best Node: ", deep_node[0], deep_node[1])

        print("Model Count : ", len(root.collect_run_param_list()))
