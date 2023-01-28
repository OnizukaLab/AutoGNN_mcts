import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge, GATConv
from torch_sparse import SparseTensor
import torch_sparse
import time


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, mechanism="gat"):
        super(GraphAttentionLayer, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.mechanism = mechanism

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))).to(self.device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1))).to(self.device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh, adj).to(self.device)

        attention = e
        row_s = attention.storage.row()
        v = attention.storage.value()
        rowmax = torch_sparse.reduce.max(attention, dim=-1)
        v = v - rowmax[row_s]
        attention = torch_sparse.SparseTensor(row=attention.storage.row(), col=attention.storage.col(), value=torch.exp(v))
        rowsum = torch_sparse.reduce.sum(attention, dim=-1)
        attention = attention / rowsum.view(-1, 1)
        h_prime = torch_sparse.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh, adj):
        if self.mechanism == "gat":
            Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
            Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
            row_n = adj.storage.row()
            col_n = adj.storage.col()
            value_n = self.leakyrelu(Wh1.T[0][row_n] + Wh2.T[0][col_n])
            e = torch_sparse.SparseTensor(row=row_n, col=col_n, value=value_n)
            return e
        elif self.mechanism == "constant":
            e = adj
            return e
        elif self.mechanism == "gcn":
            deg = torch_sparse.reduce.sum(adj, dim=1)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.)
            adj = torch_sparse.mul(adj, deg_inv_sqrt.view(-1, 1))
            adj = torch_sparse.mul(adj, deg_inv_sqrt.view(1, -1))
            e = adj
            return e

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"



class GNN(nn.Module):
    def __init__(self, input_size, layer_size, emb_size, activation_func, attention_mech, jumping_knowledge, jknet_preMLP, last_emb=-1, dropout=0.0, alpha=0.2, nheads=1):
        super(GNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layer_size = layer_size
        self.activation_func = activation_func
        self.attention_mech = attention_mech
        self.jumping_knowledge = jumping_knowledge
        self.jknet_preMLP = jknet_preMLP
        self.dropout= dropout

        self.bns = nn.ModuleList()
        self.attentions = []
        for layer in range(self.layer_size):
            in_size = input_size if layer == 0 else emb_size[layer-1] * nheads
            if layer == self.layer_size-1 and last_emb != -1:
                out_size = last_emb
            else:
                out_size = emb_size[layer]
            p_nheads = 1 if layer == self.layer_size-1 else nheads
            if self.attention_mech[layer] == "constant":
                attentions = [GraphAttentionLayer(in_size, out_size, dropout, alpha, True, "{}".format(self.attention_mech[layer])) for _ in range(p_nheads)]
            if self.attention_mech[layer] == "gat":
                attentions = [GATConv(in_size, out_size, heads=p_nheads, concat=True, add_self_loops=True) for _ in range(1)]
            if self.attention_mech[layer] == "gcn":
                attentions = [GCNConv(in_size, out_size, cached=False, normalize=True, add_self_loops=True) for _ in range(p_nheads)]
            self.attentions.append(attentions)
            for i, attention in enumerate(attentions):
                self.add_module("attention_{}_{}".format(layer, i), attention)
            self.bns.append(nn.BatchNorm1d(emb_size[layer] * nheads))
        if self.jumping_knowledge != "none":
            self.jk = JumpingKnowledge(mode=jumping_knowledge)

    def forward(self, x, edge_index):
        row, col = edge_index.to(self.device)
        value = torch.ones(edge_index.shape[1]).to(self.device)
        adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(x.shape[0], x.shape[0]))
        if not adj.has_value():
            adj = adj.fill_value(0.)
        adj = SparseTensor.fill_diag(adj, 1.)
        xs = []
        if self.jknet_preMLP:
            xs.append(x)
        for layer in range(self.layer_size):
            x = torch.cat([att(x, adj if isinstance(att, GraphAttentionLayer) else edge_index) for att in self.attentions[layer]], dim=1)
            if layer != self.layer_size-1:
                x = self.bns[layer](x)
            x = activation_function(x, self.activation_func[layer])
            xs.append(x)
            if layer != self.layer_size-1:
                x = F.dropout(x, self.dropout, training=self.training)
        if self.jumping_knowledge != "none":
            x = self.jk(xs)
        return x


class PreMLP(nn.Module):
    def __init__(self, input_size, emb_size):
        super(PreMLP, self).__init__()
        self.fc_mid = nn.Linear(input_size, emb_size)

    def forward(self, x):
        return F.relu(self.fc_mid(x))

class MLP(nn.Module):
    def __init__(self, hidden_layer, emb_size, out_size, hidden_size=64):
        super(MLP, self).__init__()
        self.hidden_layer = hidden_layer
        self.fc_in = nn.Linear(emb_size, hidden_size)
        self.fc_mid = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        for layer in range(self.hidden_layer-1):
            x = F.relu(self.fc_mid(x))
        return self.fc_out(x)



def activation_function(x, func):
    if func == "relu":
        x = F.relu(x)
    elif func == "sigmoid":
        x = torch.sigmoid(x)
    elif func == "tanh":
        x = torch.tanh(x)
    else:
        pass
    return x


class NodePredictionModel(torch.nn.Module):
    def __init__(self, input_size, num_classes,
                 gnn_layer_size, gnn_emb_size, activation_func, attention_mech, jumping_knowledge,
                 preMLP_layer_sizes, preMLP_emb_sizes, postMLP_hidden_layers, postMLP_hidden_sizes, lr, jknet_preMLP,
                 dropout=0.2):
        super(NodePredictionModel, self).__init__()

        self.preMLP_layer_sizes = preMLP_layer_sizes
        self.mlp_hidden_layer = postMLP_hidden_layers
        self.lr = lr

        jknet_preMLP_param = (preMLP_layer_sizes != 0 and jknet_preMLP)

        if preMLP_layer_sizes == 0:
            gnn_input_size = input_size
        else:
            self.preMLP = PreMLP(input_size, preMLP_emb_sizes)
            gnn_input_size = preMLP_emb_sizes

        if postMLP_hidden_layers == 0:
            self.GNN = GNN(input_size=gnn_input_size, layer_size=gnn_layer_size, emb_size=gnn_emb_size,
                           activation_func=activation_func, attention_mech=attention_mech,
                           jumping_knowledge=jumping_knowledge, jknet_preMLP=jknet_preMLP_param,
                           last_emb=num_classes)
        else:
            self.GNN = GNN(input_size=gnn_input_size, layer_size=gnn_layer_size, emb_size=gnn_emb_size,
                           activation_func=activation_func, attention_mech=attention_mech,
                           jumping_knowledge=jumping_knowledge, jknet_preMLP=jknet_preMLP_param)

        if jumping_knowledge == "cat":
            if jknet_preMLP_param:
                emb_size = sum(gnn_emb_size) + preMLP_emb_sizes
            else:
                emb_size = sum(gnn_emb_size)
        else:
            emb_size = gnn_emb_size[-1]

        if postMLP_hidden_layers != 0:
            self.MLP = MLP(hidden_layer=postMLP_hidden_layers, emb_size=emb_size, out_size=num_classes, hidden_size=postMLP_hidden_sizes)

    def forward(self, data, nodes):
        x = data.x
        if self.preMLP_layer_sizes != 0:
            x = self.preMLP(x)
        start_GNN_pred = time.time()
        self.emb = self.GNN(x, data.edge_index)
        end_GNN_pred = time.time()
        node_emb = [self.emb[i] for i in nodes]
        node_emb = torch.stack(node_emb)
        if self.mlp_hidden_layer == 0:
            pred = node_emb
        else:
            pred = self.MLP(node_emb)
        end_MLP_pred = time.time()
        return pred, end_GNN_pred - start_GNN_pred, end_MLP_pred - end_GNN_pred

    def getLr(self):
        return self.lr

def selectModel(comb, feature_size, num_classes):
    gnn_layer_size, gnn_emb_size, activation_func, attention_mech, jumping_knowledge, preMLP_layer_sizes, preMLP_emb_sizes, postMLP_hidden_layers, postMLP_hidden_sizes, lr, jknet_preMLP = comb
    return NodePredictionModel(feature_size, num_classes, gnn_layer_size, gnn_emb_size, activation_func, attention_mech, jumping_knowledge, preMLP_layer_sizes, preMLP_emb_sizes, postMLP_hidden_layers, postMLP_hidden_sizes, lr, jknet_preMLP)


