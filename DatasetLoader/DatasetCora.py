import zipfile
import torch
import os.path as osp
from torch_geometric.data import InMemoryDataset, download_url, Data

##
# Install ogb
# !pip install ogb
##


class DatasetCora(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DatasetCora, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "cora.zip"

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        download_url("https://linqs-data.soe.ucsc.edu/public/datasets/cora/cora.zip", self.raw_dir)
        

    def process(self):
        with zipfile.ZipFile(osp.join(self.raw_dir, "cora.zip")) as existing_zip:
          existing_zip.extractall(osp.join(self.raw_dir, "ext"))
        with open(osp.join(self.raw_dir, "ext/cora/cora.content")) as f:
          dataset_content = f.read()
        with open(osp.join(self.raw_dir, "ext/cora/cora.cites")) as f:
          dataset_cites = f.read()

        dataset_content_list = dataset_content.split("\n")
        x2 = [-1] * 2708
        paper_id_dict={}
        
        y2 = [-1] * 2708
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(['Neural_Networks', 'Rule_Learning', 'Reinforcement_Learning', 'Probabilistic_Methods', 'Theory', 'Genetic_Algorithms', 'Case_Based'])
        
        for node_id, node in enumerate(dataset_content_list):
          if node.replace(" ","") == "":
            continue
          attr_list = node.split("\t")
          paper_id = int(attr_list[0])
          class_label = attr_list[-1]
          word_attributes = attr_list[1:-1]
          word_attributes2 = [int(i) for i in word_attributes]
          paper_id_dict[paper_id]=node_id
          x2[node_id]=word_attributes2
          y2[node_id] = class_label
          
        class_label_id = le.transform(y2)
          
        x=torch.tensor(x2).to(torch.float)
        y=torch.tensor(class_label_id).to(torch.float)

        dataset_cites_list = dataset_cites.split("\n")
        row = []
        col = []
        for edge in dataset_cites_list:
          if edge.replace(" ","") == "":
            continue
          paper_ids = edge.split("\t")
          row.append(paper_id_dict[int(paper_ids[1])])
          col.append(paper_id_dict[int(paper_ids[0])])
        row = torch.tensor(row).to(torch.long)
        col = torch.tensor(col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        data = Data(x=x, y=y, edge_index=edge_index)

        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
