import zipfile
import torch
import os.path as osp
from torch_geometric.data import InMemoryDataset, download_url, Data

##
# Install ogb
# !pip install ogb
##


class DatasetPubMed(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DatasetPubMed, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "pubmed-diabetes.zip"

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        download_url("https://linqs-data.soe.ucsc.edu/public/datasets/pubmed-diabetes/pubmed-diabetes.zip", self.raw_dir)
        

    def process(self):
        with zipfile.ZipFile(osp.join(self.raw_dir, "pubmed-diabetes.zip")) as existing_zip:
          existing_zip.extractall(osp.join(self.raw_dir, "ext"))
        with open(osp.join(self.raw_dir, "ext/pubmed-diabetes/data/Pubmed-Diabetes.NODE.paper.tab")) as f:
          dataset_content = f.read()
        with open(osp.join(self.raw_dir, "ext/pubmed-diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab")) as f:
          dataset_cites = f.read()

        dataset_content_list = dataset_content.split("\n")
        x2 = []
        y2 = []
        class_labels = list(set([ x.split("\t")[1].split('=')[1] for x in dataset_content_list[2:] if x.replace(" ","") != ""]))
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(class_labels)

        paper_id_dict={}
        vecidx = []
        for node_idx, node in enumerate(dataset_content_list):
          if node.replace(" ","") == "":
            continue

          attr_list = node.split("\t")
          if node_idx == 1:
              for attr in attr_list[1:-1]:
                  vecidx.append(attr.split(':')[1])
          elif node_idx != 0:
              paper_id = int(attr_list[0])
              label_id = attr_list[1].split('=')[1]
              y2.append(label_id)
              word_attributes = [0.0] * 500
              for col in attr_list[2:-1]:
                  word_attributes[vecidx.index(col.split('=')[0])] = float(col.split('=')[1])
              paper_id_dict[paper_id] = node_idx - 2
              x2.append(word_attributes)
        x=torch.tensor(x2).to(torch.float)
        class_label_id = le.transform(y2)
        y = torch.tensor(class_label_id).to(torch.float)

        dataset_cites_list = dataset_cites.split("\n")
        row = []
        col = []
        for edge in dataset_cites_list[2:]:
          if edge.replace(" ","") == "":
            continue
          paper_ids = edge.split("\t")
          row.append(paper_id_dict[int(paper_ids[1].split(':')[1])])
          col.append(paper_id_dict[int(paper_ids[3].split(':')[1])])
        row = torch.tensor(row).to(torch.long)
        col = torch.tensor(col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        data = Data(x=x, y=y, edge_index=edge_index)

        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
