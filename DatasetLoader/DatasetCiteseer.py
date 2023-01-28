import zipfile
import torch
import os.path as osp
from torch_geometric.data import InMemoryDataset, download_url, Data

##
# Install ogb
# !pip install ogb
##


class DatasetCiteseer(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DatasetCiteseer, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "citeseer.zip"

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        download_url("https://linqs-data.soe.ucsc.edu/public/datasets/citeseer-doc-classification/citeseer-doc-classification.zip", self.raw_dir)
        

    def process(self):
        with zipfile.ZipFile(osp.join(self.raw_dir, "citeseer-doc-classification.zip")) as existing_zip:
          existing_zip.extractall(osp.join(self.raw_dir, "ext"))
        with open(osp.join(self.raw_dir, "ext/citeseer-doc-classification/citeseer.content")) as f:
          dataset_content = f.read()
        with open(osp.join(self.raw_dir, "ext/citeseer-doc-classification/citeseer.cites")) as f:
          dataset_cites = f.read()

        dataset_content_list = dataset_content.split("\n")
        x2=[]
        paper_id_dict={}
        max_node_id=0
        word_attributes_len=0
        
        y2 = []
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(['Agents', 'IR', 'DB', 'AI', 'HCI', 'ML'])
        
        for node_id, node in enumerate(dataset_content_list):
          if node.replace(" ","") == "":
            continue
          attr_list = node.split("\t")
          paper_id = attr_list[0]
          class_label = attr_list[-1]
          word_attributes = attr_list[1:-1]
          word_attributes2 = [int(i) for i in word_attributes]
          paper_id_dict[paper_id]=node_id
          x2.append(word_attributes2)
          max_node_id=node_id
          word_attributes_len=len(word_attributes2)
          y2.append(class_label)
        

        dataset_cites_list = dataset_cites.split("\n")
        row = []
        col = []
        for edge in dataset_cites_list:
          if edge.replace(" ","") == "":
            continue
          paper_ids = edge.split("\t")
          if paper_ids[1] in paper_id_dict:
            row.append(paper_id_dict[paper_ids[1]])
          else:
            max_node_id+=1
            paper_id_dict[paper_ids[1]]=max_node_id
            row.append(max_node_id)
            x2.append([0]*word_attributes_len)
            y2.append('Agents')
          if paper_ids[0] in paper_id_dict:
            col.append(paper_id_dict[paper_ids[0]])
          else:
            max_node_id+=1
            paper_id_dict[paper_ids[0]]=max_node_id
            col.append(max_node_id)
            x2.append([0]*word_attributes_len)
            y2.append('Agents')

        x=torch.tensor(x2).to(torch.float)

        class_label_id = le.transform(y2)
        y=torch.tensor(class_label_id).to(torch.float)
        
        row = torch.tensor(row).to(torch.long)
        col = torch.tensor(col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        data = Data(x=x, y=y, edge_index=edge_index)

        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
