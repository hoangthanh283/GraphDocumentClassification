import os
import sys
import json
import shutil
import codecs

import xlrd
import argparse
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

sys.setrecursionlimit(1500)
CLASSES = [0, 1, 2, 3]



class BOW(object):
    def __init__(self, corpus):
        self.corpus = corpus

    def transform(self, data):
        bow_fts = [0 for idx in range(len(self.corpus))]

        for char in data:
            idx = self.corpus.find(char)
            if char != -1:
                bow_fts[idx] += 1
        return bow_fts

class DatasetLoader(Dataset):
    """

    """
    def __init__(self, kwargs, transform=True):
        super().__init__()
        self.root_dir = kwargs.get('root_dir', None)
        self.data_path = os.path.join(\
            self.root_dir, kwargs.get('data_file', None))
        self.corpus_path = os.path.join(\
            self.root_dir, kwargs.get('corpus_file', None))
        self.label_path = os.path.join(\
            self.root_dir, kwargs.get('label_file', None))

        self.file_list, self.label_dict, self.corpus = \
            self._load(self.data_path, self.label_path, self.corpus_path)
        self.file_paths = list(map(lambda f: \
            os.path.join(self.root_dir, f), self.file_list))
        
        self.max_node = self._get_max_nodes(self.file_paths)
        self.bow_encoder = BOW(self.corpus)
        self.num_samples = len(self.file_paths)

    def read_image_list(self, path_to_list, prefix=None):
        f = open(path_to_list, 'r')
        filenames = []

        for line in f:
            if line[0] != '#':
                if line[-1] == '\n':
                    filename = line[:-1]
                else:
                    filename = line
                if prefix is not None:
                    filename = prefix + filename
                filenames.append(filename)
            else:
                if len(line) < 3:
                    break
        f.close()
        return filenames

    def get_label_from_excel(self, filename):
        loc = (filename)
        wb = xlrd.open_workbook(loc)
        sheet = wb.sheet_by_index(0)
        sheet.cell_value(0,0)
        dict_file={}

        for i in range(sheet.nrows-1):
            file_name = sheet.cell_value(i+1, 1)
            class_ = sheet.cell_value(i+1, 2)
            dict_file.update({file_name: class_})
        return dict_file

    def encode_onehot(self, label):
        label_onehot = np.identity(\
            len(CLASSES))[CLASSES.index(label)]
        return label_onehot

    def __euclid_distance(self, loc_1, loc_2):
        dis = [0, 0, 0, 0]
        c1 = np.array([loc_1[0], loc_1[1]])
        c2 = np.array([loc_2[0], loc_2[1]])
        e_dis = np.linalg.norm(c1 - c2)

        if c1[0] > c2[0] and c1[1] > c2[1]:
            dis[0] = e_dis
        elif c1[0] < c2[0] and c1[1] > c2[1]:
            dis[1] = e_dis
        elif c1[0] < c2[0] and c1[1] < c2[1]:
            dis[2] = e_dis
        else:
            dis[3] = e_dis
        return dis

    def __hv_distance(self, loc_1, loc_2):
        dis = [0, 0, 0, 0]
        x_dis = abs(loc_1[0] - loc_2[0])
        y_dis = abs(loc_1[1] - loc_2[1])

        if loc_1[0] < loc_2[0]:
            dis[0] = x_dis
        else:
            dis[2] = x_dis

        if loc_1[1] < loc_2[1]:
            dis[1] = y_dis
        else:
            dis[3] = y_dis
        return dis

    def _transform(self, file_path):
        data = {}
        node_fts_txt = []
        node_fts_loc = []
        # edge_lbls = []

        filename = os.path.basename(file_path)
        with codecs.open(file_path, 'r', 'utf-8-sig') as f:
            f_data = json.load(f)
        all_items = f_data.get("lines", [])

        raw_label = int(self.label_dict[os.path.splitext(filename)[0]])
        # graph_lbl = self.encode_onehot(raw_label)
        graph_lbl = raw_label

        for line_item in all_items:
            n_txt = self.bow_encoder.transform(line_item["text"])
            n_loc = line_item["box"] # (x1, y1, x2, y2)
            # if n_loc:
            #     n_loc = [n_loc[0], n_loc[1], n_loc[2]-n_loc[0], n_loc[3]-n_loc[1]]

            node_fts_txt.append(n_txt)
            node_fts_loc.append(n_loc)

        edge_fts = np.zeros((len(node_fts_loc), len(node_fts_loc), 8)) # nodes x nodes x (2 * locations)
        for i, loc_1 in enumerate(node_fts_loc):
            for j, loc_2 in enumerate(node_fts_loc):
                e_fts = []
                e_dis = self.__euclid_distance(loc_1, loc_2)
                hv_dis = self.__hv_distance(loc_1, loc_2)
                e_fts.extend(e_dis)
                e_fts.extend(hv_dis)
                # edge_fts.append(e_fts)
                edge_fts[i, j, :] = e_fts

        node_fts_txt = torch.tensor(node_fts_txt, dtype=torch.float)
        node_fts_txt = torch.cat([node_fts_txt, \
            torch.zeros(self.max_node - len(all_items), len(self.corpus))], dim=0)
        
        node_fts_loc = torch.tensor(node_fts_loc, dtype=torch.float)
        node_fts_loc = torch.cat([node_fts_loc, \
            torch.zeros(self.max_node - len(all_items), 4)], dim=0)

        b_edge_fts = torch.zeros(self.max_node, self.max_node, 8, dtype=torch.float)
        edge_fts = torch.tensor(edge_fts, dtype=torch.float)
        edge_size = list(edge_fts.size())
        b_edge_fts[ :edge_size[0], :edge_size[1], :] = edge_fts

        node_fts_txt = torch.div(node_fts_txt, torch.max(node_fts_txt))
        node_fts_loc = torch.div(node_fts_loc, torch.max(node_fts_loc))
        # edge_fts = torch.div(edge_fts, torch.max(edge_fts))
        b_edge_fts = torch.div(b_edge_fts, torch.max(b_edge_fts))
        graph_lbl = torch.tensor(graph_lbl, dtype=torch.long)

        data = {'node_fts_txt': node_fts_txt,
                'node_fts_loc': node_fts_loc,
                'edge_fts': b_edge_fts,
                'graph_lbl': graph_lbl}

        return data

    def _get_max_nodes(self, file_paths):
        """ Get a max value of all files """ 
        def _get_num_items(_fp):
            with codecs.open(_fp, 'r', 'utf-8-sig') as f:
                f_data = json.load(f)
            return len(f_data.get("lines", []))

        max_num_items = max(list(map(_get_num_items, file_paths)))
        # print("max_num_items: ", max_num_items)
        return max_num_items

    def _load(self, data_path, label_path, corpus_path):        
        with open(corpus_path) as f:
            corpus = f.read()
        file_paths = self.read_image_list(data_path)
        label_dict = self.get_label_from_excel(label_path)
        return (file_paths, label_dict, corpus)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = self._transform(self.file_paths[idx])
        return data


if __name__ == "__main__":
    # Dataloader settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="./data/toyota_data", help='root path of data')
    parser.add_argument('--train_file', type=str, default="train.txt", help='a text file that contains training json label paths')
    parser.add_argument('--val_file', type=str, default="val.txt", help='a text file that contains validating json label paths')
    parser.add_argument('--corpus_file', type=str, default="charset.txt", help='a file that contains all corpus')
    parser.add_argument('--label_file', type=str, default="classification.xlsx", help='a excel file that contains true labels')
    args = parser.parse_args()

    kwargs = {
        'root_dir': args.root_dir,
        'data_file': args.train_file,
        'corpus_file': args.corpus_file,
        'label_file': args.label_file
    }

    train_ds = DatasetLoader(kwargs=kwargs, transform=True)
    train_loader = DataLoader(train_ds, shuffle=True)

    for data in train_loader:
        print("*"*30)
        print(data['node_fts_txt'].size())
        print(data['node_fts_loc'].size())
        print(data['edge_fts'].size())
        print(data['graph_lbl'].size())