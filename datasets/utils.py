#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    utils.py: Functions to process dataset graphs.

    Usage:

"""

from __future__ import print_function

import rdkit
import torch
from joblib import Parallel, delayed
import multiprocessing
import networkx as nx
import numpy as np
import shutil
import os
import json
from skimage.metrics import structural_similarity as ssim
__author__ = "chen shao"
__email__ = "chen.shao@student.kit.edu"


def qm9_nodes(graphs, hydrogen=False):
    h = []
    for n, d in graphs.nodes(data=True):
        h_t = []
        # Add Supernode Type 
        # h_t += [int(d["NodeType"] == "Supernode")]
        h_t += [i for i, x in enumerate(['None', "Supernode"]) if d["NodeType"] == x]
        # Atom type (One-hot H, C, N, O F)
        h_t += [int(d['AtomSymbol'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'Br', 'Cl']]
        # Atomic number
        h_t.append(d['NumAtomic'])
        # Acceptor
        h_t.append(d['acceptor'])
        # Donor
        h_t.append(d['donor'])
        # Aromatic
        h_t.append(int(d['IsAromatic']))
        # FormalCharge
        h_t.append(d["FormalCharge"])
        # NumExplicit
        h_t.append(d["NumExplicit"])
        # NumImplicit
        h_t.append(d["NumImplicit"])
        # ChiralTag
        h_t+=[i for i, x in enumerate([None, rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW]) if d["ChiralTag"] == x]
        # Hybradization
        h_t+=[i for i, x in enumerate([None, rdkit.Chem.rdchem.HybridizationType.OTHER, rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED, rdkit.Chem.rdchem.HybridizationType.SP, rdkit.Chem.rdchem.HybridizationType.SP2, rdkit.Chem.rdchem.HybridizationType.SP3, rdkit.Chem.rdchem.HybridizationType.SP3D, rdkit.Chem.rdchem.HybridizationType.SP3D2]) if d['Hybridization'] == x]
        # If number hydrogen is used as a
        if hydrogen:
            h_t.append(d['TotalNum'])
        h.append(h_t)
    return h
# for keys in graphs.nodes(data=True)[0].keys():  
#     for n, d in graphs.nodes(data=True): 
#         print(d[keys])

def qm9_edges(g, e_representation='raw_distance'):
    remove_edges = []
    e={}    
    for n1, n2, d in g.edges(data=True):
        e_t = []
        # Raw distance function
        if e_representation == 'chem_graph':
            if d['BondType'] is None:
                remove_edges += [(n1, n2)]
            else:
                e_t += [i+1 for i, x in enumerate([rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                                                rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC])
                        if x == d['b_type']]
        elif e_representation == 'distance_bin':
            if d['BondType'] is None:
                step = (6-2)/8.0
                start = 2
                b = 9
                for i in range(0, 9):
                    if d['distance'] < (start+i*step):
                        b = i
                        break
                e_t.append(b+5)
            else:
                e_t += [i+1 for i, x in enumerate([rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                                                   rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC])
                        if x == d['BondType']]
        elif e_representation == 'raw_distance':
            if d['BondType'] is None:
                remove_edges += [(n1, n2)]
            else:
                e_t.append(d['distance'])
                e_t += [int(d['BondType'] == x) for x in ["Supernode", rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                                                        rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC]]
                e_t += [int(d['StereoType'] == x) for x in [rdkit.Chem.rdchem.BondStereo.STEREONONE, rdkit.Chem.rdchem.BondStereo.STEREOE,
                                                        rdkit.Chem.rdchem.BondStereo.STEREOZ]]

        else:
            print('Incorrect Edge representation transform')
            quit()
        if e_t:
            e[(n1, n2)] = e_t
    for edg in remove_edges:
        g.remove_edge(*edg)
    #return nx.to_numpy_matrix(g), e
    return nx.to_numpy_matrix(g), e


def target_encoder(target):
    return np.array([target[-1]])
    

def normalize_data(data, mean, std):
    return (data-mean)/std


def get_values(obj, start, end, prop):
    vals = []
    for i in range(start, end):
        v = {}
        if 'degrees' in prop:
            v['degrees'] = set(sum(obj[i][0][0].sum(axis=0, dtype='int').tolist(), []))
        if 'edge_labels' in prop:
            v['edge_labels'] = set(sum(list(obj[i][0][2].values()), []))
        if 'target_mean' in prop or 'target_std' in prop:
            v['params'] = obj[i][1]
        vals.append(v)
    return vals


def get_graph_stats(graph_obj_handle, prop='degrees'):
    # if prop == 'degrees':
    num_cores = multiprocessing.cpu_count()
    inputs = [int(i*len(graph_obj_handle)/num_cores) for i in range(num_cores)] + [len(graph_obj_handle)]
    res = Parallel(n_jobs=num_cores)(delayed(get_values)(graph_obj_handle, inputs[i], inputs[i+1], prop) for i in range(num_cores))

    stat_dict = {}

    if 'degrees' in prop:
        stat_dict['degrees'] = list(set([d for core_res in res for file_res in core_res for d in file_res['degrees']]))
    if 'edge_labels' in prop:
        stat_dict['edge_labels'] = list(set([d for core_res in res for file_res in core_res for d in file_res['edge_labels']]))
    if 'target_mean' in prop or 'target_std' in prop:
        param = np.array([file_res['params'] for core_res in res for file_res in core_res])
    if 'target_mean' in prop:
        stat_dict['target_mean'] = np.mean(param.astype(np.float), axis=0)
    if 'target_std' in prop:
        stat_dict['target_std'] = np.std(param.astype(np.float), axis=0)
    return stat_dict


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred = pred.type_as(target)
    target = target.type_as(pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def collate_g(batch):

    batch_sizes = np.max(
        np.array([[len(input_b[1]), len(input_b[1][0]), len(input_b[2]),
                                len(list(input_b[2].values())[0])]
                                if input_b[2] else [len(input_b[1]), len(input_b[1][0]), 0,0]
                                for (input_b, target_b) in batch]), axis=0) # is target b here a scala or a vector

    g = np.zeros((len(batch), batch_sizes[0], batch_sizes[0]))
    h = np.zeros((len(batch), batch_sizes[0], batch_sizes[1]))
    e = np.zeros((len(batch), batch_sizes[0], batch_sizes[0], batch_sizes[3]))

    target = np.zeros((len(batch), len(np.asarray([batch[0][1]]))))

    for i in range(len(batch)):

        num_nodes = len(batch[i][0][1])

        # Adjacency matrix
        g[i, 0:num_nodes, 0:num_nodes] = batch[i][0][0]

        # Node features
        h[i, 0:num_nodes, :] = batch[i][0][1] # last four atoms has different dimensions for node feature why 

        # Edges
        for edge in batch[i][0][2].keys():
            e[i, edge[0], edge[1], :] = batch[i][0][2][edge]
            e[i, edge[1], edge[0], :] = batch[i][0][2][edge]

        # Target
        target[i, :] = batch[i][1]

    g = torch.FloatTensor(g)
    h = torch.FloatTensor(h)
    e = torch.FloatTensor(e)
    target = torch.FloatTensor(target)

    return g, h, e, target


def save_checkpoint(state, is_best, directory):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_model_file)


def save2dict(graph):
    dict_graph = {}
    for i, d in graph.nodes(data=True):
        dict_graph.update({str(i):d})
    with open("plots/current_graph.json", "w") as write_file:
        json.dump(dict_graph, write_file, indent=4)


def add_supernode(g):
    g = nx.disjoint_union(g[0] ,g[1])

    supernode_index = len(g.nodes())
    g.add_node(supernode_index, 
        NodeType="Supernode",
        AtomSymbol = "None", 
        NumAtomic = 0, 
        acceptor= 0, 
        donor= 0,
        IsAromatic= 0, 
        FormalCharge = 0,
        NumExplicit = 0,
        NumImplicit = 0,
        ChiralTag = None,
        Hybridization = None,
        TotalNum= 0,
        coord=np.zeros((1, 3))) 
    # add supernode bond type 
    for n, dn in g.nodes(data=True):
        for m, dm in g.nodes(data=True):
            if dn["NodeType"] == 'Supernode' or dm["NodeType"] == 'Supernode':
                g.add_edge(n, m, BondType="Supernode", StereoType=None,
                    distance=np.linalg.norm(g.nodes[n]['coord']-g.nodes[m]['coord']))
    g.remove_edge(*(supernode_index, supernode_index))
    return g

def save_checkpoint(model, optimizer, epoch, filename, root="./checkpoints"):
    if not os.path.isdir(root):
        os.makedirs(root)

    fpath = os.path.join(root, filename)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }
        , fpath)

class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg

def compute_errors(pred, target):
    # linear version
    # can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    # torch.mean(torch.abs(output - target) / torch.abs(target))
    thresh = torch.maximum(torch.divide(target, pred), torch.divide(pred, target))
    a1 = torch.mul((thresh < 0.25), thresh).mean()
    a2 = (torch.mul((thresh < 0.25), thresh)** 2).mean()
    a3 = (torch.mul((thresh < 0.25), thresh)** 3).mean()
    abs_rel = torch.mean(torch.divide(torch.abs(target - pred) ,target))
    sq_rel = torch.mean(((target - pred) ** 2) / target)

    rmse = (target - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    # log version
    thresh_log = torch.maximum(torch.exp(torch.log(target)- torch.log(pred)), torch.exp(torch.log(pred) - torch.log(target)))
    a1_stable = torch.mul((thresh_log < 0.25), thresh_log).mean() 
    a2_stable = (torch.mul((thresh_log < 0.25), thresh_log)** 2).mean()
    a3_stable = (torch.mul((thresh_log < 0.25), thresh_log)** 2).mean()
    abs_rel_stable = torch.mean(torch.exp(torch.log(torch.abs(target - pred)) - torch.log(target)))
     
    rmse_log = (torch.log(target) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    err = torch.log(pred) - torch.log(target)
    silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100

    log_10 = (torch.abs(torch.log10(target) - torch.log10(pred))).mean()
    
    
    gt_ori = torch.divide(target - torch.min(target), torch.max(target)-torch.min(target))
    pred_ori = torch.divide(pred - torch.min(pred), torch.max(pred)- torch.min(pred))

    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log, silog=silog, sq_rel=sq_rel, a1_stable = a1_stable, a2_stable=a2_stable, a3_stable=a3_stable, abs_rel_stable=abs_rel_stable)

def is_rank_zero(args):
    return args.rank == 0

def log_images(params):
    raise NotImplementedError
    wandb.log(
        {
            "Input": [wandb.Image(img)],
            "GT": [wandb.Image(depth)],
            "Prediction": [wandb.Image(pred)]
        }, step=step)

# Parser check
def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
    return x